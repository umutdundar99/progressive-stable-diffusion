"""
Evaluation pipeline for diffusion models with FID, IS, LPIPS, and class-conditional metrics.

This module generates samples from a trained diffusion model and computes:
- FID (Fréchet Inception Distance): Measures distribution similarity
- IS (Inception Score): Measures quality and diversity
- LPIPS (Learned Perceptual Image Patch Similarity): Measures perceptual diversity
- SSIM (Structural Similarity): Measures structural similarity
- Class-conditional FID: Per-class FID scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.models.diffusion_module import DiffusionModule

# Optional imports for metrics (will check availability)
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("Warning: torchmetrics not installed. Install with: pip install torchmetrics[image]")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed. Install with: pip install lpips")

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not installed. Install with: pip install scikit-image")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained diffusion model using FID, IS, LPIPS, and other metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a Lightning checkpoint produced during training.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Hydra-compatible config file describing the training setup.",
    )
    parser.add_argument(
        "--real-data-dir",
        type=Path,
        required=True,
        help="Directory containing real images organized by class (e.g., train/0, train/1, ...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory where evaluation results and generated images are saved.",
    )
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=100,
        help="Number of samples to generate per class for evaluation.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of diffusion steps used during sampling (for DDIM).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["ddpm", "ddim"],
        default="ddim",
        help="Sampling method to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation and metric computation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device used for inference (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images to disk.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of classes (MES scores 0-3).",
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_config(config_path: Path) -> DictConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    return OmegaConf.load(config_path)


class RealImageDataset(Dataset):
    """Dataset for loading real images for FID computation."""
    
    def __init__(self, root: str, image_size: int = 256):
        self.dataset = ImageFolder(
            root=root,
            transform=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.dataset[idx]


def _ddim_sample_batch(
    module: DiffusionModule,
    labels: Tensor,
    sampling_steps: int,
    device: torch.device,
    eta: float = 0.0,
) -> Tensor:
    """DDIM sampling for a batch of labels."""
    num_samples = labels.shape[0]
    height = module.cfg.dataset.image_size
    T = module.diff_cfg.num_train_timesteps
    guidance_scale = module.diff_cfg.guidance_scale

    latents = torch.randn(
        num_samples,
        module.cfg.model.latent_channels,
        height // 8,
        height // 8,
        device=device,
        dtype=torch.float32,
    )

    alphas_cumprod = module.alphas_cumprod

    timesteps = torch.linspace(
        T - 1,
        0,
        steps=sampling_steps,
        dtype=torch.long,
        device=device,
    )

    # FIX: Pass is_training=False during inference
    con_embed = module.ordinal_embedder(labels, is_training=False, unconditional=False)
    uncond_embed = module.ordinal_embedder(labels, is_training=False, unconditional=True)
    embed = torch.cat([uncond_embed, con_embed], dim=0)

    for i, t_scalar in enumerate(timesteps):
        t_int = int(t_scalar.item())
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=device)

        latent_model_input = torch.cat([latents, latents], dim=0)
        t_model_input = torch.cat([t, t], dim=0)

        noise_pred = module(latent_model_input, t_model_input, embed)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        eps_theta = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        alpha_bar_t = alphas_cumprod[t_int].to(device=latents.device, dtype=latents.dtype)

        if i == sampling_steps - 1:
            alpha_bar_prev = torch.tensor(1.0, device=latents.device, dtype=latents.dtype)
        else:
            t_prev_int = int(timesteps[i + 1].item())
            alpha_bar_prev = alphas_cumprod[t_prev_int].to(device=latents.device, dtype=latents.dtype)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x0_pred = (latents - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t

        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

        if eta == 0.0:
            latents = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * eps_theta
        else:
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(latents)
            latents = (
                sqrt_alpha_bar_prev * x0_pred
                + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_theta
                + sigma * noise
            )

    return latents


def _latents_to_images(module: DiffusionModule, latents: Tensor) -> Tensor:
    """Decode latents with the frozen SD VAE and map to [0, 1] RGB."""
    with torch.no_grad():
        scaled = latents / module.diff_cfg.latent_scale
        decoded = module.vae.decode(scaled)

    if hasattr(decoded, "sample"):
        images = decoded.sample
    else:
        images = decoded

    images = images.clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0
    return images.clamp(0.0, 1.0)


def generate_samples_for_class(
    module: DiffusionModule,
    class_label: float,
    num_samples: int,
    batch_size: int,
    sampling_steps: int,
    device: torch.device,
) -> List[Tensor]:
    """Generate samples for a specific class label."""
    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        labels = torch.full((current_batch_size,), class_label, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            latents = _ddim_sample_batch(module, labels, sampling_steps, device)
            images = _latents_to_images(module, latents)
        
        all_images.append(images.cpu())
    
    return torch.cat(all_images, dim=0)[:num_samples]


def compute_fid(
    real_images: Tensor,
    fake_images: Tensor,
    device: torch.device,
) -> float:
    """Compute FID between real and fake images."""
    if not HAS_TORCHMETRICS:
        print("Skipping FID: torchmetrics not installed")
        return -1.0
    
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Images should be in [0, 1] range and uint8 format for FID
    real_uint8 = (real_images * 255).to(torch.uint8)
    fake_uint8 = (fake_images * 255).to(torch.uint8)
    
    # Process in batches to avoid OOM
    batch_size = 32
    for i in range(0, len(real_uint8), batch_size):
        batch = real_uint8[i:i+batch_size].to(device)
        fid.update(batch, real=True)
    
    for i in range(0, len(fake_uint8), batch_size):
        batch = fake_uint8[i:i+batch_size].to(device)
        fid.update(batch, real=False)
    
    return fid.compute().item()


def compute_inception_score(
    images: Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute Inception Score for generated images."""
    if not HAS_TORCHMETRICS:
        print("Skipping IS: torchmetrics not installed")
        return -1.0, -1.0
    
    inception = InceptionScore(normalize=True).to(device)
    
    images_uint8 = (images * 255).to(torch.uint8)
    
    batch_size = 32
    for i in range(0, len(images_uint8), batch_size):
        batch = images_uint8[i:i+batch_size].to(device)
        inception.update(batch)
    
    mean, std = inception.compute()
    return mean.item(), std.item()


def compute_lpips_diversity(
    images: Tensor,
    device: torch.device,
    num_pairs: int = 100,
) -> float:
    """Compute average LPIPS distance between random pairs (measures diversity)."""
    if not HAS_LPIPS:
        print("Skipping LPIPS: lpips not installed")
        return -1.0
    
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    
    n = len(images)
    if n < 2:
        return 0.0
    
    # Random pairs
    indices = torch.randperm(n)[:min(num_pairs * 2, n)]
    if len(indices) % 2 == 1:
        indices = indices[:-1]
    
    pairs_a = indices[:len(indices)//2]
    pairs_b = indices[len(indices)//2:]
    
    distances = []
    with torch.no_grad():
        for a, b in zip(pairs_a, pairs_b):
            img_a = images[a:a+1].to(device) * 2 - 1  # [0,1] -> [-1,1]
            img_b = images[b:b+1].to(device) * 2 - 1
            dist = lpips_fn(img_a, img_b)
            distances.append(dist.item())
    
    return np.mean(distances)


def compute_ssim_to_real(
    real_images: Tensor,
    fake_images: Tensor,
    num_comparisons: int = 100,
) -> float:
    """Compute average SSIM between generated and random real images."""
    if not HAS_SKIMAGE:
        print("Skipping SSIM: scikit-image not installed")
        return -1.0
    
    n_real = len(real_images)
    n_fake = len(fake_images)
    
    ssim_values = []
    for _ in range(min(num_comparisons, n_fake)):
        real_idx = np.random.randint(0, n_real)
        fake_idx = np.random.randint(0, n_fake)
        
        real_np = real_images[real_idx].permute(1, 2, 0).numpy()
        fake_np = fake_images[fake_idx].permute(1, 2, 0).numpy()
        
        ssim_val = ssim(real_np, fake_np, multichannel=True, channel_axis=2, data_range=1.0)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def save_images(
    images: Tensor,
    labels: List[float],
    output_dir: Path,
    prefix: str = "gen",
) -> None:
    """Save generated images to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (image, label) in enumerate(zip(images, labels)):
        array = (image.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
        pil_image = Image.fromarray(array)
        filename = output_dir / f"{prefix}_class{int(label)}_{idx:04d}.png"
        pil_image.save(filename)


def load_real_images_for_class(
    data_dir: Path,
    class_idx: int,
    image_size: int,
    max_images: int = 1000,
) -> Tensor:
    """Load real images for a specific class."""
    class_dir = data_dir / str(class_idx)
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    images = []
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    for img_path in class_dir.iterdir():
        if img_path.suffix.lower() in extensions and len(images) < max_images:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
    
    if not images:
        raise ValueError(f"No images found in {class_dir}")
    
    return torch.stack(images)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)
    
    cfg = _load_config(args.config)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    module = DiffusionModule.load_from_checkpoint(
        str(args.checkpoint),
        cfg=cfg,
    )
    module = module.to(device)
    module = module.to(torch.float32)
    module.eval()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results dictionary
    results = {
        "checkpoint": str(args.checkpoint),
        "num_samples_per_class": args.num_samples_per_class,
        "sampling_steps": args.sampling_steps,
        "sampler": args.sampler,
        "seed": args.seed,
        "metrics": {},
    }
    
    all_fake_images = []
    all_real_images = []
    all_labels = []
    
    print("\n=== Generating samples and computing metrics ===")
    
    # Per-class metrics
    for class_idx in range(args.num_classes):
        print(f"\nProcessing class {class_idx}...")
        
        # Generate fake images
        print(f"  Generating {args.num_samples_per_class} samples...")
        fake_images = generate_samples_for_class(
            module=module,
            class_label=float(class_idx),
            num_samples=args.num_samples_per_class,
            batch_size=args.batch_size,
            sampling_steps=args.sampling_steps,
            device=device,
        )
        
        # Load real images
        print(f"  Loading real images...")
        try:
            real_images = load_real_images_for_class(
                data_dir=args.real_data_dir,
                class_idx=class_idx,
                image_size=cfg.dataset.image_size,
                max_images=args.num_samples_per_class * 2,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  Warning: {e}")
            real_images = None
        
        # Save images if requested
        if args.save_images:
            save_images(
                fake_images,
                [float(class_idx)] * len(fake_images),
                args.output_dir / f"class_{class_idx}",
                prefix="gen",
            )
        
        # Compute per-class metrics
        class_metrics = {}
        
        if real_images is not None:
            print(f"  Computing FID...")
            class_fid = compute_fid(real_images, fake_images, device)
            class_metrics["fid"] = class_fid
            print(f"    FID: {class_fid:.2f}")
            
            print(f"  Computing SSIM...")
            class_ssim = compute_ssim_to_real(real_images, fake_images)
            class_metrics["ssim"] = class_ssim
            print(f"    SSIM: {class_ssim:.4f}")
            
            all_real_images.append(real_images)
        
        print(f"  Computing Inception Score...")
        is_mean, is_std = compute_inception_score(fake_images, device)
        class_metrics["inception_score_mean"] = is_mean
        class_metrics["inception_score_std"] = is_std
        print(f"    IS: {is_mean:.2f} ± {is_std:.2f}")
        
        print(f"  Computing LPIPS diversity...")
        lpips_div = compute_lpips_diversity(fake_images, device)
        class_metrics["lpips_diversity"] = lpips_div
        print(f"    LPIPS diversity: {lpips_div:.4f}")
        
        results["metrics"][f"class_{class_idx}"] = class_metrics
        
        all_fake_images.append(fake_images)
        all_labels.extend([float(class_idx)] * len(fake_images))
    
    # Compute overall metrics
    print("\n=== Computing overall metrics ===")
    
    all_fake = torch.cat(all_fake_images, dim=0)
    
    if all_real_images:
        all_real = torch.cat(all_real_images, dim=0)
        
        print("Computing overall FID...")
        overall_fid = compute_fid(all_real, all_fake, device)
        results["metrics"]["overall_fid"] = overall_fid
        print(f"  Overall FID: {overall_fid:.2f}")
        
        print("Computing overall SSIM...")
        overall_ssim = compute_ssim_to_real(all_real, all_fake)
        results["metrics"]["overall_ssim"] = overall_ssim
        print(f"  Overall SSIM: {overall_ssim:.4f}")
    
    print("Computing overall Inception Score...")
    overall_is_mean, overall_is_std = compute_inception_score(all_fake, device)
    results["metrics"]["overall_inception_score_mean"] = overall_is_mean
    results["metrics"]["overall_inception_score_std"] = overall_is_std
    print(f"  Overall IS: {overall_is_mean:.2f} ± {overall_is_std:.2f}")
    
    print("Computing overall LPIPS diversity...")
    overall_lpips = compute_lpips_diversity(all_fake, device, num_pairs=200)
    results["metrics"]["overall_lpips_diversity"] = overall_lpips
    print(f"  Overall LPIPS diversity: {overall_lpips:.4f}")
    
    # Save results
    results_path = args.output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    if "overall_fid" in results["metrics"]:
        print(f"Overall FID: {results['metrics']['overall_fid']:.2f}")
    print(f"Overall IS: {results['metrics']['overall_inception_score_mean']:.2f} ± {results['metrics']['overall_inception_score_std']:.2f}")
    print(f"Overall LPIPS Diversity: {results['metrics']['overall_lpips_diversity']:.4f}")
    if "overall_ssim" in results["metrics"]:
        print(f"Overall SSIM: {results['metrics']['overall_ssim']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
