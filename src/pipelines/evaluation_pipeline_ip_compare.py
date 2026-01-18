"""
Evaluation Pipeline for Comparing IP-Adapter Model Checkpoints.

This module generates samples from trained IP-Adapter diffusion models and computes:
- FID (Fréchet Inception Distance): Measures distribution similarity
- IS (Inception Score): Measures quality and diversity
- LPIPS (Learned Perceptual Image Patch Similarity): Measures perceptual diversity
- SSIM (Structural Similarity): Measures structural similarity

The pipeline compares multiple checkpoints with different guidance scales and
generates a comprehensive comparison report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision import transforms

from src.models.diffusion_module_ip import DiffusionModuleWithIP


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, Path):
        return str(obj)
    return obj


# Optional imports for metrics
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    # Test if torch-fidelity is installed by trying to create instance
    _ = FrechetInceptionDistance(feature=2048, normalize=True)
    HAS_TORCHMETRICS = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_TORCHMETRICS = False
    print(
        f"Warning: torchmetrics/torch-fidelity not fully installed: {e}\n"
        "Install with: pip install torchmetrics[image] torch-fidelity"
    )

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


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""

    checkpoint_path: str
    checkpoint_name: str
    guidance_scales: List[float]
    num_samples_per_class: int
    sampling_steps: int
    batch_size: int
    num_classes: int
    image_scale: float = 1.0
    eta: float = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare IP-Adapter model checkpoints with multiple guidance scales."
    )
    parser.add_argument(
        "--checkpoint1",
        type=Path,
        default=Path(
            "/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/kwd2qy49/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
        ),
        help="Path to first checkpoint.",
    )
    parser.add_argument(
        "--checkpoint2",
        type=Path,
        default=Path(
            "/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/pvq7gpe7/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
        ),
        help="Path to second checkpoint.",
    )
    parser.add_argument(
        "--checkpoint3",
        type=Path,
        default=Path(
            "/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/ejfpqk2s/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
        ),
        help="Path to third checkpoint (no blur, uniform weights).",
    )
    parser.add_argument(
        "--name1",
        type=str,
        default="model_kwd2qy49",
        help="Name for first model (for reporting).",
    )
    parser.add_argument(
        "--name2",
        type=str,
        default="model_pvq7gpe7",
        help="Name for second model (for reporting).",
    )
    parser.add_argument(
        "--name3",
        type=str,
        default="model_ejfpqk2s",
        help="Name for third model (for reporting).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_ip.yaml"),
        help="Config file for IP-Adapter training setup.",
    )
    parser.add_argument(
        "--real-data-dir",
        type=Path,
        default=Path(
            "/home/umut_dundar/repositories/progressive-stable-diffusion/data/limuc/processed_data_scale1/test"
        ),
        help="Directory containing real test images organized by class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation_ip_compare"),
        help="Directory where evaluation results are saved.",
    )
    parser.add_argument(
        "--guidance-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5],
        help="List of guidance scales to evaluate.",
    )
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=50,
        help="Number of samples to generate per class.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of classes (MES scores 0-3).",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images to disk.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode with fewer samples and guidance scales.",
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


def load_model(
    checkpoint_path: Path,
    config: DictConfig,
    device: torch.device,
    frequency_dominant_scale: float = 1.0,
    frequency_non_dominant_scale: float = 1.0,
) -> DiffusionModuleWithIP:
    """Load IP-Adapter model from checkpoint with specific frequency scales.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Base config (will be modified with frequency scales)
        device: Target device
        frequency_dominant_scale: Scale for high-resolution layers
        frequency_non_dominant_scale: Scale for low-resolution layers
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"  frequency_dominant_scale: {frequency_dominant_scale}")
    print(f"  frequency_non_dominant_scale: {frequency_non_dominant_scale}")

    # Create a copy of config and set the correct frequency scales
    config_copy = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_copy.model.frequency_dominant_scale = frequency_dominant_scale
    config_copy.model.frequency_non_dominant_scale = frequency_non_dominant_scale

    module = DiffusionModuleWithIP.load_from_checkpoint(
        str(checkpoint_path),
        cfg=config_copy,
        strict=False,
    )
    module = module.to(device)
    module = module.to(torch.float32)
    module.eval()
    return module


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

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    images = []
    extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    for img_path in sorted(class_dir.iterdir()):
        if img_path.suffix.lower() in extensions and len(images) < max_images:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"  Warning: Could not load {img_path}: {e}")

    if not images:
        raise ValueError(f"No images found in {class_dir}")

    return torch.stack(images)


def _ddim_sample_unconditioned(
    module: DiffusionModuleWithIP,
    labels: Tensor,
    sampling_steps: int,
    device: torch.device,
    eta: float = 0.0,
    guidance_scale: float = 0.0,
) -> Tensor:
    """
    DDIM sampling WITHOUT image conditioning (unconditioned generation).

    For comparing models on their ability to generate realistic images
    based only on class labels (MES scores).

    Args:
        module: DiffusionModuleWithIP
        labels: Mayo scores (B,)
        sampling_steps: Number of DDIM steps
        device: Target device
        eta: DDIM stochasticity
        guidance_scale: CFG scale for class conditioning

    Returns:
        Generated latents (B, 4, H//8, W//8)
    """
    num_samples = labels.shape[0]
    height = module.cfg.dataset.image_size
    T = module.diff_cfg.num_train_timesteps

    # Initialize latents
    latents = torch.randn(
        num_samples,
        module.cfg.model.latent_channels,
        height // 8,
        height // 8,
        device=device,
        dtype=torch.float32,
    )

    alphas_cumprod = module.alphas_cumprod

    # DDIM timestep schedule
    timesteps = torch.linspace(
        T - 1,
        0,
        steps=sampling_steps,
        dtype=torch.long,
        device=device,
    )

    # Prepare conditioning embeddings (no image, only AOE for class)
    # AOE embeddings for class conditioning
    aoe_cond = module.ordinal_embedder(labels, is_training=False)
    if aoe_cond.dim() == 2:
        aoe_cond = aoe_cond.unsqueeze(1)  # (B, 1, D)

    # Unconditional AOE (for CFG)
    aoe_uncond = module.ordinal_embedder.get_negative_embedding(
        labels, is_training=False
    )
    if aoe_uncond.dim() == 2:
        aoe_uncond = aoe_uncond.unsqueeze(1)  # (B, 1, D)

    # Zero image embeddings (unconditioned on image)
    num_tokens = module.diff_cfg.num_image_tokens
    cross_dim = module.cfg.model.conditioning_dim
    zero_image_embeds = torch.zeros(
        num_samples, num_tokens, cross_dim, device=device, dtype=aoe_cond.dtype
    )

    # Combine: [AOE, Zero Image]
    cond_embed = torch.cat([aoe_cond, zero_image_embeds], dim=1)
    uncond_embed = torch.cat([aoe_uncond, zero_image_embeds], dim=1)

    # CFG: concat unconditional + conditional
    embed = torch.cat([uncond_embed, cond_embed], dim=0)  # (2*B, seq, D)

    # DDIM loop
    for i, t_scalar in enumerate(timesteps):
        t_int = int(t_scalar.item())
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=device)

        # Duplicate latents for CFG
        latent_model_input = torch.cat([latents, latents], dim=0)
        t_model_input = torch.cat([t, t], dim=0)

        # Predict noise
        noise_pred = module(latent_model_input, t_model_input, embed)

        # CFG
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        if guidance_scale > 0:
            eps_theta = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            # No guidance - use only conditional prediction
            eps_theta = noise_pred_cond

        # DDIM update
        alpha_bar_t = alphas_cumprod[t_int].to(device=device, dtype=latents.dtype)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # Predict x0
        x0_pred = (latents - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
        x0_pred = x0_pred.clamp(-4.0, 4.0)

        # Final step
        if i == sampling_steps - 1:
            latents = x0_pred
            continue

        # Get alpha_bar for previous timestep
        t_prev_int = int(timesteps[i + 1].item())
        alpha_bar_prev = alphas_cumprod[t_prev_int].to(
            device=device, dtype=latents.dtype
        )
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

        if eta == 0.0:
            # Deterministic DDIM
            latents = (
                sqrt_alpha_bar_prev * x0_pred
                + sqrt_one_minus_alpha_bar_prev * eps_theta
            )
        else:
            # Stochastic DDIM
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar_t + 1e-8)
                * (1 - alpha_bar_t / alpha_bar_prev + 1e-8)
            )
            noise = torch.randn_like(latents)
            latents = (
                sqrt_alpha_bar_prev * x0_pred
                + torch.sqrt(1 - alpha_bar_prev - sigma**2 + 1e-8) * eps_theta
                + sigma * noise
            )

    return latents


def _latents_to_images(module: DiffusionModuleWithIP, latents: Tensor) -> Tensor:
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
    module: DiffusionModuleWithIP,
    class_label: float,
    num_samples: int,
    batch_size: int,
    sampling_steps: int,
    device: torch.device,
    guidance_scale: float = 0.0,
    eta: float = 0.0,
) -> Tensor:
    """Generate samples for a specific class label without image conditioning."""
    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        labels = torch.full(
            (current_batch_size,), class_label, device=device, dtype=torch.float32
        )

        with torch.no_grad():
            latents = _ddim_sample_unconditioned(
                module=module,
                labels=labels,
                sampling_steps=sampling_steps,
                device=device,
                eta=eta,
                guidance_scale=guidance_scale,
            )
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
        return -1.0

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    real_uint8 = (real_images * 255).to(torch.uint8)
    fake_uint8 = (fake_images * 255).to(torch.uint8)

    batch_size = 32
    for i in range(0, len(real_uint8), batch_size):
        batch = real_uint8[i : i + batch_size].to(device)
        fid.update(batch, real=True)

    for i in range(0, len(fake_uint8), batch_size):
        batch = fake_uint8[i : i + batch_size].to(device)
        fid.update(batch, real=False)

    return fid.compute().item()


def compute_inception_score(
    images: Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute Inception Score for generated images."""
    if not HAS_TORCHMETRICS:
        return -1.0, -1.0

    inception = InceptionScore(normalize=True).to(device)
    images_uint8 = (images * 255).to(torch.uint8)

    batch_size = 32
    for i in range(0, len(images_uint8), batch_size):
        batch = images_uint8[i : i + batch_size].to(device)
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
        return -1.0

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    n = len(images)
    if n < 2:
        return 0.0

    indices = torch.randperm(n)[: min(num_pairs * 2, n)]
    if len(indices) % 2 == 1:
        indices = indices[:-1]

    pairs_a = indices[: len(indices) // 2]
    pairs_b = indices[len(indices) // 2 :]

    distances = []
    with torch.no_grad():
        for a, b in zip(pairs_a, pairs_b):
            img_a = images[a : a + 1].to(device) * 2 - 1
            img_b = images[b : b + 1].to(device) * 2 - 1
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
        return -1.0

    n_real = len(real_images)
    n_fake = len(fake_images)

    ssim_values = []
    for _ in range(min(num_comparisons, n_fake)):
        real_idx = np.random.randint(0, n_real)
        fake_idx = np.random.randint(0, n_fake)

        real_np = real_images[real_idx].permute(1, 2, 0).numpy()
        fake_np = fake_images[fake_idx].permute(1, 2, 0).numpy()

        ssim_val = ssim(
            real_np, fake_np, multichannel=True, channel_axis=2, data_range=1.0
        )
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)


def save_images(
    images: Tensor,
    output_dir: Path,
    class_idx: int,
    guidance_scale: float,
    model_name: str,
) -> None:
    """Save generated images to disk."""
    save_dir = (
        output_dir
        / model_name
        / f"guidance_{guidance_scale:.1f}"
        / f"class_{class_idx}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, image in enumerate(images):
        array = (image.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
        pil_image = Image.fromarray(array)
        filename = save_dir / f"sample_{idx:04d}.png"
        pil_image.save(filename)


def evaluate_model_with_guidance(
    module: DiffusionModuleWithIP,
    model_name: str,
    guidance_scale: float,
    real_images_by_class: Dict[int, Tensor],
    config: EvaluationConfig,
    device: torch.device,
    output_dir: Path,
    save_images_flag: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single model with a specific guidance scale."""
    results = {
        "model_name": model_name,
        "guidance_scale": guidance_scale,
        "num_samples_per_class": config.num_samples_per_class,
        "sampling_steps": config.sampling_steps,
        "per_class": {},
    }

    all_fake_images = []
    all_real_images = []

    for class_idx in range(config.num_classes):
        print(f"    Class {class_idx}...", end=" ", flush=True)

        # Generate samples
        fake_images = generate_samples_for_class(
            module=module,
            class_label=float(class_idx),
            num_samples=config.num_samples_per_class,
            batch_size=config.batch_size,
            sampling_steps=config.sampling_steps,
            device=device,
            guidance_scale=guidance_scale,
            eta=config.eta,
        )

        real_images = real_images_by_class.get(class_idx)

        if save_images_flag:
            save_images(fake_images, output_dir, class_idx, guidance_scale, model_name)

        # Per-class metrics
        class_metrics = {}
        class_fid = -1.0

        if real_images is not None and HAS_TORCHMETRICS:
            class_fid = compute_fid(real_images, fake_images, device)
            class_metrics["fid"] = class_fid

        is_mean, is_std = compute_inception_score(fake_images, device)
        class_metrics["is_mean"] = is_mean
        class_metrics["is_std"] = is_std

        lpips_div = compute_lpips_diversity(fake_images, device)
        class_metrics["lpips_diversity"] = lpips_div

        results["per_class"][class_idx] = class_metrics

        all_fake_images.append(fake_images)
        if real_images is not None:
            all_real_images.append(real_images)

        if class_fid >= 0:
            print(f"FID={class_fid:.2f}, IS={is_mean:.2f}")
        else:
            print(f"IS={is_mean:.2f}, LPIPS={lpips_div:.4f}")

    # Overall metrics
    all_fake = torch.cat(all_fake_images, dim=0)

    if all_real_images and HAS_TORCHMETRICS:
        all_real = torch.cat(all_real_images, dim=0)
        results["overall_fid"] = compute_fid(all_real, all_fake, device)
        results["overall_ssim"] = compute_ssim_to_real(all_real, all_fake)
    elif all_real_images:
        all_real = torch.cat(all_real_images, dim=0)
        results["overall_ssim"] = compute_ssim_to_real(all_real, all_fake)

    overall_is_mean, overall_is_std = compute_inception_score(all_fake, device)
    results["overall_is_mean"] = overall_is_mean
    results["overall_is_std"] = overall_is_std
    results["overall_lpips_diversity"] = compute_lpips_diversity(
        all_fake, device, num_pairs=200
    )

    return results


def load_all_real_images(
    data_dir: Path,
    num_classes: int,
    image_size: int,
    max_per_class: int = 500,
) -> Dict[int, Tensor]:
    """Load real images for all classes."""
    real_images = {}
    for class_idx in range(num_classes):
        try:
            real_images[class_idx] = load_real_images_for_class(
                data_dir, class_idx, image_size, max_per_class
            )
            print(
                f"  Loaded {len(real_images[class_idx])} real images for class {class_idx}"
            )
        except Exception as e:
            print(f"  Warning: Could not load class {class_idx}: {e}")
            real_images[class_idx] = None
    return real_images


def create_comparison_report(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Create a comparison table from results."""
    rows = []
    for r in results:
        row = {
            "Model": r["model_name"],
            "Guidance": r["guidance_scale"],
            "FID": r.get("overall_fid", -1),
            "IS Mean": r.get("overall_is_mean", -1),
            "IS Std": r.get("overall_is_std", -1),
            "LPIPS Div": r.get("overall_lpips_diversity", -1),
            "SSIM": r.get("overall_ssim", -1),
        }
        # Per-class FID
        for class_idx, class_metrics in r.get("per_class", {}).items():
            row[f"FID_C{class_idx}"] = class_metrics.get("fid", -1)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by model and guidance scale
    df = df.sort_values(["Model", "Guidance"])

    # Save to CSV
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    return df


def print_comparison_summary(df: pd.DataFrame) -> None:
    """Print a formatted comparison summary."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Find best FID for each model
    models = df["Model"].unique()
    for model in models:
        model_df = df[df["Model"] == model]
        best_fid_row = model_df.loc[model_df["FID"].idxmin()]
        print(f"\n{model}:")
        print(
            f"  Best FID: {best_fid_row['FID']:.2f} (guidance={best_fid_row['Guidance']})"
        )
        print(f"  IS: {best_fid_row['IS Mean']:.2f} ± {best_fid_row['IS Std']:.2f}")
        print(f"  LPIPS Diversity: {best_fid_row['LPIPS Div']:.4f}")

    # Overall comparison
    print("\n" + "-" * 80)
    print("OVERALL BEST (Lowest FID):")
    best_overall = df.loc[df["FID"].idxmin()]
    print(f"  Model: {best_overall['Model']}")
    print(f"  Guidance: {best_overall['Guidance']}")
    print(f"  FID: {best_overall['FID']:.2f}")
    print("=" * 80)


def main() -> None:
    args = _parse_args()

    # Quick test mode
    if args.quick_test:
        args.guidance_scales = [0.0, 1.0, 2.0]
        args.num_samples_per_class = 10
        print("Running in quick test mode with reduced samples")

    device = _resolve_device(args.device)
    _set_seed(args.seed)

    print(f"Device: {device}")
    print(f"Guidance scales to test: {args.guidance_scales}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load config
    cfg = _load_config(args.config)
    image_size = cfg.dataset.image_size

    # Load real images
    print("\nLoading real images...")
    real_images_by_class = load_all_real_images(
        args.real_data_dir,
        args.num_classes,
        image_size,
        max_per_class=args.num_samples_per_class * 2,
    )

    # Models to evaluate with their specific frequency scales
    # Model configs:
    # - kwd2qy49: blur=ON, dominant=1.0, non_dominant=1.0 (uniform + blur)
    # - pvq7gpe7: blur=ON, dominant=1.5, non_dominant=0.5 (frequency-aware + blur)
    # - ejfpqk2s: blur=OFF, dominant=1.0, non_dominant=1.0 (uniform + no blur)
    checkpoints = [
        (args.checkpoint1, args.name1, 1.0, 1.0),  # kwd2qy49: uniform
        (args.checkpoint2, args.name2, 1.5, 0.5),  # pvq7gpe7: frequency-aware
    ]

    # Add third checkpoint if it exists
    if args.checkpoint3.exists():
        checkpoints.append(
            (args.checkpoint3, args.name3, 1.0, 1.0)
        )  # ejfpqk2s: uniform, no blur

    all_results = []

    for ckpt_path, model_name, freq_dom, freq_non in checkpoints:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Frequency scales: dominant={freq_dom}, non_dominant={freq_non}")
        print("=" * 60)

        # Load model with correct frequency scales
        module = load_model(ckpt_path, cfg, device, freq_dom, freq_non)

        for guidance_scale in args.guidance_scales:
            print(f"\n  Guidance scale: {guidance_scale}")

            eval_config = EvaluationConfig(
                checkpoint_path=str(ckpt_path),
                checkpoint_name=model_name,
                guidance_scales=[guidance_scale],
                num_samples_per_class=args.num_samples_per_class,
                sampling_steps=args.sampling_steps,
                batch_size=args.batch_size,
                num_classes=args.num_classes,
            )

            results = evaluate_model_with_guidance(
                module=module,
                model_name=model_name,
                guidance_scale=guidance_scale,
                real_images_by_class=real_images_by_class,
                config=eval_config,
                device=device,
                output_dir=output_dir,
                save_images_flag=args.save_images,
            )

            all_results.append(results)

            # Print intermediate results
            print(f"    Overall FID: {results.get('overall_fid', -1):.2f}")
            print(f"    Overall IS: {results.get('overall_is_mean', -1):.2f}")

        # Free memory
        del module
        torch.cuda.empty_cache()

    # Save detailed results
    results_path = output_dir / "detailed_results.json"
    serializable_results = convert_to_serializable(all_results)
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    # Create comparison report
    df = create_comparison_report(all_results, output_dir)
    print("\n" + df.to_string(index=False))

    # Print summary
    print_comparison_summary(df)

    # Save args
    args_dict = vars(args)
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in args_dict.items()}
    with open(output_dir / "args.json", "w") as f:
        json.dump(args_dict, f, indent=2)


if __name__ == "__main__":
    main()
