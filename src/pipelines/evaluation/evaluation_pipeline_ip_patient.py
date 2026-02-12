"""
Patient-Conditioned Evaluation Pipeline for IP-Adapter Models.

This module evaluates IP-Adapter models with ACTUAL image conditioning,
testing the model's ability to:
1. Preserve patient-specific anatomical features
2. Generate accurate disease progression for a given patient
3. Maintain identity consistency across MES levels

Unlike the zero-conditioned evaluation, this pipeline:
- Uses real patient images as structure conditioning
- Measures identity preservation (LPIPS, SSIM between source and generated)
- Tests ordinal consistency (whether generated images respect MES ordering)
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
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor

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

    HAS_TORCHMETRICS = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_TORCHMETRICS = False
    print(f"Warning: torchmetrics not installed: {e}")

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
    print("Warning: scikit-image not installed.")


@dataclass
class PatientEvaluationConfig:
    """Configuration for patient-conditioned evaluation."""

    checkpoint_path: str
    checkpoint_name: str
    guidance_scale: float
    image_scale: float
    num_patients_per_class: int
    sampling_steps: int
    num_classes: int
    mes_steps: int  # Number of MES levels to generate per patient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate IP-Adapter with patient-specific image conditioning."
    )
    parser.add_argument(
        "--checkpoint1",
        type=Path,
        default=Path(
            "/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/kwd2qy49/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
        ),
        help="Path to first checkpoint (blur + uniform).",
    )
    parser.add_argument(
        "--checkpoint2",
        type=Path,
        default=Path(
            "/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/pvq7gpe7/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
        ),
        help="Path to second checkpoint (blur + freq-aware).",
    )
    parser.add_argument(
        "--checkpoint3",
        type=Path,
        default=Path(
            "/home/umut_dundar/repositories/progressive-stable-diffusion/prog-disease-generation-ip/ejfpqk2s/checkpoints/ip-ddpm-epochepoch=0149.ckpt"
        ),
        help="Path to third checkpoint (no blur + uniform).",
    )
    parser.add_argument(
        "--name1",
        type=str,
        default="model_kwd2qy49_blur_uniform",
        help="Name for first model.",
    )
    parser.add_argument(
        "--name2",
        type=str,
        default="model_pvq7gpe7_blur_freqaware",
        help="Name for second model.",
    )
    parser.add_argument(
        "--name3",
        type=str,
        default="model_ejfpqk2s_noblur_uniform",
        help="Name for third model.",
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
        default=Path("outputs/evaluation_ip_patient"),
        help="Directory where evaluation results are saved.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="CFG guidance scale (use best from zero-cond eval).",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="Scale for image conditioning strength.",
    )
    parser.add_argument(
        "--num-patients-per-class",
        type=int,
        default=10,
        help="Number of patient images to sample per MES class.",
    )
    parser.add_argument(
        "--mes-steps",
        type=int,
        default=4,
        help="Number of MES levels to generate (4 = MES 0,1,2,3).",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps.",
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
        "--save-images",
        action="store_true",
        help="Save generated progression images to disk.",
    )
    parser.add_argument(
        "--apply-blur",
        action="store_true",
        help="Apply blur to structure images (for blur-trained models).",
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
    """Load IP-Adapter model with specific frequency scales."""
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"  frequency_dominant_scale: {frequency_dominant_scale}")
    print(f"  frequency_non_dominant_scale: {frequency_non_dominant_scale}")

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


def _apply_gaussian_blur(
    images: Tensor,
    kernel_size: int = 21,
    sigma: float = 5.0,
) -> Tensor:
    """Apply Gaussian blur to images."""
    device = images.device
    dtype = images.dtype

    x = torch.arange(kernel_size, device=device, dtype=dtype)
    x = x - kernel_size // 2
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    gauss_h = gauss.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    gauss_v = gauss.view(1, 1, -1, 1).expand(3, 1, -1, 1)

    pad = kernel_size // 2
    blurred = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(blurred, gauss_h, groups=3)
    blurred = F.conv2d(blurred, gauss_v, groups=3)

    return blurred.clamp(0, 1)


def load_patient_images(
    data_dir: Path,
    class_idx: int,
    num_patients: int,
    image_size: int,
) -> Tuple[List[Tensor], List[Path]]:
    """Load patient images from a specific class."""
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
    paths = []
    extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    for img_path in sorted(class_dir.iterdir()):
        if img_path.suffix.lower() in extensions and len(images) < num_patients:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                images.append(img_tensor)
                paths.append(img_path)
            except Exception as e:
                print(f"  Warning: Could not load {img_path}: {e}")

    return images, paths


def preprocess_for_clip(
    image_tensor: Tensor,
    device: torch.device,
    apply_blur: bool = True,
) -> Tensor:
    """Preprocess image tensor for CLIP encoder."""
    # image_tensor: (3, H, W) in [0, 1]
    if apply_blur:
        blurred = _apply_gaussian_blur(
            image_tensor.unsqueeze(0),
            kernel_size=21,
            sigma=5.0,
        ).squeeze(0)
    else:
        blurred = image_tensor

    # Convert to PIL for CLIP processor
    pil_image = transforms.ToPILImage()(blurred)

    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_inputs = clip_processor(images=pil_image, return_tensors="pt")

    return clip_inputs.pixel_values.to(device)


def _prepare_conditioning(
    module: DiffusionModuleWithIP,
    labels: Tensor,
    structure_image: Tensor,
    is_unconditional: bool = False,
    image_scale: float = 1.0,
) -> Tensor:
    """Prepare combined AOE + Image conditioning embeddings."""
    batch_size = labels.shape[0]
    device = labels.device

    if is_unconditional:
        aoe_embeds = module.ordinal_embedder.get_negative_embedding(
            labels, is_training=False
        )
    else:
        aoe_embeds = module.ordinal_embedder(labels, is_training=False)

    if aoe_embeds.dim() == 2:
        aoe_embeds = aoe_embeds.unsqueeze(1)

    if is_unconditional:
        num_tokens = module.diff_cfg.num_image_tokens
        cross_dim = module.cfg.model.conditioning_dim
        image_embeds = torch.zeros(
            batch_size, num_tokens, cross_dim, device=device, dtype=aoe_embeds.dtype
        )
    else:
        structure_batch = structure_image.expand(batch_size, -1, -1, -1)
        image_embeds = module._get_image_embeds(structure_batch)

        if image_scale != 1.0:
            image_embeds = image_embeds * image_scale

    return torch.cat([aoe_embeds, image_embeds], dim=1)


def ddim_sample_with_image_condition(
    module: DiffusionModuleWithIP,
    labels: Tensor,
    structure_image: Tensor,
    sampling_steps: int,
    device: torch.device,
    eta: float = 0.0,
    guidance_scale: float = 3.0,
    image_scale: float = 1.0,
) -> Tensor:
    """
    DDIM sampling WITH image conditioning for patient-specific generation.

    Args:
        module: DiffusionModuleWithIP
        labels: Mayo scores (B,)
        structure_image: CLIP-preprocessed patient image (1, 3, 224, 224)
        sampling_steps: Number of DDIM steps
        device: Target device
        eta: DDIM stochasticity
        guidance_scale: CFG scale
        image_scale: Image conditioning strength

    Returns:
        Generated latents (B, 4, H//8, W//8)
    """
    num_samples = labels.shape[0]
    height = module.cfg.dataset.image_size
    T = module.diff_cfg.num_train_timesteps

    # Same initial noise for all MES levels (fair comparison)
    single_latent = torch.randn(
        1,
        module.cfg.model.latent_channels,
        height // 8,
        height // 8,
        device=device,
        dtype=torch.float32,
    )
    latents = single_latent.repeat(num_samples, 1, 1, 1)

    alphas_cumprod = module.alphas_cumprod

    timesteps = torch.linspace(
        T - 1,
        0,
        steps=sampling_steps,
        dtype=torch.long,
        device=device,
    )

    # Prepare embeddings WITH image conditioning
    cond_embed = _prepare_conditioning(
        module, labels, structure_image, is_unconditional=False, image_scale=image_scale
    )
    uncond_embed = _prepare_conditioning(
        module, labels, structure_image, is_unconditional=True, image_scale=image_scale
    )

    embed = torch.cat([uncond_embed, cond_embed], dim=0)

    for i, t_scalar in enumerate(timesteps):
        t_int = int(t_scalar.item())
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=device)

        latent_model_input = torch.cat([latents, latents], dim=0)
        t_model_input = torch.cat([t, t], dim=0)

        noise_pred = module(latent_model_input, t_model_input, embed)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        if guidance_scale > 0:
            eps_theta = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            eps_theta = noise_pred_cond

        alpha_bar_t = alphas_cumprod[t_int].to(device=device, dtype=latents.dtype)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x0_pred = (latents - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
        x0_pred = x0_pred.clamp(-4.0, 4.0)

        if i == sampling_steps - 1:
            latents = x0_pred
            continue

        t_prev_int = int(timesteps[i + 1].item())
        alpha_bar_prev = alphas_cumprod[t_prev_int].to(
            device=device, dtype=latents.dtype
        )
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

        if eta == 0.0:
            latents = (
                sqrt_alpha_bar_prev * x0_pred
                + sqrt_one_minus_alpha_bar_prev * eps_theta
            )
        else:
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


def latents_to_images(module: DiffusionModuleWithIP, latents: Tensor) -> Tensor:
    """Decode latents to images."""
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


# Global LPIPS model (cached to avoid reloading)
_LPIPS_MODEL = None


def _get_lpips_model(device: torch.device):
    """Get cached LPIPS model."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None and HAS_LPIPS:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device)
    return _LPIPS_MODEL


def compute_identity_preservation(
    source_image: Tensor,
    generated_images: Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute identity preservation metrics between source and generated images.

    Args:
        source_image: Original patient image (3, H, W) in [0, 1]
        generated_images: Generated progression (N, 3, H, W) in [0, 1]
        device: Compute device

    Returns:
        Dictionary with LPIPS and SSIM scores for each generated image
    """
    results = {}
    num_generated = generated_images.shape[0]

    # Expand source to match generated batch
    source_batch = source_image.unsqueeze(0).expand(num_generated, -1, -1, -1)

    # LPIPS (lower = more similar)
    lpips_fn = _get_lpips_model(device)
    if lpips_fn is not None:
        with torch.no_grad():
            # LPIPS expects [-1, 1] range
            source_lpips = source_batch.to(device) * 2 - 1
            gen_lpips = generated_images.to(device) * 2 - 1
            lpips_scores = lpips_fn(source_lpips, gen_lpips)
            results["lpips_per_mes"] = lpips_scores.squeeze().cpu().numpy().tolist()
            results["lpips_mean"] = float(lpips_scores.mean().item())

    # SSIM (higher = more similar)
    if HAS_SKIMAGE:
        ssim_scores = []
        for i in range(num_generated):
            source_np = source_image.permute(1, 2, 0).cpu().numpy()
            gen_np = generated_images[i].permute(1, 2, 0).cpu().numpy()
            score = ssim(source_np, gen_np, channel_axis=2, data_range=1.0)
            ssim_scores.append(score)
        results["ssim_per_mes"] = ssim_scores
        results["ssim_mean"] = float(np.mean(ssim_scores))

    return results


def generate_patient_progression(
    module: DiffusionModuleWithIP,
    patient_image: Tensor,
    mes_steps: int,
    sampling_steps: int,
    device: torch.device,
    guidance_scale: float = 3.0,
    image_scale: float = 1.0,
    apply_blur: bool = True,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Generate disease progression for a single patient.

    Args:
        module: IP-Adapter model
        patient_image: Source patient image (3, H, W) in [0, 1]
        mes_steps: Number of MES levels to generate
        sampling_steps: DDIM steps
        device: Compute device
        guidance_scale: CFG scale
        image_scale: Image conditioning strength
        apply_blur: Whether to blur the structure image

    Returns:
        - Generated images (N, 3, H, W)
        - Identity preservation metrics
    """
    # Preprocess for CLIP
    structure_image = preprocess_for_clip(patient_image, device, apply_blur=apply_blur)

    # Generate MES labels
    labels = torch.linspace(0.0, 3.0, mes_steps, device=device, dtype=torch.float32)

    # Generate with image conditioning
    with torch.no_grad():
        latents = ddim_sample_with_image_condition(
            module=module,
            labels=labels,
            structure_image=structure_image,
            sampling_steps=sampling_steps,
            device=device,
            guidance_scale=guidance_scale,
            image_scale=image_scale,
        )
        generated = latents_to_images(module, latents)

    # Compute identity preservation
    identity_metrics = compute_identity_preservation(
        patient_image.to(device),
        generated,
        device,
    )

    return generated.cpu(), identity_metrics


def evaluate_model_patient_conditioned(
    module: DiffusionModuleWithIP,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    config: PatientEvaluationConfig,
    device: torch.device,
    save_images: bool = False,
    apply_blur: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model with patient-specific image conditioning.

    For each patient in each class:
    1. Use patient image as structure conditioning
    2. Generate images at all MES levels
    3. Measure identity preservation
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"  Guidance Scale: {config.guidance_scale}")
    print(f"  Image Scale: {config.image_scale}")
    print(f"  Apply Blur: {apply_blur}")
    print(f"{'='*60}")

    image_size = module.cfg.dataset.image_size
    all_metrics = []
    all_generated = []

    # Create output subdirectory for this model
    model_output_dir = output_dir / model_name
    if save_images:
        model_output_dir.mkdir(parents=True, exist_ok=True)

    for class_idx in range(config.num_classes):
        print(f"\nProcessing MES {class_idx} patients...")

        # Load patient images from this class
        patient_images, patient_paths = load_patient_images(
            data_dir,
            class_idx,
            config.num_patients_per_class,
            image_size,
        )

        print(f"  Found {len(patient_images)} patients in MES {class_idx}")

        for patient_idx, (patient_img, patient_path) in enumerate(
            tqdm(
                zip(patient_images, patient_paths),
                total=len(patient_images),
                desc=f"  MES {class_idx}",
            )
        ):
            # Generate progression
            generated, identity_metrics = generate_patient_progression(
                module=module,
                patient_image=patient_img,
                mes_steps=config.mes_steps,
                sampling_steps=config.sampling_steps,
                device=device,
                guidance_scale=config.guidance_scale,
                image_scale=config.image_scale,
                apply_blur=apply_blur,
            )

            # Store metrics
            patient_metrics = {
                "source_class": class_idx,
                "patient_path": str(patient_path.name),
                **identity_metrics,
            }
            all_metrics.append(patient_metrics)
            all_generated.append(generated)

            # Save images if requested
            if save_images:
                patient_dir = model_output_dir / f"mes{class_idx}" / patient_path.stem
                patient_dir.mkdir(parents=True, exist_ok=True)

                # Save source
                source_pil = transforms.ToPILImage()(patient_img)
                source_pil.save(patient_dir / "source.png")

                # Save generated progression
                for mes_idx in range(config.mes_steps):
                    mes_value = mes_idx * 3.0 / (config.mes_steps - 1)
                    gen_pil = transforms.ToPILImage()(generated[mes_idx])
                    gen_pil.save(patient_dir / f"generated_mes{mes_value:.1f}.png")

    # Aggregate metrics
    aggregated = aggregate_patient_metrics(all_metrics, config.num_classes)

    return {
        "model_name": model_name,
        "config": {
            "guidance_scale": config.guidance_scale,
            "image_scale": config.image_scale,
            "sampling_steps": config.sampling_steps,
            "mes_steps": config.mes_steps,
            "num_patients_per_class": config.num_patients_per_class,
            "apply_blur": apply_blur,
        },
        "per_patient_metrics": all_metrics,
        "aggregated": aggregated,
    }


def aggregate_patient_metrics(
    metrics: List[Dict[str, Any]],
    num_classes: int,
) -> Dict[str, Any]:
    """Aggregate patient-level metrics."""
    # Overall averages
    all_lpips = [m["lpips_mean"] for m in metrics if "lpips_mean" in m]
    all_ssim = [m["ssim_mean"] for m in metrics if "ssim_mean" in m]

    # Per source class
    per_class = {}
    for class_idx in range(num_classes):
        class_metrics = [m for m in metrics if m["source_class"] == class_idx]
        if class_metrics:
            per_class[f"mes_{class_idx}"] = {
                "lpips_mean": np.mean(
                    [m["lpips_mean"] for m in class_metrics if "lpips_mean" in m]
                ),
                "ssim_mean": np.mean(
                    [m["ssim_mean"] for m in class_metrics if "ssim_mean" in m]
                ),
                "num_patients": len(class_metrics),
            }

    # LPIPS per target MES (from lpips_per_mes arrays)
    lpips_per_target = {}
    for m in metrics:
        if "lpips_per_mes" in m:
            for i, lpips_val in enumerate(m["lpips_per_mes"]):
                key = f"target_mes_{i}"
                if key not in lpips_per_target:
                    lpips_per_target[key] = []
                lpips_per_target[key].append(lpips_val)

    lpips_per_target_avg = {k: np.mean(v) for k, v in lpips_per_target.items()}

    return {
        "overall_lpips_mean": float(np.mean(all_lpips)) if all_lpips else None,
        "overall_ssim_mean": float(np.mean(all_ssim)) if all_ssim else None,
        "per_source_class": per_class,
        "lpips_per_target_mes": lpips_per_target_avg,
    }


def main() -> None:
    args = _parse_args()

    # Setup
    device = _resolve_device(args.device)
    _set_seed(args.seed)
    config = _load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    # Model configurations with frequency scales
    model_configs = [
        {
            "checkpoint": args.checkpoint1,
            "name": args.name1,
            "frequency_dominant_scale": 1.0,
            "frequency_non_dominant_scale": 1.0,
            "apply_blur": True,  # Model A was trained with blur
        },
        {
            "checkpoint": args.checkpoint2,
            "name": args.name2,
            "frequency_dominant_scale": 1.5,  # IMPORTANT: freq-aware model
            "frequency_non_dominant_scale": 0.5,
            "apply_blur": True,  # Model B was trained with blur
        },
        {
            "checkpoint": args.checkpoint3,
            "name": args.name3,
            "frequency_dominant_scale": 1.0,
            "frequency_non_dominant_scale": 1.0,
            "apply_blur": False,  # Model C was trained WITHOUT blur
        },
    ]

    all_results = []

    for model_cfg in model_configs:
        if not model_cfg["checkpoint"].exists():
            print(f"Checkpoint not found: {model_cfg['checkpoint']}")
            continue

        # Load model with correct frequency scales
        module = load_model(
            model_cfg["checkpoint"],
            config,
            device,
            frequency_dominant_scale=model_cfg["frequency_dominant_scale"],
            frequency_non_dominant_scale=model_cfg["frequency_non_dominant_scale"],
        )

        eval_config = PatientEvaluationConfig(
            checkpoint_path=str(model_cfg["checkpoint"]),
            checkpoint_name=model_cfg["name"],
            guidance_scale=args.guidance_scale,
            image_scale=args.image_scale,
            num_patients_per_class=args.num_patients_per_class,
            sampling_steps=args.sampling_steps,
            num_classes=4,
            mes_steps=args.mes_steps,
        )

        results = evaluate_model_patient_conditioned(
            module=module,
            model_name=model_cfg["name"],
            data_dir=args.real_data_dir,
            output_dir=output_dir,
            config=eval_config,
            device=device,
            save_images=args.save_images,
            apply_blur=model_cfg["apply_blur"],
        )

        all_results.append(results)

        # Free memory
        del module
        torch.cuda.empty_cache()

    # Save results
    results_path = output_dir / "patient_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    # Create summary table
    print("\n" + "=" * 80)
    print("PATIENT-CONDITIONED EVALUATION SUMMARY")
    print("=" * 80)

    summary_data = []
    for result in all_results:
        agg = result["aggregated"]
        summary_data.append(
            {
                "Model": result["model_name"],
                "LPIPS (↓)": f"{agg['overall_lpips_mean']:.4f}"
                if agg["overall_lpips_mean"]
                else "N/A",
                "SSIM (↑)": f"{agg['overall_ssim_mean']:.4f}"
                if agg["overall_ssim_mean"]
                else "N/A",
            }
        )

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # LPIPS per target MES
    print("\nLPIPS by Target MES Level:")
    for result in all_results:
        print(f"\n  {result['model_name']}:")
        for key, val in result["aggregated"]["lpips_per_target_mes"].items():
            print(f"    {key}: {val:.4f}")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
