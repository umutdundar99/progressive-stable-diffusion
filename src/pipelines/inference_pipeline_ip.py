"""
Run DDPM/DDIM inference for MES progression with IP-Adapter conditioning.

This pipeline generates disease progression sequences while maintaining
patient-specific anatomical structure via image conditioning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision import transforms
from transformers import CLIPImageProcessor

from src.models.diffusion_module_ip import DiffusionModuleWithIP


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MES progression with patient-specific anatomical structure."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a Lightning checkpoint from IP-Adapter training.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_ip.yaml"),
        help="Hydra-compatible config file for IP-Adapter training setup.",
    )
    parser.add_argument(
        "--structure-image",
        type=Path,
        required=True,
        help="Reference image for anatomical structure (will be blurred/processed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference_ip"),
        help="Directory where generated progression images are saved.",
    )
    parser.add_argument(
        "--mes-steps",
        type=int,
        default=13,
        help="Number of MES values to sweep from 0 to 3.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
        help="Number of diffusion steps used during sampling (for DDIM).",
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
        default=None,  # None = random seed each run
        help="Random seed for sampling. If not set, uses random seed for variety.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="CFG guidance scale.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="Scale factor for image conditioning strength.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter (0=deterministic, 1=DDPM).",
    )
    parser.add_argument(
        "--no-blur",
        action="store_true",
        default=False,  # Changed: use raw images by default (no blur)
        help="Skip blurring the structure image (use raw image features).",
    )
    parser.add_argument(
        "--zero-image",
        action="store_true",
        default=False,
        help="Use zero image conditioning (c_img=0). Only AOE ordinal conditioning active. "
        "This is useful for testing progression without patient-specific features.",
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_config(config_path: Path) -> DictConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    return OmegaConf.load(config_path)


def _build_labels(
    num_steps: int,
    start: float = 0.0,
    end: float = 3.0,
    device: torch.device | None = None,
) -> Tensor:
    device = device or torch.device("cpu")
    if num_steps <= 0:
        raise ValueError("`mes_steps` must be a positive integer.")
    return torch.linspace(
        start, end, steps=num_steps, device=device, dtype=torch.float32
    )


def _load_and_preprocess_structure_image(
    image_path: Path,
    target_size: int,
    device: torch.device,
    apply_blur: bool = False,  # Changed: use raw images by default
    blur_kernel_size: int = 7,  # Reduced from 15
    blur_sigma: float = 2.0,  # Reduced from 5.0
) -> Tuple[Tensor, Tensor]:
    """
    Load and preprocess the structure image.

    Returns:
        - structure_tensor: For image encoder (B, 3, 224, 224) - CLIP preprocessed
        - display_tensor: For saving/visualization (3, H, W) in [0, 1]
    """
    # Load image
    pil_image = Image.open(image_path).convert("RGB")

    transform_display = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),  # [0, 1]
        ]
    )
    display_tensor = transform_display(pil_image)  # (3, H, W) in [0, 1]

    if apply_blur:
        display_tensor = _apply_gaussian_blur(
            display_tensor.unsqueeze(0),  # (1, 3, H, W)
            kernel_size=blur_kernel_size,
            sigma=blur_sigma,
        ).squeeze(0)  # (3, H, W)

    blurred_pil = transforms.ToPILImage()(display_tensor)

    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_inputs = clip_processor(images=blurred_pil, return_tensors="pt")
    structure_tensor = clip_inputs.pixel_values.to(device)

    return structure_tensor, display_tensor


def _apply_gaussian_blur(
    images: Tensor,
    kernel_size: int = 15,
    sigma: float = 5.0,
) -> Tensor:
    """
    Apply Gaussian blur to extract structural information.

    Args:
        images: (B, C, H, W) in [0, 1]

    Returns:
        Blurred images (B, C, H, W) in [0, 1]
    """
    device = images.device
    dtype = images.dtype

    # Create Gaussian kernel
    x = torch.arange(kernel_size, device=device, dtype=dtype)
    x = x - kernel_size // 2
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # Separable convolution
    gauss_h = gauss.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    gauss_v = gauss.view(1, 1, -1, 1).expand(3, 1, -1, 1)

    # Pad and convolve
    pad = kernel_size // 2
    blurred = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(blurred, gauss_h, groups=3)
    blurred = F.conv2d(blurred, gauss_v, groups=3)

    return blurred.clamp(0, 1)


def _prepare_conditioning(
    module: DiffusionModuleWithIP,
    labels: Tensor,
    structure_image: Tensor,
    is_unconditional: bool = False,
    image_scale: float = 1.0,
) -> Tensor:
    """
    Prepare combined AOE + Image conditioning embeddings.

    Args:
        module: The diffusion module
        labels: Mayo scores (B,)
        structure_image: CLIP-preprocessed structure image (1, 3, 224, 224)
        is_unconditional: If True, use zero image embeddings for CFG
        image_scale: Scale factor for image conditioning strength

    Returns:
        Combined embeddings (B, 1 + num_image_tokens, D)
    """
    batch_size = labels.shape[0]
    device = labels.device

    # Get AOE embeddings
    if is_unconditional:
        # For unconditional, use negative embedding (Mayo 0 or learned null)
        aoe_embeds = module.ordinal_embedder.get_negative_embedding(
            labels, is_training=False
        )
    else:
        aoe_embeds = module.ordinal_embedder(labels, is_training=False)

    if aoe_embeds.dim() == 2:
        aoe_embeds = aoe_embeds.unsqueeze(1)  # (B, 1, D)

    # Get Image embeddings
    if is_unconditional:
        # Zero image embeddings for unconditional (CFG)
        num_tokens = module.diff_cfg.num_image_tokens
        cross_dim = module.cfg.model.conditioning_dim
        image_embeds = torch.zeros(
            batch_size, num_tokens, cross_dim, device=device, dtype=aoe_embeds.dtype
        )
    else:
        # Expand structure image to batch size
        structure_batch = structure_image.expand(batch_size, -1, -1, -1)
        image_embeds = module._get_image_embeds(structure_batch)  # (B, num_tokens, D)

        # Apply image scale to control conditioning strength
        if image_scale != 1.0:
            image_embeds = image_embeds * image_scale

    # Concatenate: [AOE, Image]
    combined = torch.cat([aoe_embeds, image_embeds], dim=1)

    return combined


def _contribution_to_heatmap(
    contribution: Tensor,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Convert contribution map (L2 norms) to a colorized heatmap.

    Args:
        contribution: Tensor of shape (batch, seq_len) or (seq_len,)
        target_size: (H, W) to resize heatmap to

    Returns:
        Colorized heatmap as numpy array (H, W, 3) in range [0, 255]
    """
    import matplotlib.pyplot as plt

    if contribution.dim() == 2:
        contribution = contribution[0]

    seq_len = contribution.shape[0]
    spatial_size = int(np.sqrt(seq_len))

    heatmap = contribution.view(spatial_size, spatial_size).numpy()

    print(
        f"    Heatmap stats: min={heatmap.min():.4f}, max={heatmap.max():.4f}, std={heatmap.std():.4f}"
    )

    p_low, p_high = np.percentile(heatmap, [2, 98])
    if p_high - p_low < 1e-6:
        print("Low variance in heatmap, using uniform visualization")
        heatmap = np.ones_like(heatmap) * 0.5
    else:
        heatmap = np.clip(heatmap, p_low, p_high)
        heatmap = (heatmap - p_low) / (p_high - p_low)

    cmap = plt.cm.jet
    colored = cmap(heatmap)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)

    colored_pil = Image.fromarray(colored)
    colored_pil = colored_pil.resize(target_size, Image.Resampling.BILINEAR)

    return np.array(colored_pil)


def _overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay heatmap on image with transparency.

    Args:
        image: RGB image (H, W, 3) in range [0, 255]
        heatmap: Colorized heatmap (H, W, 3) in range [0, 255]
        alpha: Blend factor (0=image only, 1=heatmap only)

    Returns:
        Blended image (H, W, 3) in range [0, 255]
    """
    blended = (1 - alpha) * image.astype(np.float32) + alpha * heatmap.astype(
        np.float32
    )
    return blended.clip(0, 255).astype(np.uint8)


def _visualize_contribution_maps(
    attn_maps: Dict[str, List[Tensor]],
    images: Tensor,
    labels: Tensor,
    output_dir: Path,
) -> None:
    """
    Create and save contribution map visualizations.

    Args:
        attn_maps: Dict with 'aoe' and 'ip' keys, each containing list of contribution maps
        images: Generated images (B, 3, H, W) in range [0, 1]
        labels: MES labels (B,)
        output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt

    viz_dir = output_dir / "attention_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Average contribution maps across timesteps
    aoe_maps = [m for m in attn_maps["aoe"] if m is not None]
    ip_maps = [m for m in attn_maps["ip"] if m is not None]

    if not aoe_maps and not ip_maps:
        print("⚠️  No attention maps collected for visualization")
        return

    height, width = images.shape[2], images.shape[3]
    target_size = (width, height)

    # Process each sample in batch
    batch_size = images.shape[0]

    for b in range(batch_size):
        label = labels[b].item()
        print(f"📊 Processing attention maps for MES {label:.2f}")

        # Convert image to numpy
        img_np = (images[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Average AOE contributions across all collected timesteps for this sample
        if aoe_maps:
            print(f"  AOE maps collected: {len(aoe_maps)}")
            aoe_avg = torch.stack([m[b] if m.dim() == 2 else m for m in aoe_maps]).mean(
                dim=0
            )
            print("  AOE (query-value alignment):")
            aoe_heatmap = _contribution_to_heatmap(aoe_avg, target_size)
            aoe_overlay = _overlay_heatmap_on_image(img_np, aoe_heatmap, alpha=0.4)

            # Save AOE heatmap
            Image.fromarray(aoe_heatmap).save(
                viz_dir / f"aoe_heatmap_mes_{label:.2f}.png"
            )
            Image.fromarray(aoe_overlay).save(
                viz_dir / f"aoe_overlay_mes_{label:.2f}.png"
            )

        # Average IP contributions across all collected timesteps for this sample
        if ip_maps:
            print(f"  IP maps collected: {len(ip_maps)}")
            ip_avg = torch.stack([m[b] if m.dim() == 2 else m for m in ip_maps]).mean(
                dim=0
            )
            print("  IP (attention sum):")
            ip_heatmap = _contribution_to_heatmap(ip_avg, target_size)
            ip_overlay = _overlay_heatmap_on_image(img_np, ip_heatmap, alpha=0.4)

            # Save IP heatmap
            Image.fromarray(ip_heatmap).save(
                viz_dir / f"ip_heatmap_mes_{label:.2f}.png"
            )
            Image.fromarray(ip_overlay).save(
                viz_dir / f"ip_overlay_mes_{label:.2f}.png"
            )

        # Create combined figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Attention Contribution Maps - MES {label:.2f}", fontsize=14)

        # Row 1: AOE
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Generated Image")
        axes[0, 0].axis("off")

        if aoe_maps:
            axes[0, 1].imshow(aoe_heatmap)
            axes[0, 1].set_title("AOE Contribution")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(aoe_overlay)
            axes[0, 2].set_title("AOE Overlay")
            axes[0, 2].axis("off")

        # Row 2: IP
        axes[1, 0].imshow(img_np)
        axes[1, 0].set_title("Generated Image")
        axes[1, 0].axis("off")

        if ip_maps:
            axes[1, 1].imshow(ip_heatmap)
            axes[1, 1].set_title("IP Contribution")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(ip_overlay)
            axes[1, 2].set_title("IP Overlay")
            axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(
            viz_dir / f"combined_mes_{label:.2f}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"✅ Saved attention visualizations to {viz_dir}")


def _ddim_sample_ip(
    module: DiffusionModuleWithIP,
    labels: Tensor,
    structure_image: Tensor,
    sampling_steps: int,
    device: torch.device,
    eta: float = 0.0,
    guidance_scale: float = 2.0,
    image_scale: float = 1.0,
) -> Tensor:
    """
    DDIM sampling with IP-Adapter conditioning.

    Args:
        module: DiffusionModuleWithIP
        labels: Mayo scores (B,)
        structure_image: CLIP-preprocessed structure image (1, 3, 224, 224)
        sampling_steps: Number of DDIM steps
        device: Target device
        eta: DDIM stochasticity (0=deterministic)
        guidance_scale: CFG scale
        image_scale: Scale for image conditioning

    Returns:
        Generated latents (B, 4, H//8, W//8)
    """
    num_samples = labels.shape[0]
    height = module.cfg.dataset.image_size
    T = module.diff_cfg.num_train_timesteps

    if sampling_steps > T:
        raise ValueError(
            f"sampling_steps={sampling_steps} must be <= num_train_timesteps={T}"
        )

    # Initialize latents - same noise for all MES levels for fair comparison
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

    # DDIM timestep schedule
    timesteps = torch.linspace(
        T - 1,
        0,
        steps=sampling_steps,
        dtype=torch.long,
        device=device,
    )

    # Prepare conditional and unconditional embeddings
    cond_embed = _prepare_conditioning(
        module, labels, structure_image, is_unconditional=False, image_scale=image_scale
    )
    uncond_embed = _prepare_conditioning(
        module, labels, structure_image, is_unconditional=True, image_scale=image_scale
    )

    # CFG: concat unconditional + conditional
    embed = torch.cat([uncond_embed, cond_embed], dim=0)  # (2*B, seq, D)

    # TODO: for visualization only
    timesteps_to_visualize = timesteps[:10]
    attn_maps = {"ip": [], "aoe": []}

    # DDIM loop
    for i, t_scalar in enumerate(timesteps):
        t_int = int(t_scalar.item())
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=device)

        # Duplicate latents for CFG
        latent_model_input = torch.cat([latents, latents], dim=0)  # (2*B, 4, H, W)
        t_model_input = torch.cat([t, t], dim=0)  # (2*B,)

        # Predict noise
        noise_pred = module(latent_model_input, t_model_input, embed)
        if t_scalar in timesteps_to_visualize:
            # Only collect from mid_block (consistent spatial resolution)
            for name, attn_processor in module.unet.unet.attn_processors.items():
                if "mid_block" in name and hasattr(
                    attn_processor, "get_last_attention_maps"
                ):
                    maps = attn_processor.get_last_attention_maps()
                    if maps["ip"] is not None:
                        cond_ip = maps["ip"][num_samples:]
                        attn_maps["ip"].append(cond_ip.cpu())
                    if maps["aoe"] is not None:
                        cond_aoe = maps["aoe"][num_samples:]
                        attn_maps["aoe"].append(cond_aoe.cpu())

        # CFG
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        eps_theta = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # DDIM update
        alpha_bar_t = alphas_cumprod[t_int].to(device=device, dtype=latents.dtype)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # Predict x0
        x0_pred = (latents - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
        x0_pred = x0_pred.clamp(-4.0, 4.0)  # Stability clipping

        # Final step: return x0 directly
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
                / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(latents)
            latents = (
                sqrt_alpha_bar_prev * x0_pred
                + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_theta
                + sigma * noise
            )

    return latents, attn_maps


def _latents_to_images(module: DiffusionModuleWithIP, latents: Tensor) -> Tensor:
    """Decode latents with the frozen SD VAE and map to [0, 1] RGB."""
    with torch.no_grad():
        # Undo SD latent scaling
        scaled = latents / module.diff_cfg.latent_scale
        decoded = module.vae.decode(scaled)

    if hasattr(decoded, "sample"):
        images = decoded.sample
    else:
        images = decoded

    images = images.clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0
    return images.clamp(0.0, 1.0)


def _save_sequence(
    images: Tensor,
    labels: Tensor,
    output_dir: Path,
    structure_image: Optional[Tensor] = None,
) -> None:
    """Save generated progression images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = images.cpu()

    # Save structure reference if provided
    if structure_image is not None:
        struct_array = (
            structure_image.permute(1, 2, 0).mul(255).to(torch.uint8)
        ).numpy()
        struct_pil = Image.fromarray(struct_array)
        struct_pil.save(output_dir / "structure_reference.png")

    # Save progression images
    for idx, (image, label) in enumerate(zip(images, labels)):
        array = (image.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
        pil_image = Image.fromarray(array)
        filename = output_dir / f"mes_{label.item():.2f}_{idx:02d}.png"
        pil_image.save(filename)


def _create_progression_grid(
    images: Tensor,
    labels: Tensor,
    structure_image: Optional[Tensor] = None,
    output_path: Path = None,
) -> Image.Image:
    """Create a grid visualization of the progression."""
    images = images.cpu()
    num_images = len(images)

    # Calculate grid dimensions
    ncols = min(num_images, 7)
    nrows = (num_images + ncols - 1) // ncols

    # Add row for structure image if provided
    if structure_image is not None:
        nrows += 1

    img_h, img_w = images.shape[2], images.shape[3]
    padding = 4

    # Create grid
    grid_h = nrows * (img_h + padding) + padding
    grid_w = ncols * (img_w + padding) + padding
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    row_offset = 0

    # Add structure image in first row (centered)
    if structure_image is not None:
        struct_array = (
            structure_image.permute(1, 2, 0).mul(255).to(torch.uint8)
        ).numpy()
        struct_pil = Image.fromarray(struct_array)
        # Resize to match generated image size
        struct_pil = struct_pil.resize((img_w, img_h))
        x = (grid_w - img_w) // 2
        y = padding
        grid.paste(struct_pil, (x, y))
        row_offset = 1

    # Add generated images
    for idx, (image, label) in enumerate(zip(images, labels)):
        array = (image.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
        pil_img = Image.fromarray(array)

        row = idx // ncols + row_offset
        col = idx % ncols
        x = padding + col * (img_w + padding)
        y = padding + row * (img_h + padding)
        grid.paste(pil_img, (x, y))

    if output_path:
        grid.save(output_path)

    return grid


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)

    # Handle seed: if None, generate random seed for variety
    if args.seed is None:
        import time

        seed = int(time.time() * 1000) % (2**32)
        print(f"🎲 Using random seed: {seed}")
    else:
        seed = args.seed
        print(f"🎲 Using fixed seed: {seed}")
    _set_seed(seed)

    print(f"🔧 Device: {device}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"🖼️  Structure image: {args.structure_image}")
    print(f"Applying blur to structure image: {not args.no_blur}")

    # Load config
    cfg = _load_config(args.config)

    # Load model
    print("📦 Loading model...")
    module = DiffusionModuleWithIP.load_from_checkpoint(
        str(args.checkpoint),
        cfg=cfg,
        map_location="cpu",
        weights_only=False,
    )
    module = module.to(device)
    module = module.to(torch.float32)
    module.eval()

    # Load and preprocess structure image
    print("🖼️  Processing structure image...")
    blur_config = cfg.model
    structure_tensor, display_tensor = _load_and_preprocess_structure_image(
        args.structure_image,
        target_size=cfg.dataset.image_size,
        device=device,
        apply_blur=not args.no_blur,
        blur_kernel_size=getattr(blur_config, "blur_kernel_size", 15),
        blur_sigma=getattr(blur_config, "blur_sigma", 5.0),
    )

    # Build MES labels (num_classes=4 means MES scores 0-3)
    labels = _build_labels(
        args.mes_steps,
        start=0.0,
        end=float(cfg.dataset.num_classes - 1),
        device=device,
    )

    print(
        f"🎯 Generating {len(labels)} images with MES from {labels[0]:.2f} to {labels[-1]:.2f}"
    )
    print(f"   Guidance scale: {args.guidance_scale}")
    print(f"   Image scale: {args.image_scale}")
    print(f"   Sampling steps: {args.sampling_steps}")

    # Zero image conditioning mode (like Meryem's thesis - only AOE active)
    if args.zero_image:
        print("🔇 Zero image conditioning mode (c_img = 0, only AOE active)")
        effective_image_scale = 0.0
    else:
        effective_image_scale = args.image_scale

    # Generate
    with torch.no_grad():
        latents, attn_maps = _ddim_sample_ip(
            module,
            labels,
            structure_tensor,
            args.sampling_steps,
            device,
            eta=args.eta,
            guidance_scale=args.guidance_scale,
            image_scale=effective_image_scale,
        )

        images = _latents_to_images(module, latents)

        # Save individual images
        _save_sequence(images, labels, args.output_dir, display_tensor)

        # Create and save progression grid
        grid_path = args.output_dir / "progression_grid.png"
        _create_progression_grid(images, labels, display_tensor, grid_path)

        # Visualize attention contribution maps
        _visualize_contribution_maps(attn_maps, images, labels, args.output_dir)

    print(f"✅ Saved {len(labels)} progression images to {args.output_dir}")
    print(f"✅ Saved progression grid to {grid_path}")


if __name__ == "__main__":
    main()
