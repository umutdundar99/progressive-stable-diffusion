"""
Run DDPM/DDIM inference for MES progression with IP-Adapter conditioning.

This pipeline generates disease progression sequences while maintaining
patient-specific anatomical structure via image conditioning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision import transforms
from transformers import CLIPImageProcessor

from src.models.diffusion_module_ip import DiffusionModuleWithIP


def _load_leace_projection(leace_path: Path, device: torch.device) -> dict:
    """Load a pre-computed LEACE projection matrix."""
    data = torch.load(leace_path, map_location="cpu", weights_only=True)
    data["P_null"] = data["P_null"].to(device)
    data["mu"] = data["mu"].to(device)
    print(
        f"Loaded LEACE projection (rank={data['rank']}, "
        f"tokens={data['num_tokens']}, dim={data['token_dim']})"
    )
    return data


def _apply_leace(image_embeds: Tensor, leace: dict) -> Tensor:
    """
    Apply LEACE projection to erase disease information from image embeddings.

    Args:
        image_embeds: (B, num_tokens, D)
        leace: dict with P_null (T*D, T*D), mu (T*D,)

    Returns:
        Cleaned image embeddings (B, num_tokens, D)
    """
    B, T, D = image_embeds.shape
    P_null = leace["P_null"].to(device=image_embeds.device, dtype=image_embeds.dtype)
    mu = leace["mu"].to(device=image_embeds.device, dtype=image_embeds.dtype)

    flat = image_embeds.reshape(B, T * D)

    flat_centered = flat - mu.unsqueeze(0)
    flat_projected = flat_centered @ P_null.T
    flat_clean = flat_projected + mu.unsqueeze(0)

    return flat_clean.reshape(B, T, D)


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
        default=None,
        help="Random seed for sampling. If not set, uses random seed for variety.",
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
        "--zero-image",
        action="store_true",
        default=False,
        help="Use zero image conditioning (c_img=0). Only AOE ordinal conditioning active. "
        "This is useful for testing progression without patient-specific features.",
    )
    parser.add_argument(
        "--leace",
        type=Path,
        default=None,
        help="Path to a LEACE projection .pt file (from scripts/compute_leace_projection.py). "
        "When provided, disease information is erased from image embeddings before conditioning.",
    )
    parser.add_argument(
        "--source-label",
        type=float,
        default=None,
        help="Mayo score of the input structure image (e.g. 2.0 for a Mayo 2 image). "
        "Used by FeaturePurifier to remove disease info from IP-adapter features. "
        "Required for cross-severity generation. If omitted, defaults to 0.",
    )
    parser.add_argument(
        "--steer-scale",
        type=float,
        default=0.0,
        help="Attention-level delta steering scale.  When > 0, the delta pathway "
        "inside SplitInjectionAttentionProcessor is activated, adding/subtracting "
        "the directional disease change (E[target] - E[source]).  "
        "Recommended range: 0.3–1.0.  0 disables steering.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG guidance scale (baseline mode only). "
        "Default reads from config or 1.0.",
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
            transforms.ToTensor(),
        ]
    )
    display_tensor = transform_display(pil_image)

    blurred_pil = transforms.ToPILImage()(display_tensor)

    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_inputs = clip_processor(
        images=blurred_pil, return_tensors="pt", do_rescale=True
    )
    structure_tensor = clip_inputs.pixel_values.to(device)

    return structure_tensor, display_tensor


def _prepare_conditioning(
    module: DiffusionModuleWithIP,
    target_labels: Tensor,
    source_labels: Tensor,
    structure_image: Tensor,
    image_scale: float = 1.0,
    leace: Optional[dict] = None,
    zero_aoe: bool = False,
) -> Tensor:
    """
    Prepare conditioning embeddings for inference.

    When ``use_routing_gates=True``:
        3-segment: [Target_AOE(N) | E_clean(N) | Delta_AOE(N)]

    When ``use_routing_gates=False``:
        2-segment: [AOE(N) | Image(N)]

    Args:
        module: The diffusion module
        target_labels: Desired Mayo scores for the output (B,)
        source_labels: Mayo score of the input structure image (B,)
        structure_image: CLIP-preprocessed structure image (1, 3, 224, 224)
        image_scale: Scale factor for image conditioning strength
        leace: Optional LEACE projection dict (legacy, applied before purifier)
        zero_aoe: If True, replace AOE with ``get_negative_embedding``
                  (unconditional pass for CFG).

    Returns:
        Combined embeddings (B, total_tokens, D)
    """
    batch_size = target_labels.shape[0]
    use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)

    target_aoe = module.ordinal_embedder(target_labels, is_training=False)
    if target_aoe.dim() == 2:
        target_aoe = target_aoe.unsqueeze(1)

    # CFG unconditional pass: replace AOE with negative conditioning
    if zero_aoe:
        target_aoe = module.ordinal_embedder.get_negative_embedding(
            target_labels, is_training=False
        )
        if target_aoe.dim() == 2:
            target_aoe = target_aoe.unsqueeze(1)

    source_aoe = module.ordinal_embedder(source_labels, is_training=False)
    if source_aoe.dim() == 2:
        source_aoe = source_aoe.unsqueeze(1)

    structure_batch = structure_image.expand(batch_size, -1, -1, -1)
    image_embeds = module._get_image_embeds(structure_batch)

    if leace is not None:
        image_embeds = _apply_leace(image_embeds, leace)

    if module.feature_purifier is not None:
        image_embeds = module.feature_purifier(image_embeds, source_aoe)

    if image_scale != 1.0:
        image_embeds = image_embeds * image_scale

    if use_routing_gates:
        # 3-segment: [Target_AOE | E_clean | Delta_AOE]
        delta_embeds = module.ordinal_embedder.get_ordinal_delta_embedding(
            source_labels, target_labels
        )
        if delta_embeds.dim() == 2:
            delta_embeds = delta_embeds.unsqueeze(1)
        combined = torch.cat([target_aoe, image_embeds, delta_embeds], dim=1)
    else:
        # 2-segment: [AOE | Image]
        combined = torch.cat([target_aoe, image_embeds], dim=1)

    return combined


def _set_delta_scale_on_processors(
    module: DiffusionModuleWithIP,
    delta_scale: float,
) -> None:
    """Set ``delta_scale`` on every :class:`SplitInjectionAttentionProcessor` in the UNet."""
    for _name, mod in module.unet.unet.named_modules():
        if hasattr(mod, "processor") and hasattr(mod.processor, "delta_scale"):
            mod.processor.delta_scale = delta_scale


def _ddim_sample_ip(
    module: DiffusionModuleWithIP,
    target_labels: Tensor,
    source_labels: Tensor,
    structure_image: Tensor,
    sampling_steps: int,
    device: torch.device,
    eta: float = 0.0,
    image_scale: float = 1.0,
    leace: Optional[dict] = None,
    steer_scale: float = 0.0,
    guidance_scale: float = 1.0,
) -> Tensor:
    """
    DDIM sampling with mode-aware disease control.

    **Routing-gates mode** (``use_routing_gates=True``):
        Single conditional pass.  ``steer_scale`` controls delta pathway.
        ``guidance_scale`` is ignored.

    **Baseline mode** (``use_routing_gates=False``):
        CFG dual-pass when ``guidance_scale != 1.0``::

            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        Unconditional AOE uses ``get_negative_embedding``.

    Args:
        module: DiffusionModuleWithIP
        target_labels: Desired Mayo scores for the output (B,)
        source_labels: Mayo score of the input structure image (B,)
        structure_image: CLIP-preprocessed structure image (1, 3, 224, 224)
        sampling_steps: Number of DDIM steps
        device: Target device
        eta: DDIM stochasticity (0=deterministic)
        image_scale: Scale for image conditioning
        leace: Optional LEACE projection dict
        steer_scale: Delta pathway scale (routing gates only; 0 disables)
        guidance_scale: CFG scale (baseline only; 1.0 = no guidance)

    Returns:
        Generated latents (B, 4, H//8, W//8)
    """
    use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)
    do_cfg = (not use_routing_gates) and (guidance_scale != 1.0)

    num_samples = target_labels.shape[0]
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

    # Prepare conditional embeddings
    embed_cond = _prepare_conditioning(
        module,
        target_labels,
        source_labels,
        structure_image,
        image_scale=image_scale,
        leace=leace,
    )

    # Unconditional embeddings for CFG (baseline mode only)
    embed_uncond = None
    if do_cfg:
        embed_uncond = _prepare_conditioning(
            module,
            target_labels,
            source_labels,
            structure_image,
            image_scale=image_scale,
            leace=leace,
            zero_aoe=True,
        )

    _set_delta_scale_on_processors(module, steer_scale)

    for i, t_scalar in enumerate(timesteps):
        t_int = int(t_scalar.item())
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=device)

        if do_cfg:
            eps_cond = module(latents, t, embed_cond)
            eps_uncond = module(latents, t, embed_uncond)
            eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_theta = module(latents, t, embed_cond)

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
                / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(latents)
            latents = (
                sqrt_alpha_bar_prev * x0_pred
                + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_theta
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


def _save_sequence(
    images: Tensor,
    labels: Tensor,
    output_dir: Path,
    structure_image: Optional[Tensor] = None,
) -> None:
    """Save generated progression images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = images.cpu()

    if structure_image is not None:
        struct_array = (
            structure_image.permute(1, 2, 0).mul(255).to(torch.uint8)
        ).numpy()
        struct_pil = Image.fromarray(struct_array)
        struct_pil.save(output_dir / "structure_reference.png")

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

    ncols = min(num_images, 7)
    nrows = (num_images + ncols - 1) // ncols

    if structure_image is not None:
        nrows += 1

    img_h, img_w = images.shape[2], images.shape[3]
    padding = 4

    grid_h = nrows * (img_h + padding) + padding
    grid_w = ncols * (img_w + padding) + padding
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    row_offset = 0

    if structure_image is not None:
        struct_array = (
            structure_image.permute(1, 2, 0).mul(255).to(torch.uint8)
        ).numpy()
        struct_pil = Image.fromarray(struct_array)

        struct_pil = struct_pil.resize((img_w, img_h))
        x = (grid_w - img_w) // 2
        y = padding
        grid.paste(struct_pil, (x, y))
        row_offset = 1

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

    cfg = _load_config(args.config)

    print("📦 Loading model...")
    module = DiffusionModuleWithIP.load_from_checkpoint(
        str(args.checkpoint),
        cfg=cfg,
        weights_only=False,
    )
    module = module.to(device)
    module = module.to(torch.float32)
    module.eval()

    print("🖼️  Processing structure image...")
    structure_tensor, display_tensor = _load_and_preprocess_structure_image(
        args.structure_image,
        target_size=cfg.dataset.image_size,
        device=device,
    )

    target_labels = _build_labels(
        args.mes_steps,
        start=0.0,
        end=float(cfg.dataset.num_classes - 1),
        device=device,
    )

    source_value = args.source_label if args.source_label is not None else 0.0
    source_labels = torch.full_like(target_labels, source_value)

    print(
        f"🎯 Generating {len(target_labels)} images with MES from "
        f"{target_labels[0]:.2f} to {target_labels[-1]:.2f}"
    )
    print(f"   Source label (input image): {source_value:.2f}")
    print(f"   Image scale: {args.image_scale}")
    print(f"   Steer scale: {args.steer_scale}")
    print(f"   Sampling steps: {args.sampling_steps}")

    # -- Resolve guidance scale ------------------------------------------------
    use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)
    if args.guidance_scale is not None:
        guidance_scale = args.guidance_scale
    else:
        guidance_scale = getattr(module.diff_cfg, "guidance_scale", 1.0)
    if use_routing_gates:
        guidance_scale = 1.0
    print(
        f"   use_routing_gates={use_routing_gates}  " f"guidance_scale={guidance_scale}"
    )

    if args.zero_image:
        print("🔇 Zero image conditioning mode (c_img = 0, only AOE active)")
        effective_image_scale = 0.0
    else:
        effective_image_scale = args.image_scale

    leace = None
    if args.leace is not None:
        leace = _load_leace_projection(args.leace, device)
        print(f"🧹 LEACE disease erasure active (rank={leace['rank']})")

    with torch.no_grad():
        latents = _ddim_sample_ip(
            module,
            target_labels,
            source_labels,
            structure_tensor,
            args.sampling_steps,
            device,
            eta=args.eta,
            image_scale=effective_image_scale,
            leace=leace,
            steer_scale=args.steer_scale,
            guidance_scale=guidance_scale,
        )

        images = _latents_to_images(module, latents)

        _save_sequence(images, target_labels, args.output_dir, display_tensor)

        grid_path = args.output_dir / "progression_grid.png"
        _create_progression_grid(images, target_labels, display_tensor, grid_path)

    print(f"✅ Saved {len(target_labels)} progression images to {args.output_dir}")
    print(f"✅ Saved progression grid to {grid_path}")


if __name__ == "__main__":
    main()
