"""Run DDPM / DDIM inference to visualize a progression across MES severity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision import transforms

from src.models.diffusion_module import DiffusionModule


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MES progression samples from a trained diffusion model."
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
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Directory where generated MES progression images are saved.",
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
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2,
        help="CFG guidance scale (default: use value from config).",
    )
    parser.add_argument(
        "--structure-image",
        type=Path,
        default=None,
        help="Path to structure image for IP-Adapter or DDIM inversion (if used).",
    )
    parser.add_argument(
        "--use-ddim-invert",
        action="store_true",
        help="Use DDIM inversion to convert structure image to latent space conditioning.",
    )
    parser.add_argument(
        "--ddim-invert-steps",
        type=int,
        default=50,
        help="Number of DDIM steps used for inverting structure image to latent space.",
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


def _images_to_latents(module: DiffusionModule, images: Tensor) -> Tensor:
    """Encode images to latent space using VAE."""
    with torch.no_grad():
        vae_output = module.vae.encode(images)
        latents = vae_output.latent_dist.sample() * module.diff_cfg.latent_scale
    return latents


def _load_structure_image(
    image_path: Path,
    target_size: int,
    device: torch.device,
) -> Tensor:
    """
    Load and preprocess structure image for DDIM inversion.

    Returns:
        Structure image tensor (1, 3, H, W) in [-1, 1] range
    """
    pil_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ]
    )
    image_tensor = transform(pil_image).unsqueeze(0)  # (1, 3, H, W)
    return image_tensor.to(device)


def _ddim_invert(
    module: DiffusionModule,
    structure_image: Tensor,
    invert_steps: int,
    device: torch.device,
    guidance_scale: float = 0.0,
) -> Tensor:
    """
    DDIM inversion: convert image to latent space by running diffusion backward.

    Adapted from Hugging Face DDIM Inversion. Inverts a clean image to noisy latent
    space by running the reverse diffusion process, capturing structural information
    that can be used as conditioning in the generation process.

    Args:
        module: DiffusionModule
        structure_image: Image tensor (1, 3, H, W) in [-1, 1]
        invert_steps: Number of DDIM inversion steps
        device: Target device
        guidance_scale: Guidance scale for inversion (optional, usually 0.0)

    Returns:
        Inverted latent tensor (1, 4, H//8, W//8)
    """
    # Encode image to latent space (clean latent)
    latents = _images_to_latents(module, structure_image)  # (1, 4, H//8, W//8)

    intermediate_latents = []

    T = module.diff_cfg.num_train_timesteps
    alphas_cumprod = module.alphas_cumprod

    if invert_steps > T:
        raise ValueError(
            f"invert_steps={invert_steps} must be <= num_train_timesteps={T}"
        )

    # Create timestep schedule: T-1 → 0 (reversed, going from noisy to clean during forward pass)
    timesteps = torch.linspace(
        T - 1,
        0,
        steps=invert_steps,
        dtype=torch.long,
        device=device,
    )

    # Reverse the timesteps for inversion (we go T-1 → 0)
    timesteps = reversed(timesteps)

    # Neutral label for unconditional embeddings during inversion
    neutral_label = torch.tensor([0.0], device=device)
    uncond_embed = module.ordinal_embedder(
        neutral_label, is_training=False, unconditional=True
    )

    # DDIM inversion loop: go from clean image (x_0) to noisy latent (x_T)
    for i, t_scalar in enumerate(timesteps):
        t_int = int(t_scalar.item())
        t = torch.full((1,), t_int, dtype=torch.long, device=device)

        # Predict noise at current timestep
        with torch.no_grad():
            noise_pred = module(latents, t, uncond_embed)

        # Get alpha values for the inversion step
        # current_t: where we are now
        # next_t: where we're going (one step noisier)
        current_t = max(0, t_int - (T // invert_steps))
        next_t = t_int

        alpha_t = alphas_cumprod[current_t].to(device=device, dtype=latents.dtype)
        alpha_t_next = alphas_cumprod[next_t].to(device=device, dtype=latents.dtype)

        # Inversion step: re-arrange DDIM to get x(t+1) from x(t)
        # This adds noise progressively to go from clean to noisy
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        sqrt_one_minus_alpha_t_next = torch.sqrt(1.0 - alpha_t_next)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_next = torch.sqrt(alpha_t_next)

        # Inverted update step (HF formula)
        latents = (latents - sqrt_one_minus_alpha_t * noise_pred) * (
            sqrt_alpha_t_next / sqrt_alpha_t
        ) + sqrt_one_minus_alpha_t_next * noise_pred
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


def _ddim_sample(
    module: DiffusionModule,
    labels: Tensor,
    sampling_steps: int,
    device: torch.device,
    structure_latent: Optional[Tensor] = None,
    start_step: int = 20,
    guidance_scale: float | None = None,
) -> Tensor:
    """DDIM sampling (deterministic when eta=0)."""
    num_samples = labels.shape[0]
    height = module.cfg.dataset.image_size
    T = module.diff_cfg.num_train_timesteps

    # Use provided guidance_scale or fall back to config
    if guidance_scale is None:
        guidance_scale = module.diff_cfg.guidance_scale

    if sampling_steps > T:
        raise ValueError(
            f"sampling_steps={sampling_steps} must be <= num_train_timesteps={T} for DDIM sampling."
        )

    if structure_latent is not None:
        latents = structure_latent[-(start_step + 1)][None, :].repeat(
            num_samples, 1, 1, 1
        )
    else:
        single_latent = torch.randn(
            1,
            module.cfg.model.latent_channels,
            height // 8,
            height // 8,
            device=device,
            dtype=torch.float32,
        )
        latents = single_latent.repeat(num_samples, 1, 1, 1)

    alphas_cumprod = module.alphas_cumprod  # (T,)

    # Choose a subset of timesteps (T-1 down to 0)
    timesteps = torch.linspace(
        T - 1,
        0,
        steps=sampling_steps,
        dtype=torch.long,
        device=device,
    )

    con_embed = module.ordinal_embedder(labels, is_training=False, unconditional=False)
    uncond_embed = module.ordinal_embedder.get_negative_embedding(
        labels, is_training=False
    )
    embed = torch.cat([uncond_embed, con_embed], dim=0)  # (2*B, D)

    # HF approach: skip first start_step iterations if using inverted latent
    for i, t_scalar in enumerate(timesteps):
        # Skip early steps if we're starting from inverted latent
        if structure_latent is not None and i < start_step:
            continue

        t_int = int(t_scalar.item())
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=device)

        latent_model_input = torch.cat([latents, latents], dim=0)  # (2*B, 4, H, W)

        t_model_input = torch.cat([t, t], dim=0)  # (2*B,)

        noise_pred = module(latent_model_input, t_model_input, embed)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        eps_theta = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        alpha_bar_t = alphas_cumprod[t_int]
        alpha_bar_t = alpha_bar_t.to(device=latents.device, dtype=latents.dtype)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x0_pred = (latents - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t

        # Clip x0 prediction for stability (latent space typically in [-4, 4] range)
        x0_pred = x0_pred.clamp(-4.0, 4.0)

        # At the final step (t=0), return x0_pred directly
        # Going beyond t=0 with alpha_bar_prev=1.0 causes artifacts
        if i == sampling_steps - 1:
            latents = x0_pred
            continue

        t_prev_int = int(timesteps[i + 1].item())
        alpha_bar_prev = alphas_cumprod[t_prev_int]
        alpha_bar_prev = alpha_bar_prev.to(device=latents.device, dtype=latents.dtype)

        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

        # Deterministic DDIM
        latents = (
            sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * eps_theta
        )

    return latents


def _latents_to_images(module: DiffusionModule, latents: Tensor) -> Tensor:
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


def _save_sequence(images: Tensor, labels: Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = images.cpu()
    for idx, (image, label) in enumerate(zip(images, labels)):
        array = (image.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
        pil_image = Image.fromarray(array)
        filename = output_dir / f"mes_{label.item():.2f}_{idx:02d}.png"
        pil_image.save(filename)

    # Also save a combined grid image for quick visualization
    grid = torchvision.utils.make_grid(images, nrow=images.shape[0] // 4, padding=2)
    array = (grid.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
    pil_grid = Image.fromarray(array)
    pil_grid.save(output_dir / "mes_progression_grid.png")


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)  # Use seed from command line argument

    cfg = _load_config(args.config)

    module = DiffusionModule.load_from_checkpoint(
        str(args.checkpoint),
        cfg=cfg,
    )
    module = module.to(device)
    module = module.to(torch.float32)
    module.eval()

    labels = _build_labels(
        args.mes_steps, end=cfg.dataset.num_classes - 1, device=device
    )

    with torch.no_grad():
        if args.use_ddim_invert and args.structure_image:
            # DDIM Inversion approach: convert structure image to latent conditioning
            print(f"📸 Loading structure image from: {args.structure_image}")
            structure_image = _load_structure_image(
                args.structure_image,
                target_size=cfg.dataset.image_size,
                device=device,
            )

            print(
                f"🔄 Inverting structure image to latent space ({args.ddim_invert_steps} steps)..."
            )
            structure_latent = _ddim_invert(
                module,
                structure_image,
                invert_steps=args.ddim_invert_steps,
                device=device,
            )

            print("🎨 Generating progression with inverted latent conditioning...")
            latents = _ddim_sample(
                module,
                labels,
                args.sampling_steps,
                device,
                structure_latent=structure_latent,
                start_step=20,
                guidance_scale=args.guidance_scale,
            )
        else:
            # Standard DDIM sampling without conditioning
            latents = _ddim_sample(
                module,
                labels,
                args.sampling_steps,
                device,
                guidance_scale=args.guidance_scale,
            )

        images = _latents_to_images(module, latents)
        _save_sequence(images, labels, args.output_dir)

    print(f"✅ Saved {len(labels)} progression images to {args.output_dir}")


if __name__ == "__main__":
    main()
