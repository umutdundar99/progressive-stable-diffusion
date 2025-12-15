"""Run DDPM / DDIM inference to visualize a progression across MES severity."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor

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


def _ddim_sample(
    module: DiffusionModule,
    labels: Tensor,
    sampling_steps: int,
    device: torch.device,
    eta: float = 0.0,
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
    # "Mayo 0 (normal mucosa) serves as the negative prompt for higher severity levels"
    # "Mayo 1 serves as the negative prompt for Mayo 0"
    # This emphasizes distinctive features by contrasting against other severity levels
    uncond_embed = module.ordinal_embedder.get_negative_embedding(
        labels, is_training=False
    )
    embed = torch.cat([uncond_embed, con_embed], dim=0)  # (2*B, D)

    for i, t_scalar in enumerate(timesteps):
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

    labels = _build_labels(args.mes_steps, device=device)

    with torch.no_grad():
        latents = _ddim_sample(
            module,
            labels,
            args.sampling_steps,
            device,
            eta=0.0,
            guidance_scale=args.guidance_scale,
        )

        images = _latents_to_images(module, latents)
        _save_sequence(images, labels, args.output_dir)

    print(f"Saved {len(labels)} progression images to {args.output_dir}")


if __name__ == "__main__":
    main()
