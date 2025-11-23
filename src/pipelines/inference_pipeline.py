"""Run DDPM inference to visualize a progression across MES severity."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor

from src.models.diffusion_module import DiffusionModule


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate MES progression samples from a trained DDPM.")
	parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a Lightning checkpoint produced during training.")
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
	parser.add_argument("--mes-steps", type=int, default=13, help="Number of MES values to sweep from 0 to 3.")
	parser.add_argument("--sampling-steps", type=int, default=50, help="Number of DDPM timesteps to roll out during sampling.")
	parser.add_argument("--device", type=str, default="auto", help="Device used for inference (auto, cpu, cuda).")
	parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling.")
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


def _build_labels(num_steps: int, start: float = 0.0, end: float = 3.0, device: torch.device | None = None) -> Tensor:
	device = device or torch.device("cpu")
	if num_steps <= 0:
		raise ValueError("`mes_steps` must be a positive integer.")
	return torch.linspace(start, end, steps=num_steps, device=device, dtype=torch.float32)


def _ddpm_sample(
	module: DiffusionModule,
	labels: Tensor,
	sampling_steps: int,
	device: torch.device,
) -> Tensor:
	num_samples = labels.shape[0]
	height = module.cfg.dataset.image_size
	latents = torch.randn(
		num_samples,
		module.unet.in_channels,
		height // 8,
		height // 8,
		device=device,
	)

	for step in reversed(range(sampling_steps)):
		t = torch.full((num_samples,), step, dtype=torch.long, device=device)
		cond_embed = module.ordinal_embedder(labels)
		pred_noise = module.unet(latents, t, cond_embed)
		alpha = module.alphas_cumprod[t]
		alpha_prev = module.alpha_cumprod_prev[t]
		beta = module.betas[t]
		sqrt_recip_alpha = torch.sqrt(1.0 / alpha)
		sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha)
		model_mean = sqrt_recip_alpha * (latents - beta / sqrt_one_minus_alpha * pred_noise)
		if step > 0:
			noise = torch.randn_like(latents)
			sigma = torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha) * beta)
			latents = model_mean + sigma * noise
		else:
			latents = model_mean
	return latents


def _latents_to_images(latents: Tensor) -> Tensor:
	latents = torch.clamp(latents, -1.0, 1.0)
	channels = latents.shape[1]
	if channels >= 3:
		visuals = latents[:, :3]
	else:
		repeats = (3 + channels - 1) // channels
		visuals = latents.repeat_interleave(repeats, dim=1)[:, :3]
	visuals = (visuals + 1.0) / 2.0
	return visuals.clamp(0.0, 1.0)


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
	_set_seed(args.seed)
	cfg = _load_config(args.config)
	module = DiffusionModule.load_from_checkpoint(str(args.checkpoint), cfg=cfg)
	module = module.to(device)
	module.eval()

	labels = _build_labels(args.mes_steps, device=device)
	with torch.no_grad():
		samples = _ddpm_sample(module, labels, args.sampling_steps, device)
		visuals = _latents_to_images(samples)
		_save_sequence(visuals, labels, args.output_dir)

	print(f"Saved {len(labels)} progression images to {args.output_dir}")


if __name__ == "__main__":
	main()
