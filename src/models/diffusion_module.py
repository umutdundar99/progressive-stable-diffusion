from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from .lr_scheduler import LinearWarmupCosineAnnealingLR
from .ordinal_embedder import AdditiveOrdinalEmbedder, BasicOrdinalEmbedder
from .unet import OrdinalUNet, UNetConfig
from .vae import SDVAE


@dataclass
class DiffusionConfig:
    num_train_timesteps: int
    beta_start: float
    beta_end: float
    noise_schedule: str = "linear"
    sampling_steps: int = 50
    guidance_scale: float = 2.0
    min_snr_gamma: float = 1.0
    ema_update_interval: int = 10
    latent_scale: float = 0.18215
    noise_offset: float = 0.0
    input_perturbation: float = 0.0


class DiffusionModule(pl.LightningModule):
    """
    LightningModule wrapping the full diffusion training pipeline:

    - SD VAE (frozen)
    - Ordinal-conditioned UNet (Stable Diffusion v1.4 style)
    - Min-SNR gamma weighting for loss
    """

    def __init__(self, cfg: Any) -> None:
        """
        Args:
            cfg: Hydra config object (train.yaml) with fields:
                 - cfg.model
                 - cfg.dataset
                 - cfg.training
                 - cfg.optimizer
                 - cfg.scheduler
                 - cfg.diffusion
        """
        super().__init__()
        self.cfg = cfg

        self.diff_cfg = DiffusionConfig(
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            noise_schedule=cfg.diffusion.noise_schedule,
            sampling_steps=cfg.diffusion.sampling_steps,
            guidance_scale=cfg.diffusion.guidance_scale,
            min_snr_gamma=cfg.diffusion.min_snr_gamma,
            ema_update_interval=cfg.diffusion.ema_update_interval,
            latent_scale=getattr(cfg.diffusion, "latent_scale", 0.18215),
            noise_offset=getattr(cfg.training, "noise_offset", 0.0),
            input_perturbation=getattr(cfg.training, "input_perturbation", 0.0),
        )

        if getattr(cfg.model, "use_pretrained_vae", True):
            vae_path = getattr(
                cfg.model, "pretrained_vae_path", cfg.model.pretrained_unet_path
            )
            self.vae = SDVAE(
                pretrained_path=vae_path,
                torch_dtype=torch.float16
                if self.cfg.training.precision == 16
                else torch.float32,
                local_files_only=False,
            )
        else:
            raise ValueError("This module expects a pretrained SD VAE.")

        emb_cfg = cfg.model.ordinal_embedder
        if emb_cfg.type.lower() == "boe":
            self.ordinal_embedder = BasicOrdinalEmbedder(
                num_classes=emb_cfg.num_classes,
                embedding_dim=cfg.model.embedding_dim,
            )
        elif emb_cfg.type.lower() == "aoe":
            self.ordinal_embedder = AdditiveOrdinalEmbedder(
                num_classes=emb_cfg.num_classes,
                embedding_dim=cfg.model.embedding_dim,
                delta_scale=getattr(emb_cfg.aoe, "delta_scale", 0.1),
            )
        else:
            raise ValueError(f"Unknown ordinal embedder type: {emb_cfg.type}")

        unet_config = UNetConfig(
            pretrained_unet_path=cfg.model.pretrained_unet_path,
            conditioning_dim=cfg.model.conditioning_dim,
            in_channels=cfg.model.latent_channels,
            out_channels=cfg.model.latent_channels,
        )
        self.unet = OrdinalUNet(unet_config)

        # Enable gradient checkpointing for memory efficiency
        if getattr(cfg.training, "gradient_checkpointing", False):
            if hasattr(self.unet.unet, "enable_gradient_checkpointing"):
                self.unet.unet.enable_gradient_checkpointing()
                print("Gradient checkpointing enabled for UNet")

        # torch.compile for speedup (PyTorch 2.0+)
        if getattr(cfg.training, "use_compile", False):
            compile_mode = getattr(cfg.training, "compile_mode", "reduce-overhead")
            try:
                self.unet = torch.compile(self.unet, mode=compile_mode)
                print(f"torch.compile enabled with mode='{compile_mode}'")
            except Exception as e:
                print(f"torch.compile failed: {e}. Continuing without compilation.")

        betas, alphas_cumprod = self._build_noise_schedule()
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

        # Previous cumulative alphas (for sampling; DDPM/DDIM)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]],
            dim=0,
        )
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev, persistent=False
        )

        # For Min-SNR computation
        snr_values = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8)
        self.register_buffer("snr_values", snr_values, persistent=False)

        self.save_hyperparameters(ignore=["vae", "unet", "ordinal_embedder"])

    def _build_noise_schedule(self) -> Tuple[Tensor, Tensor]:
        """
        Build beta schedule and cumulative alphas for the DDPM forward process.

        Currently only supports a linear beta schedule:
            beta_t = linspace(beta_start, beta_end, T)
        """
        if self.diff_cfg.noise_schedule != "linear":
            raise NotImplementedError(
                f"Only linear noise schedule is implemented, got: {self.diff_cfg.noise_schedule}"
            )

        betas = torch.linspace(
            self.diff_cfg.beta_start,
            self.diff_cfg.beta_end,
            self.diff_cfg.num_train_timesteps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return betas, alphas_cumprod

    def _sample_timesteps(self, batch_size: int) -> Tensor:
        """Sample random timesteps uniformly."""
        return torch.randint(
            low=0,
            high=self.diff_cfg.num_train_timesteps,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def _q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """
        Forward diffusion (add noise):
            x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        alphas_cumprod = self.alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus * noise

    def _min_snr_weight(self, t: Tensor) -> Tensor:
        """Compute Min-SNR-γ weight for each timestep."""
        if not self.cfg.training.use_min_snr_weighting:
            return torch.ones_like(t, dtype=torch.float32, device=self.device)

        gamma = self.diff_cfg.min_snr_gamma
        snr = self.snr_values[t]
        clipped = torch.minimum(
            snr, torch.tensor(gamma, device=snr.device, dtype=snr.dtype)
        )
        weight = clipped / (snr + 1e-8)
        return weight

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        cond_embed: Tensor,
    ) -> Tensor:
        """
        Forward pass through UNet to predict noise.

        Args:
            latents: (B, 4, H, W)
            timesteps: (B,)
            cond_embed: (B, D)

        Returns:
            Predicted noise of shape (B, 4, H, W)
        """
        return self.unet(latents, timesteps, cond_embed)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        images, labels = batch  # images: (B, 3, H, W), labels: (B,) continuous MES

        # Encode images to latents and apply SD latent scaling
        vae_output = self.vae.encode(images)
        latents = (
            vae_output.latent_dist.sample() * self.diff_cfg.latent_scale
        )  # (B, 4, H/8, W/8)

        # Sample noise with optional noise offset for better contrast
        # Reference: https://www.crosslabs.org/blog/diffusion-with-offset-noise
        noise = torch.randn_like(latents)
        if self.diff_cfg.noise_offset > 0:
            # Add channel-wise offset noise for better contrast in dark/bright regions
            noise_offset = self.diff_cfg.noise_offset * torch.randn(
                latents.shape[0],
                latents.shape[1],
                1,
                1,
                device=latents.device,
                dtype=latents.dtype,
            )
            noise = noise + noise_offset

        # Sample timesteps
        t = self._sample_timesteps(latents.shape[0])

        # Optional: Input perturbation for better training stability
        # Reference: https://arxiv.org/abs/2301.11706 (Common Diffusion Noise Schedules)
        if self.diff_cfg.input_perturbation > 0:
            new_noise = noise + self.diff_cfg.input_perturbation * torch.randn_like(
                noise
            )
            noisy_latents = self._q_sample(latents, t, new_noise)
        else:
            noisy_latents = self._q_sample(latents, t, noise)

        # Get ordinal conditioning embeddings
        cond_embed = self.ordinal_embedder(
            labels, is_training=True, unconditional=False
        )

        # Predict noise with UNet
        noise_pred = self.unet(noisy_latents, t, cond_embed)

        # Compute MSE loss (always against original noise, not perturbed)
        base_loss = F.mse_loss(noise_pred, noise, reduction="none")
        base_loss = base_loss.mean(dim=(1, 2, 3))  # per-sample

        # Apply Min-SNR weighting
        weight = self._min_snr_weight(t)
        loss = (weight * base_loss).mean()

        # Note: sync_dist=False for step logging (sync_dist=True causes GPU sync every step!)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_base", base_loss.mean(), on_step=False, on_epoch=True)
        self.log(
            "train/min_snr_weight_mean", weight.mean(), on_step=False, on_epoch=True
        )

        # Log embedding statistics every 500 steps to monitor for exploding/collapsing embeddings
        if batch_idx % 500 == 0:
            embed_stats = self.ordinal_embedder.log_embedding_stats()
            for key, value in embed_stats.items():
                self.log(key, value, on_step=True, on_epoch=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        opt_cfg = self.cfg.optimizer

        if opt_cfg.name.lower() != "adamw":
            raise NotImplementedError(f"Only AdamW is implemented, got: {opt_cfg.name}")

        params_to_optimize = list(self.unet.parameters()) + list(
            self.ordinal_embedder.parameters()
        )
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
        )

        # Learning rate scheduler
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.cfg.scheduler.warmup_epochs,
            max_epochs=self.cfg.training.max_epochs,
            warmup_start_lr=self.cfg.optimizer.lr * 0.01,
            eta_min=self.cfg.scheduler.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
