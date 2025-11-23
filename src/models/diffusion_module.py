from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl

from .vae import SDVAE
from .ordinal_embedder import BasicOrdinalEmbedder, AdditiveOrdinalEmbedder
from .unet import OrdinalUNet, UNetConfig


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


class DiffusionModule(pl.LightningModule):
    """
    LightningModule wrapping the full diffusion training pipeline:

    - SD VAE (frozen)
    - Ordinal-conditioned UNet (Stable Diffusion v1.4 style)
    - EMA weights for stable sampling
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
        )


        if getattr(cfg.model, "use_pretrained_vae", True):
            vae_path = getattr(cfg.model, "pretrained_vae_path", cfg.model.pretrained_unet_path)
            self.vae = SDVAE(
                pretrained_path=vae_path,
                torch_dtype=torch.float16 if self.cfg.training.precision == 16 else torch.float32,
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

        # EMA copy of UNet for stable inference
        self.ema_unet = OrdinalUNet(unet_config)
        self._init_ema()

        betas, alphas_cumprod = self._build_noise_schedule()
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

        # Previous cumulative alphas (for sampling; even if not used yet)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]],
            dim=0,
        )
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=False)

        # For Min-SNR computation
        snr_values = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8)
        self.register_buffer("snr_values", snr_values, persistent=False)

        # Save hyperparameters for reproducibility (excluding non-serializable objects)
        self.save_hyperparameters(ignore=["vae", "unet", "ema_unet", "ordinal_embedder"])

    def _init_ema(self) -> None:
        """Initialize EMA weights to match the UNet."""
        self.ema_unet.load_state_dict(self.unet.state_dict())
        for p in self.ema_unet.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _update_ema(self, decay: float) -> None:
        """Update EMA weights using the current UNet parameters."""
        ema_params = dict(self.ema_unet.named_parameters())
        model_params = dict(self.unet.named_parameters())
        for name, param in model_params.items():
            ema_param = ema_params[name]
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


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
        clipped = torch.minimum(snr, torch.tensor(gamma, device=snr.device, dtype=snr.dtype))
        weight = clipped / (snr + 1e-8)
        return weight


    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        cond_embed: Tensor,
        use_ema: bool = True,
    ) -> Tensor:
        """
        Forward pass through UNet (EMA or raw) to predict noise.

        Args:
            latents: (B, 4, H, W)
            timesteps: (B,)
            cond_embed: (B, D)
            use_ema: Whether to use EMA weights (for inference).

        Returns:
            Predicted noise of shape (B, 4, H, W)
        """
        model = self.ema_unet if use_ema else self.unet
        return model(latents, timesteps, cond_embed)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        images, labels = batch  # images: (B, 3, H, W), labels: (B,) continuous MES

        # Encode images to latents and apply SD latent scaling
        vae_output = self.vae.encode(images)
        latents = vae_output.latent_dist.sample() * self.diff_cfg.latent_scale  # (B, 4, H/8, W/8)

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        t = self._sample_timesteps(latents.shape[0])

        # Add noise according to q(x_t | x_0)
        noisy_latents = self._q_sample(latents, t, noise)

        # Obtain ordinal embeddings from continuous labels
        cond_embed = self.ordinal_embedder(labels)  # (B, D)

        # Predict noise with UNet (raw weights for training)
        noise_pred = self.unet(noisy_latents, t, cond_embed)

        # Compute MSE loss
        base_loss = F.mse_loss(noise_pred, noise, reduction="none")
        base_loss = base_loss.mean(dim=(1, 2, 3))  # per-sample

        # Apply Min-SNR weighting
        weight = self._min_snr_weight(t)
        loss = (weight * base_loss).mean()

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss_base", base_loss.mean(), on_step=True, on_epoch=False)
        self.log("train/min_snr_weight_mean", weight.mean(), on_step=True, on_epoch=False)

        # EMA update
        if (self.global_step + 1) % self.diff_cfg.ema_update_interval == 0:
            ema_decay = self.cfg.training.ema_decay
            self._update_ema(decay=ema_decay)

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        opt_cfg = self.cfg.optimizer

        if opt_cfg.name.lower() != "adamw":
            raise NotImplementedError(f"Only AdamW is implemented, got: {opt_cfg.name}")

        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
            eps=opt_cfg.eps,
        )

        # Cosine schedule with warmup (step-based)
        sched_cfg = self.cfg.scheduler
        warmup_steps = sched_cfg.warmup_steps
        max_steps = sched_cfg.max_steps
        min_lr = sched_cfg.min_lr
        base_lr = opt_cfg.lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            # Cosine decay from base_lr to min_lr
            cosine = 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(progress)))
            return float(min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
