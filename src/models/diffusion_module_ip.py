"""
Diffusion Module with Image-Prompt (IP) Adapter for Patient-Specific Conditioning.

This module extends the base DiffusionModule to support:
1. AOE (Additive Ordinal Embedding) for disease severity
2. Image conditioning for patient-specific anatomical structure

The key innovation is the frequency-based attention strategy:
- High-resolution layers: Image features (anatomical details, edges, folds)
- Low-resolution layers: AOE features (global disease patterns, colors)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from .attention_processor import (
    set_ordinal_ip_attention_processors,
)
from .image_encoder import ImageEncoder, ImageProjection, ImageProjectionPlus
from .lr_scheduler import LinearWarmupCosineAnnealingLR
from .ordinal_embedder import AdditiveOrdinalEmbedder
from .unet import OrdinalUNet, UNetConfig
from .vae import SDVAE


@dataclass
class DiffusionIPConfig:
    """Configuration for diffusion with IP-Adapter."""

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
    image_encoder_path: str = "openai/clip-vit-base-patch16"
    num_image_tokens: int = 4
    image_scale: float = 1.0
    use_frequency_strategy: bool = True
    use_image_projection_plus: bool = False


class DiffusionModuleWithIP(pl.LightningModule):
    """
    Diffusion module combining ordinal disease conditioning with
    patient-specific anatomical conditioning via image features.

    Architecture:
    - VAE (frozen): Encode/decode images to latent space
    - Image Encoder (frozen): Extract CLIP features from structure image
    - Image Projection (trainable): Project image features to UNet dimension
    - AOE (trainable): Ordinal embeddings for disease severity
    - UNet (trainable): Noise prediction with dual conditioning
    - OrdinalIPAttnProcessor (trainable): Custom attention for both conditions
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

        # Build config
        self.diff_cfg = DiffusionIPConfig(
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
            image_encoder_path=getattr(
                cfg.model, "image_encoder_path", "openai/clip-vit-base-patch16"
            ),
            num_image_tokens=getattr(cfg.model, "num_image_tokens", 4),
            image_scale=getattr(cfg.model, "image_scale", 1.0),
            use_frequency_strategy=getattr(cfg.model, "use_frequency_strategy", True),
            use_image_projection_plus=getattr(
                cfg.model, "use_image_projection_plus", False
            ),
        )

        # VAE (FROZEN)
        vae_path = getattr(
            cfg.model, "pretrained_vae_path", cfg.model.pretrained_unet_path
        )
        self.vae = SDVAE(
            pretrained_path=vae_path,
            torch_dtype=torch.float16
            if cfg.training.precision == 16
            else torch.float32,
            local_files_only=False,
        )

        self.image_encoder = ImageEncoder(
            pretrained_path=self.diff_cfg.image_encoder_path,
            torch_dtype=torch.float32,
            local_files_only=False,
        )

        if self.diff_cfg.use_image_projection_plus:
            self.image_projection = ImageProjectionPlus(
                clip_hidden_dim=self.image_encoder.hidden_size,
                cross_attention_dim=cfg.model.conditioning_dim,
                num_tokens=self.diff_cfg.num_image_tokens,
            )
        else:
            self.image_projection = ImageProjection(
                clip_embedding_dim=self.image_encoder.projection_dim,
                cross_attention_dim=cfg.model.conditioning_dim,
                num_tokens=self.diff_cfg.num_image_tokens,
            )

        # ORDINAL EMBEDDER (TRAINABLE)
        emb_cfg = cfg.model.ordinal_embedder

        self.ordinal_embedder = AdditiveOrdinalEmbedder(
            num_classes=emb_cfg.num_classes,
            embedding_dim=cfg.model.embedding_dim,
            delta_scale=getattr(emb_cfg.aoe, "delta_scale", 0.1),
        )

        # UNET (TRAINABLE)
        unet_config = UNetConfig(
            pretrained_unet_path=cfg.model.pretrained_unet_path,
            conditioning_dim=cfg.model.conditioning_dim,
            in_channels=cfg.model.latent_channels,
            out_channels=cfg.model.latent_channels,
            local_files_only=False,
        )
        self.unet = OrdinalUNet(unet_config)
        self._setup_attention_processors()

        betas, alphas_cumprod = self._build_noise_schedule()
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], dim=0
        )
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev, persistent=False
        )

        self.blur_kernel_size = cfg.model["blur_kernel_size"]
        self.blur_sigma = cfg.model["blur_sigma"]

        # For Min-SNR computation
        snr_values = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8)
        self.register_buffer("snr_values", snr_values, persistent=False)

        # Save hyperparameters
        self.save_hyperparameters(
            ignore=["vae", "unet", "ordinal_embedder", "image_encoder"]
        )

        print("✅ DiffusionModuleWithIP initialized successfully!")
        self._print_trainable_params()

    def _setup_attention_processors(self) -> None:
        """Replace UNet attention processors with OrdinalIPAttnProcessors."""
        unet = self.unet.unet

        frequency_dominant_scale = self.cfg.model["frequency_dominant_scale"]
        frequency_non_dominant_scale = self.cfg.model["frequency_non_dominant_scale"]

        set_ordinal_ip_attention_processors(
            unet=unet,
            num_tokens=self.diff_cfg.num_image_tokens,
            scale=self.diff_cfg.image_scale,
            use_frequency_strategy=self.diff_cfg.use_frequency_strategy,
            frequency_dominant_scale=frequency_dominant_scale,
            frequency_non_dominant_scale=frequency_non_dominant_scale,
        )

    def _print_trainable_params(self) -> None:
        """Print trainable vs frozen parameters."""

        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        def count_all_params(module):
            return sum(p.numel() for p in module.parameters())

        print("\n📊 Parameter Summary:")
        print(
            f"  VAE: {count_params(self.vae):,} trainable / {count_all_params(self.vae):,} total (frozen)"
        )
        print(
            f"  Image Encoder: {count_params(self.image_encoder):,} trainable / {count_all_params(self.image_encoder):,} total (frozen)"
        )
        print(f"  Image Projection: {count_params(self.image_projection):,} trainable")
        print(f"  Ordinal Embedder: {count_params(self.ordinal_embedder):,} trainable")
        print(
            f"  UNet: {count_params(self.unet):,} trainable / {count_all_params(self.unet):,} total"
        )

        total_trainable = (
            count_params(self.image_projection)
            + count_params(self.ordinal_embedder)
            + count_params(self.unet)
        )
        print(f"  TOTAL TRAINABLE: {total_trainable:,}")

    def _build_noise_schedule(self) -> Tuple[Tensor, Tensor]:
        """Build linear beta schedule."""
        if self.diff_cfg.noise_schedule != "linear":
            raise NotImplementedError("Only linear noise schedule is supported.")

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
        """Sample random timesteps."""
        return torch.randint(
            0,
            self.diff_cfg.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def _q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: add noise to x0."""
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus * noise

    def _min_snr_weight(self, t: Tensor) -> Tensor:
        """Compute Min-SNR-γ weight."""
        if not self.cfg.training.use_min_snr_weighting:
            return torch.ones_like(t, dtype=torch.float32, device=self.device)

        gamma = self.diff_cfg.min_snr_gamma
        snr = self.snr_values[t]
        clipped = torch.minimum(snr, torch.tensor(gamma, device=snr.device))
        return clipped / (snr + 1e-8)

    def _get_image_embeds(self, structure_images: Tensor) -> Tensor:
        """
        Extract and project image embeddings for anatomical conditioning.

        Args:
            structure_images: CLIP-preprocessed images (B, 3, 224, 224)
                              Already ImageNet-normalized by CLIPImageProcessor

        Returns:
            Projected embeddings (B, num_tokens, cross_attention_dim)
        """

        if self.diff_cfg.use_image_projection_plus:
            image_embeds = self.image_encoder.get_hidden_states(structure_images)
        else:
            image_embeds = self.image_encoder(structure_images)

        # Project to UNet dimension (trainable)
        return self.image_projection(image_embeds)

    def _prepare_conditioning(
        self,
        labels: Tensor,
        structure_images: Tensor,
        is_training: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare combined conditioning: AOE + Image embeddings.

        Args:
            labels: Mayo scores (B,)
            structure_images: Blurred/depth images (B, 3, H, W)
            is_training: Whether in training mode

        Returns:
            Combined embeddings (B, aoe_tokens + image_tokens, D)
        """

        aoe_embeds = self.ordinal_embedder(labels, is_training=is_training)
        if aoe_embeds.dim() == 2:
            aoe_embeds = aoe_embeds.unsqueeze(1)

        image_embeds = self._get_image_embeds(structure_images)

        return aoe_embeds, image_embeds

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        cond_embed: Tensor,
    ) -> Tensor:
        """Forward pass through UNet."""
        return self.unet(latents, timesteps, cond_embed)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """
        Training step with dual conditioning.

        Following Meryem's methodology (Section 3.5):
        - AOE conditioning is ALWAYS provided (no CFG dropout for ordinal embeddings)
        - Image conditioning has CFG dropout (per-sample, not per-batch)

        Expects batch to contain:
        - images: Original endoscopy images (B, 3, H, W)
        - labels: Mayo scores (B,)
        - structure_images: CLIP-preprocessed structure images (B, 3, 224, 224)
        """

        images, labels, structure_images = batch
        batch_size = images.shape[0]

        # Encode to latent space
        vae_output = self.vae.encode(images)
        latents = vae_output.latent_dist.sample() * self.diff_cfg.latent_scale

        # Sample noise
        noise = torch.randn_like(latents)
        if self.diff_cfg.noise_offset > 0:
            noise_offset = self.diff_cfg.noise_offset * torch.randn(
                batch_size,
                latents.shape[1],
                1,
                1,
                device=latents.device,
                dtype=latents.dtype,
            )
            noise = noise + noise_offset

        # Sample timesteps and add noise
        t = self._sample_timesteps(batch_size)
        noisy_latents = self._q_sample(latents, t, noise)

        # Get AOE embeddings
        aoe_embeds, image_embeds = self._prepare_conditioning(
            labels, structure_images, is_training=True
        )

        cfg_drop_prob = getattr(self.cfg.model, "cfg_drop_prob", 0.1)
        drop_mask = torch.rand(batch_size, device=self.device) < cfg_drop_prob
        zero_image_embeds = torch.zeros_like(image_embeds)

        drop_mask_expanded = drop_mask.view(-1, 1, 1).expand_as(image_embeds)
        image_embeds = torch.where(drop_mask_expanded, zero_image_embeds, image_embeds)
        combined_embeds = torch.cat([aoe_embeds, image_embeds], dim=1)

        noise_pred = self.unet(noisy_latents, t, combined_embeds)

        base_loss = F.mse_loss(noise_pred, noise, reduction="none")
        base_loss = base_loss.mean(dim=(1, 2, 3))
        weight = self._min_snr_weight(t)
        loss = (weight * base_loss).mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/cfg_drop_rate",
            drop_mask.float().mean(),
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer with separate learning rates."""
        opt_cfg = self.cfg.optimizer

        # Group parameters with different learning rates
        params = [
            {"params": self.unet.parameters(), "lr": opt_cfg.lr},
            {"params": self.ordinal_embedder.parameters(), "lr": opt_cfg.lr},
            {"params": self.image_projection.parameters(), "lr": opt_cfg.lr * 2},
        ]

        optimizer = torch.optim.AdamW(
            params,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
        )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.cfg.scheduler.warmup_epochs,
            max_epochs=self.cfg.training.max_epochs,
            warmup_start_lr=opt_cfg.lr * 0.01,
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
