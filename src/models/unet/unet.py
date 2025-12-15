"""
Ordinal-conditioned UNet wrapper around Stable Diffusion v1.4 UNet.

This module:
- Uses the exact UNet2DConditionModel architecture and weights from
  Stable Diffusion v1.4 (via diffusers)
- Replaces CLIP text embeddings with learned ordinal embeddings
  (e.g., BOE / AOE outputs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import UNet2DConditionModel
from torch import Tensor, nn


@dataclass
class UNetConfig:
    """
    Configuration for the ordinal-conditioned UNet wrapper.

    Attributes:
        pretrained_unet_path:
            Hugging Face model id or local path containing the SD UNet
            (e.g., "CompVis/stable-diffusion-v1-4").
        conditioning_dim:
            Dimensionality of the ordinal embedding (e.g., 768).
        in_channels:
            Number of input channels to the UNet (SD uses 4 for latents).
        out_channels:
            Number of output channels (usually same as in_channels).
        torch_dtype:
            Optional dtype to load the UNet weights with (e.g., torch.float16).
        local_files_only:
            If True, diffusers will not attempt to download weights and will
            only load from the local cache / path.
    """

    pretrained_unet_path: str = "CompVis/stable-diffusion-v1-4"
    conditioning_dim: int = 768
    in_channels: int = 4
    out_channels: int = 4
    torch_dtype: Optional[torch.dtype] = None
    local_files_only: bool = False


class OrdinalUNet(nn.Module):
    """
    Wrapper for Stable Diffusion v1.4 UNet with ordinal conditioning.

    This uses the official UNet2DConditionModel architecture and weights.
    Instead of CLIP text embeddings, it consumes a single ordinal embedding
    vector per sample (e.g., BOE/AOE output).

    Expected interfaces:
        - latents: (B, 4, H, W)
        - timesteps: (B,) or scalar tensor
        - cond_embed: (B, D) where D = conditioning_dim
    """

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config

        # Load pretrained Stable Diffusion UNet
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            config.pretrained_unet_path,
            subfolder="unet",
            torch_dtype=config.torch_dtype,
            local_files_only=config.local_files_only,
        )

        # Sanity checks for config consistency
        if self.unet.config.in_channels != config.in_channels:
            raise ValueError(
                f"UNet in_channels mismatch: "
                f"{self.unet.config.in_channels} (from weights) vs {config.in_channels} (config)."
            )

        if self.unet.config.out_channels != config.out_channels:
            raise ValueError(
                f"UNet out_channels mismatch: "
                f"{self.unet.config.out_channels} (from weights) vs {config.out_channels} (config)."
            )

        if self.unet.config.cross_attention_dim != config.conditioning_dim:
            raise ValueError(
                f"UNet cross_attention_dim mismatch: "
                f"{self.unet.config.cross_attention_dim} (from weights) vs {config.conditioning_dim} (config)."
            )

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        cond_embed: Tensor,
    ) -> Tensor:
        """
        Forward pass through the SD UNet with ordinal conditioning.

        Args:
            latents:
                Latent tensor of shape (B, 4, H, W).
                IMPORTANT: For Stable Diffusion, these are typically already
                scaled by a factor such as 0.18215 outside this module.
            timesteps:
                Timestep tensor of shape (B,) or scalar, indicating the
                diffusion step for each sample.
            cond_embed:
                Ordinal embedding tensor of shape (B, D), where
                D = config.conditioning_dim. This will be treated as a
                single "token" per sample for cross-attention.

        Returns:
            Predicted noise tensor of shape (B, 4, H, W).
        """
        if cond_embed.ndim != 2:
            raise ValueError(
                f"cond_embed must have shape (B, D), got {cond_embed.shape}"
            )

        encoder_hidden_states = cond_embed.unsqueeze(1)  # (B, 1, D)

        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        elif timesteps.ndim > 1:
            timesteps = timesteps.view(-1)

        timesteps = timesteps.to(latents.device)

        out = self.unet(
            sample=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )

        return out.sample
