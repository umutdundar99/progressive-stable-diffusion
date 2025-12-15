"""
Stable Diffusion v1.4 VAE Wrapper
---------------------------------

This module wraps the pretrained Stable Diffusion AutoencoderKL
and exposes a clean interface for encoding/decoding images within
the diffusion training pipeline.

Key requirements:
- Frozen VAE (no gradient updates)
- Images must be normalized to [-1, 1] before encoding
- Latent scaling (0.18215) is handled OUTSIDE this module inside
  the diffusion module (during training & inference)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl import (
    AutoencoderKLOutput,
    DecoderOutput,
)
from torch import Tensor, nn

PretrainedPath = Union[str, Path]


class SDVAE(nn.Module):
    """
    Wrapper around the Stable Diffusion v1.4 AutoencoderKL.

    - Loads pretrained SD VAE weights
    - Freezes all parameters (VAE is never trained)
    - Provides encode() and decode() with no_grad
    """

    def __init__(
        self,
        pretrained_path: PretrainedPath,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        local_files_only: bool = False,
    ) -> None:
        """
        Args:
            pretrained_path:
                Path to folder containing SD VAE weights (e.g. CompVis/stable-diffusion-v1-4).
            torch_dtype:
                Optional dtype override (e.g., `torch.float16`)
            local_files_only:
                If True, diffusers will NOT try to download weights.
        """
        super().__init__()

        # Load the pretrained AutoencoderKL
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            str(pretrained_path),
            subfolder="vae",
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

        # Set to eval & freeze parameters
        self.vae.eval()
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
        *,
        return_dict: bool = True,
    ) -> AutoencoderKLOutput:
        """
        Encode RGB images into a latent Gaussian distribution.

        Args:
            images: Tensor of shape (B, 3, H, W) normalized to [-1, 1].
            return_dict: Whether to return AutoencoderKLOutput.

        Returns:
            AutoencoderKLOutput containing latent distribution (mu, logvar).
        """
        return self.vae.encode(images, return_dict=return_dict)

    @torch.no_grad()
    def decode(
        self,
        latents: Tensor,
        *,
        return_dict: bool = True,
    ) -> DecoderOutput | Tensor:
        """
        Decode latent vectors back into RGB images.

        Args:
            latents: Tensor of shape (B, 4, H/8, W/8)
                     IMPORTANT: For Stable Diffusion, latents must be
                     *unscaled* inside the decoder:
                        decoded = vae.decode(latents / 0.18215)
                     This division should happen OUTSIDE this method.

            return_dict: Whether to return a DecoderOutput.

        Returns:
            RGB image in [-1, 1].
        """
        return self.vae.decode(latents, return_dict=return_dict)
