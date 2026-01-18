"""
Image Encoder for patient-specific anatomical conditioning.

This module provides image encoding capabilities using CLIP or similar
vision encoders to extract structural features from endoscopic images.
The encoded features are used to maintain anatomical consistency during
disease progression synthesis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class ImageEncoder(nn.Module):
    """
    Wrapper for CLIP Vision Encoder to extract image features.

    The encoder is FROZEN during training - we only train the projection layer.
    This follows the IP-Adapter approach where pretrained vision knowledge
    is preserved while learning task-specific projections.
    """

    def __init__(
        self,
        pretrained_path: str = "openai/clip-vit-base-patch16",
        torch_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_path,
            local_files_only=local_files_only,
        )

        # Freeze the image encoder - we don't train CLIP weights
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

        # Store dimensions for projection layer
        self.hidden_size = self.image_encoder.config.hidden_size  # 1024 for large
        self.projection_dim = self.image_encoder.config.projection_dim  # 768 for large

    @torch.no_grad()
    def forward(self, clip_images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features using CLIP.

        Args:
            clip_images: Tensor preprocessed by CLIPImageProcessor (B, C, 224, 224)
                        Already normalized with CLIP mean/std

        Returns:
            Image embeddings of shape (B, projection_dim)
        """
        # Input is already preprocessed by CLIPImageProcessor - pass directly
        outputs = self.image_encoder(
            pixel_values=clip_images, output_hidden_states=True
        )

        # Return the projected embedding (B, projection_dim)
        return outputs.image_embeds

    @torch.no_grad()
    def get_hidden_states(self, clip_images: torch.Tensor) -> torch.Tensor:
        """
        Get full hidden states (all patches) for richer conditioning.

        Args:
            clip_images: Tensor preprocessed by CLIPImageProcessor (B, C, 224, 224)

        Returns:
            Hidden states of shape (B, num_patches+1, hidden_size)
        """
        outputs = self.image_encoder(
            pixel_values=clip_images, output_hidden_states=True
        )

        # Return last hidden state (includes CLS token + patch tokens)
        return outputs.hidden_states[-1]  # (B, 257, 1024) for ViT-L/14


class ImageProjection(nn.Module):
    """
    Projects CLIP image embeddings to UNet cross-attention dimension.

    This is the TRAINABLE part that learns to map image features
    to a format the UNet can understand for structural conditioning.

    Based on IP-Adapter's ImageProjModel.
    """

    def __init__(
        self,
        clip_embedding_dim: int = 512,  # CLIP projection dim
        cross_attention_dim: int = 768,  # UNet cross-attention dim
        num_tokens: int = 4,  # Number of image tokens to produce
    ) -> None:
        super().__init__()

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

        # Project CLIP embedding to multiple tokens
        self.projection = nn.Linear(
            clip_embedding_dim, cross_attention_dim * num_tokens
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project image embeddings to cross-attention tokens.

        Args:
            image_embeds: (B, clip_embedding_dim)

        Returns:
            Projected tokens: (B, num_tokens, cross_attention_dim)
        """
        # Project and reshape
        embeds = self.projection(image_embeds)  # (B, cross_attention_dim * num_tokens)
        embeds = embeds.reshape(-1, self.num_tokens, self.cross_attention_dim)
        embeds = self.norm(embeds)

        return embeds


class ImageProjectionPlus(nn.Module):
    """
    Enhanced projection using full hidden states (IP-Adapter Plus style).

    Uses a Perceiver Resampler to compress patch tokens into fixed number
    of tokens. Better for preserving spatial information.
    """

    def __init__(
        self,
        clip_hidden_dim: int = 768,  # CLIP hidden dimension
        cross_attention_dim: int = 768,  # UNet cross-attention dim
        num_tokens: int = 16,  # Output tokens (more than basic)
        num_heads: int = 8,
        depth: int = 2,
    ) -> None:
        super().__init__()

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

        # Learnable query tokens
        self.latents = nn.Parameter(
            torch.randn(1, num_tokens, cross_attention_dim) * 0.02
        )

        # Input projection if dimensions don't match
        self.proj_in = (
            nn.Linear(clip_hidden_dim, cross_attention_dim)
            if clip_hidden_dim != cross_attention_dim
            else nn.Identity()
        )

        # Cross-attention layers (Perceiver style)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            cross_attention_dim, num_heads, batch_first=True
                        ),
                        "ff": nn.Sequential(
                            nn.Linear(cross_attention_dim, cross_attention_dim * 4),
                            nn.GELU(),
                            nn.Linear(cross_attention_dim * 4, cross_attention_dim),
                        ),
                        "norm1": nn.LayerNorm(cross_attention_dim),
                        "norm2": nn.LayerNorm(cross_attention_dim),
                    }
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = nn.LayerNorm(cross_attention_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to fixed number of tokens.

        Args:
            hidden_states: (B, num_patches+1, clip_hidden_dim)

        Returns:
            (B, num_tokens, cross_attention_dim)
        """
        batch_size = hidden_states.shape[0]

        # Project input
        hidden_states = self.proj_in(hidden_states)  # (B, N, cross_attention_dim)

        # Expand latent queries for batch
        latents = self.latents.expand(batch_size, -1, -1)  # (B, num_tokens, D)

        # Apply cross-attention layers
        for layer in self.layers:
            # Cross-attention: queries attend to image patches
            residual = latents
            latents = layer["norm1"](latents)
            latents, _ = layer["cross_attn"](
                query=latents,
                key=hidden_states,
                value=hidden_states,
            )
            latents = residual + latents

            # Feed-forward
            residual = latents
            latents = layer["norm2"](latents)
            latents = layer["ff"](latents)
            latents = residual + latents

        return self.norm_out(latents)
