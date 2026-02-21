"""
Feature Purifier: Embedding-level disease erasure for IP-Adapter features.

Inspired by MaskST (masking via cross-modal attention) and GLoCE (concept
erasure through projection).  The purifier sits between the Perceiver
Resampler (ImageProjectionPlus) and the UNet, operating on the 16×768
anatomy tokens *before* they enter cross-attention.

Operation:
    1. Cross-attend: e_img (query) attends to source_aoe (key/value)
       → extracts the disease component that correlates with the source
       Mayo level.
    2. Gate: A learned sigmoid gate decides per-channel how much of the
       disease component to subtract.
    3. Residual subtraction: e_clean = e_img − gate ⊙ disease_component
    4. LayerNorm for stable downstream conditioning.

During training source == target (same Mayo label), so the purifier learns
which channels/dimensions carry disease info.  At inference, the real
source label of the input image is used → correct disease removal.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeaturePurifier(nn.Module):
    """Remove disease-correlated information from IP-Adapter image tokens.

    Args:
        dim: Token dimension (must match cross_attention_dim, typically 768).
        num_heads: Number of attention heads for disease detection.
        ff_mult: Feed-forward expansion factor for the gate MLP.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim

        # ── Disease detection via cross-attention ──
        # Query = image tokens, Key/Value = source AOE tokens
        # Output = disease component present in each image token
        self.norm_img = nn.LayerNorm(dim)
        self.norm_aoe = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # ── Channel-wise gate ──
        # Decides per-dimension how strongly to erase disease.
        # Input: concat(disease_component, e_img) → gate ∈ (0, 1)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.Sigmoid(),
        )

        # ── Output normalisation ──
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        image_embeds: torch.Tensor,
        source_aoe: torch.Tensor,
    ) -> torch.Tensor:
        """Purify image tokens by removing disease-correlated components.

        Args:
            image_embeds: IP-Adapter tokens after Perceiver Resampler
                          shape ``(B, N_img, D)`` — typically ``(B, 16, 768)``.
            source_aoe:   AOE tokens for the **source** (input image) Mayo level
                          shape ``(B, N_aoe, D)`` — typically ``(B, 16, 768)``.

        Returns:
            Cleaned image tokens ``(B, N_img, D)``.
        """
        # Normalise inputs for stable attention
        img_normed = self.norm_img(image_embeds)  # (B, N_img, D)
        aoe_normed = self.norm_aoe(source_aoe)  # (B, N_aoe, D)

        # Cross-attention: image tokens query the source disease embedding
        # → extracts what disease information is present in the image tokens
        disease_component, _ = self.cross_attn(
            query=img_normed,
            key=aoe_normed,
            value=aoe_normed,
        )  # (B, N_img, D)

        # Gate: learn which channels to erase
        gate_input = torch.cat(
            [disease_component, img_normed], dim=-1
        )  # (B, N_img, 2D)
        mask = self.gate(gate_input)  # (B, N_img, D) ∈ (0, 1)

        # Residual subtraction: erase disease component
        e_clean = image_embeds - mask * disease_component

        return self.norm_out(e_clean)
