"""
Ordinal embedding strategies (BOE & AOE) used for progressive disease conditioning.

These embedders take a real-valued ordinal label y ∈ [0, K−1]
and return a D-dimensional embedding vector via interpolation.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



def _linear_interpolate(
    table: torch.Tensor,
    lower_idx: torch.Tensor,
    upper_idx: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Safely interpolate between table[lower_idx] and table[upper_idx].

    Args:
        table: (K, D)
        lower_idx: (...,)
        upper_idx: (...,)
        alpha: (...,) or (...,1)

    Returns:
        (..., D)
    """

    # Ensure alpha is broadcastable
    if alpha.dim() == lower_idx.dim():
        alpha = alpha.unsqueeze(-1)

    emb_lo = F.embedding(lower_idx, table)     # (..., D)
    emb_hi = F.embedding(upper_idx, table)     # (..., D)

    return emb_lo * (1.0 - alpha) + emb_hi * alpha



class BasicOrdinalEmbedder(nn.Module):
    """
    BOE (Basic Ordinal Embedding)
    - Learns an embedding per discrete class
    - Linearly interpolates between class vectors for real-valued labels
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2 for ordinal interpolation.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.normal_(self.embeddings, mean=0.0, std=init_std)

        # OPTIONAL: zero out padding embedding
        if padding_idx is not None:
            with torch.no_grad():
                self.embeddings[padding_idx].zero_()

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Tensor scalar or (...,) with values in [0, K-1]

        Returns:
            (..., D)
        """

        # Normalize scalar input
        scalar_input = labels.dim() == 0
        if scalar_input:
            labels = labels.unsqueeze(0)

        # Clamp into valid ordinal bounds
        max_idx = self.num_classes - 1
        labels = labels.to(self.embeddings.dtype)
        labels = torch.clamp(labels, 0.0, float(max_idx))

        # Compute bounds
        lower = torch.floor(labels)
        upper = torch.clamp(lower + 1, max=max_idx)
        alpha = labels - lower

        out = _linear_interpolate(
            self.embeddings,
            lower.long(),
            upper.long(),
            alpha,
        )

        return out.squeeze(0) if scalar_input else out



class AdditiveOrdinalEmbedder(nn.Module):
    """
    AOE (Additive Ordinal Embedding)
    - Uses a base vector + cumulative learned deltas
    - Naturally encodes monotonic progression for severity models
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2 for ordinal modeling.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Base vector
        self.base = nn.Parameter(torch.zeros(embedding_dim))

        # Deltas between ordinals
        self.deltas = nn.Parameter(torch.empty(num_classes - 1, embedding_dim))
        nn.init.normal_(self.deltas, mean=0.0, std=init_std)

    def _compute_class_table(self) -> torch.Tensor:
        """
        Returns:
            Table of shape (K, D) with:
            E[0], E[1], ..., E[K-1]
        where E[k] = base + sum(deltas[:k])
        """
        cumulative = torch.cumsum(self.deltas, dim=0)  # (K-1, D)
        offsets = torch.cat(
            [
                torch.zeros(1, self.embedding_dim, device=cumulative.device, dtype=cumulative.dtype),
                cumulative,
            ],
            dim=0,
        )
        return self.base.unsqueeze(0) + offsets

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: scalar or (...,) in [0, K-1]

        Returns:
            (..., D)
        """

        # Scalar case
        scalar_input = labels.dim() == 0
        if scalar_input:
            labels = labels.unsqueeze(0)

        table = self._compute_class_table()

        # Clamp
        max_idx = self.num_classes - 1
        labels = labels.to(table.dtype)
        labels = torch.clamp(labels, 0.0, float(max_idx))

        # Bounds
        lower = torch.floor(labels)
        upper = to
