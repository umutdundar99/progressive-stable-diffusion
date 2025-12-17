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

    emb_lo = F.embedding(lower_idx, table)  # (..., D)
    emb_hi = F.embedding(upper_idx, table)  # (..., D)

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
        learnable_null: bool = True,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2 for ordinal interpolation.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.normal_(self.embeddings, mean=0.0, std=init_std)

        if learnable_null:
            self.null_embedding = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.register_buffer("null_embedding", torch.zeros(1, embedding_dim))

        # OPTIONAL: zero out padding embedding
        if padding_idx is not None:
            with torch.no_grad():
                self.embeddings[padding_idx].zero_()

    def forward(
        self,
        labels: torch.Tensor,
        is_training: bool = False,  # Kept for API consistency with AOE
        unconditional: bool = False,
        **kwargs,  # Accept but ignore extra args for compatibility
    ) -> torch.Tensor:
        """
        Args:
            labels: Tensor scalar or (...,) with values in [0, K-1]
            is_training: Unused for BOE (no regularization noise)
            unconditional: Return null embedding for CFG

        Returns:
            (..., D)
        """
        if unconditional:
            batch_size = labels.shape[0] if labels.dim() > 0 else 1
            return self.null_embedding.expand(batch_size, -1)

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

    def get_negative_embedding(
        self,
        labels: torch.Tensor,
        is_training: bool = False,
    ) -> torch.Tensor:
        """
        Get negative conditioning embeddings with SMOOTH interpolation.
        
        Per thesis Section 3.5:
        - Mayo 0 (normal mucosa) contrasts against Mayo 1 features
        - Mayo 1+ contrasts against Mayo 0 (healthy) features
        
        To avoid discontinuity at intermediate values, we smoothly interpolate:
        - At label=0: negative_label = 1.0 (contrast against mild disease)
        - At label=1: negative_label = 0.0 (contrast against healthy)
        - At label>=1: negative_label = 0.0 (contrast against healthy)
        
        This creates smooth CFG guidance across the severity spectrum.

        Args:
            labels: Tensor of MES values in [0, K-1]
            is_training: Whether in training mode

        Returns:
            Negative embeddings of shape (B, D)
        """
        scalar_input = labels.dim() == 0
        if scalar_input:
            labels = labels.unsqueeze(0)

        # Smooth interpolation of negative labels:
        # negative_label = max(0, 1 - label) for label in [0, 1]
        # negative_label = 0 for label > 1
        # This gives: label=0 -> neg=1, label=0.25 -> neg=0.75, label=0.5 -> neg=0.5, label=1+ -> neg=0
        negative_labels = torch.clamp(1.0 - labels, min=0.0, max=1.0)

        return self.forward(
            negative_labels, is_training=is_training, unconditional=False
        )

    def log_embedding_stats(self) -> dict:
        """Return embedding statistics for monitoring during training."""
        with torch.no_grad():
            emb = self.embeddings
            return {
                "embed/mean": emb.mean().item(),
                "embed/std": emb.std().item(),
                "embed/min": emb.min().item(),
                "embed/max": emb.max().item(),
                "embed/norm": emb.norm(dim=-1).mean().item(),
            }


class AdditiveOrdinalEmbedder(nn.Module):
    """
    AOE (Additive Ordinal Embedding)
    - Uses a base vector + cumulative learned deltas
    - Naturally encodes monotonic progression for severity models

    Each higher severity class builds upon the features
    of the previous class, adding new characteristics (cumulative pathological
    features) rather than replacing them entirely.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        init_std: float = 0.02,
        delta_scale: float = 0.1,
        learnable_null: bool = True,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2 for ordinal modeling.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Base vector (represents MES 0 - healthy mucosa)
        self.base = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.base, mean=0.0, std=init_std)

        # Deltas between ordinals - initialized with MONOTONICALLY INCREASING pattern
        # This ensures E[k] = base + sum(deltas[:k]) naturally increases
        self.deltas = nn.Parameter(torch.empty(num_classes - 1, embedding_dim))
        self._initialize_monotonic_deltas(init_std, delta_scale)

        if learnable_null:
            self.null_embedding = nn.Parameter(torch.zeros(1, embedding_dim))
        else:
            self.register_buffer("null_embedding", torch.zeros(1, embedding_dim))

    def _initialize_monotonic_deltas(self, init_std: float, delta_scale: float) -> None:
        """
        Initialize deltas to encourage monotonically increasing embeddings.

        Each delta represents additional pathological
        features that accumulate as disease severity increases.
        The initialization ensures:
        - All deltas have positive mean (adding features)
        - Small variance to allow learning while maintaining ordinal structure
        """
        with torch.no_grad():
            # Initialize with small positive bias to encourage monotonic increase
            # Each delta adds "new pathological features" on top of previous
            for i in range(self.num_classes - 1):
                # Positive mean ensures cumulative sum increases
                nn.init.normal_(self.deltas[i], mean=delta_scale, std=init_std)
                # Scale by severity level to emphasize progression
                self.deltas[i] *= 1.0 + 0.1 * i

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
                torch.zeros(
                    1,
                    self.embedding_dim,
                    device=cumulative.device,
                    dtype=cumulative.dtype,
                ),
                cumulative,
            ],
            dim=0,
        )
        return self.base.unsqueeze(0) + offsets

    def forward(
        self,
        labels: torch.Tensor,
        is_training: bool = False,  # Default False for safety during inference
        unconditional: bool = False,
        noise_std: float = 0.005,  # Reduced from 0.01 for more stable training
    ) -> torch.Tensor:
        """
        Args:
            labels: scalar or (...,) in [0, K-1]
            is_training: Whether in training mode (adds regularization noise)
            unconditional: Return null embedding for CFG
            noise_std: Std of Gaussian noise for training regularization

        Returns:
            (..., D)
        """
        if unconditional:
            batch_size = labels.shape[0] if labels.dim() > 0 else 1
            return self.null_embedding.expand(batch_size, -1)

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
        upper = torch.clamp(lower + 1, max=max_idx)
        alpha = labels - lower
        out = _linear_interpolate(
            table,
            lower.long(),
            upper.long(),
            alpha,
        )

        # Add Gaussian noise during training for regularization
        if is_training and noise_std > 0:
            noise = torch.randn_like(out) * noise_std
            out = out + noise

        return out.squeeze(0) if scalar_input else out

    def get_negative_embedding(
        self,
        labels: torch.Tensor,
        is_training: bool = False,
        noise_std: float = 0.005,  # Reduced from 0.01 for more stable training
    ) -> torch.Tensor:
        """
        Get negative conditioning embeddings with SMOOTH interpolation.
        
        Per thesis Section 3.5:
        - Mayo 0 (normal mucosa) contrasts against Mayo 1 features
        - Mayo 1+ contrasts against Mayo 0 (healthy) features
        
        To avoid discontinuity at intermediate values, we smoothly interpolate:
        - At label=0: negative_label = 1.0 (contrast against mild disease)
        - At label=1: negative_label = 0.0 (contrast against healthy)
        - At label>=1: negative_label = 0.0 (contrast against healthy)
        
        This creates smooth CFG guidance across the severity spectrum.

        Args:
            labels: Tensor of MES values in [0, K-1]
            is_training: Whether in training mode
            noise_std: Std of Gaussian noise for training regularization

        Returns:
            Negative embeddings of shape (B, D)
        """
        scalar_input = labels.dim() == 0
        if scalar_input:
            labels = labels.unsqueeze(0)

        # Smooth interpolation of negative labels:
        # negative_label = max(0, 1 - label) for label in [0, 1]
        # negative_label = 0 for label > 1
        # This gives: label=0 -> neg=1, label=0.25 -> neg=0.75, label=0.5 -> neg=0.5, label=1+ -> neg=0
        negative_labels = torch.clamp(1.0 - labels, min=0.0, max=1.0)

        return self.forward(
            negative_labels,
            is_training=is_training,
            unconditional=False,
            noise_std=noise_std,
        )

    def log_embedding_stats(self) -> dict:
        """Return embedding statistics for monitoring during training."""
        with torch.no_grad():
            table = self._compute_class_table()  # (K, D)
            return {
                "embed/mean": table.mean().item(),
                "embed/std": table.std().item(),
                "embed/min": table.min().item(),
                "embed/max": table.max().item(),
                "embed/norm": table.norm(dim=-1).mean().item(),
                "embed/base_norm": self.base.norm().item(),
                "embed/delta_mean": self.deltas.mean().item(),
                "embed/delta_std": self.deltas.std().item(),
            }
