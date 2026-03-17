"""
Ordinal embedding strategies (BOE & AOE) used for progressive disease conditioning.

These embedders take a real-valued ordinal label y ∈ [0, K−1]
and return a D-dimensional embedding vector via interpolation.
"""

from __future__ import annotations

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

    if alpha.dim() == lower_idx.dim():
        alpha = alpha.unsqueeze(-1)

    emb_lo = F.embedding(lower_idx, table)
    emb_hi = F.embedding(upper_idx, table)

    return emb_lo * (1.0 - alpha) + emb_hi * alpha


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
        num_tokens: int = 16,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2 for ordinal modeling.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Base vector (represents MES 0 - healthy mucosa)
        self.base = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.base, mean=0.0, std=init_std)

        self.deltas = nn.Parameter(torch.empty(num_classes - 1, embedding_dim))
        self._initialize_monotonic_deltas(init_std, delta_scale)

        self.num_tokens = num_tokens

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(embedding_dim * num_tokens)

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
            for i in range(self.num_classes - 1):
                nn.init.normal_(self.deltas[i], mean=delta_scale, std=init_std)
                self.deltas[i] *= 1.0 + 0.1 * i

    def _compute_class_table(self) -> torch.Tensor:
        """
        Returns:
            Table of shape (K, D) with:
            E[0], E[1], ..., E[K-1]
        where E[k] = base + sum(deltas[:k])
        """
        cumulative = torch.cumsum(self.deltas, dim=0)
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
        is_training: bool = False,
        unconditional: bool = False,
        noise_std: float = 0.005,
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

        if is_training and noise_std > 0:
            noise = torch.randn_like(out) * noise_std
            out = out + noise

        out = self.projector(out)
        out = out.view(-1, self.num_tokens, self.embedding_dim)

        return out.squeeze(0) if scalar_input else out

    def get_negative_embedding(
        self,
        labels: torch.Tensor,
        is_training: bool = False,
        noise_std: float = 0.005,
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

        negative_labels = torch.clamp(1.0 - labels, min=0.0, max=1.0)

        return self.forward(
            negative_labels,
            is_training=is_training,
            unconditional=False,
            noise_std=noise_std,
        )

    def get_disease_delta_embedding(
        self,
        source_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the *pure disease component* for negative steering (Phase 3).

        Returns ``proj(E[source]) - proj(E[0])``, giving the disease-only
        signal in projected token space that can be subtracted at inference.

        The subtraction is done *after* projection so that projector bias
        terms cancel out, ensuring Mayo 0 inputs yield a true zero delta.

        Args:
            source_labels: (B,) integer or float labels in [0, K-1].

        Returns:
            (B, num_tokens, embedding_dim) — projected disease delta.
        """
        return self.get_ordinal_delta_embedding(
            source_labels=source_labels,
            target_labels=torch.zeros_like(source_labels),
        )

    def get_ordinal_delta_embedding(
        self,
        source_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute directional delta between two severity levels.

        Returns ``proj(E[target]) - proj(E[source])`` in projected token space.
        The subtraction is done *after* projection so projector bias terms cancel,
        guaranteeing exact-zero output when source == target.

        **Sign semantics:**

        * Progression (target > source): positive delta → add disease features
        * Regression  (target < source): negative delta → remove disease features
        * Same level  (target == source): zero delta → no steering

        Args:
            source_labels: (B,) Mayo scores of the input image.
            target_labels: (B,) desired Mayo scores for the output.

        Returns:
            (B, num_tokens, embedding_dim) — directional delta embeddings.
        """
        scalar_input = source_labels.dim() == 0
        if scalar_input:
            source_labels = source_labels.unsqueeze(0)
            target_labels = target_labels.unsqueeze(0)

        table = self._compute_class_table()  # (K, D)
        max_idx = self.num_classes - 1

        # Source embedding
        src = source_labels.to(table.dtype).clamp(0.0, float(max_idx))
        s_lo, s_hi = torch.floor(src), torch.clamp(torch.floor(src) + 1, max=max_idx)
        src_emb = _linear_interpolate(table, s_lo.long(), s_hi.long(), src - s_lo)

        # Target embedding
        tgt = target_labels.to(table.dtype).clamp(0.0, float(max_idx))
        t_lo, t_hi = torch.floor(tgt), torch.clamp(torch.floor(tgt) + 1, max=max_idx)
        tgt_emb = _linear_interpolate(table, t_lo.long(), t_hi.long(), tgt - t_lo)

        # Project both through the AOE projector, then subtract so biases cancel
        proj_src = self.projector(src_emb).view(-1, self.num_tokens, self.embedding_dim)
        proj_tgt = self.projector(tgt_emb).view(-1, self.num_tokens, self.embedding_dim)

        delta = proj_tgt - proj_src  # (B, T, D)

        return delta.squeeze(0) if scalar_input else delta

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
