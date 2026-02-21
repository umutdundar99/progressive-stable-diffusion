"""
Auxiliary losses for disentangling anatomy and disease embeddings.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def compute_orthogonal_loss(
    anatomy_embeds: Tensor,
    disease_embeds: Tensor,
) -> Tensor:
    """Compute orthogonal loss between anatomy and disease embeddings.

    Encourages the two embedding sets to live in orthogonal subspaces by
    penalising their cosine similarity.

    Args:
        anatomy_embeds: (B, S1, Dim) — IP-Adapter image tokens.
        disease_embeds: (B, S2, Dim) — AOE tokens (S2 may differ from S1).

    Returns:
        Scalar: squared cosine similarity of mean-pooled embeddings (→ 0 when orthogonal).
    """
    # Mean-pool across tokens → (B, D) — handles different token counts
    a = anatomy_embeds.mean(dim=1)
    d = disease_embeds.mean(dim=1)

    # Per-sample cosine similarity → (B,)
    cos_sim = F.cosine_similarity(a, d, dim=-1)
    return (cos_sim**2).mean()
