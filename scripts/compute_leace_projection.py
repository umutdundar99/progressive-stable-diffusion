"""
Compute LEACE (LEAst-Concept Erasure) projection matrix.

This script extracts image embeddings from the training set, fits a linear
model that predicts Mayo score from image embeddings, then computes the
null-space projection matrix that *erases* disease information while
preserving anatomy.

Usage:
    python -m scripts.compute_leace_projection \
        --checkpoint <path_to_ckpt> \
        --config configs/train_ip.yaml \
        --data-root data/limuc/processed_data_scale2/train \
        --output leace_projection.pt

The saved .pt file contains:
    P_null  : (D, D) projection matrix  (D = num_tokens * cross_attention_dim)
    mu      : (D,)   embedding mean used for centering
    mayo_dir: (D, R) top-R singular vectors of the disease subspace
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import CLIPImageProcessor

from src.models.diffusion_module_ip import DiffusionModuleWithIP


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute LEACE projection matrix.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/train_ip.yaml"))
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to training data root (e.g. data/limuc/.../train)",
    )
    p.add_argument("--output", type=Path, default=Path("leace_projection.pt"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Number of disease-direction singular vectors to erase. "
        "1 = erase only the strongest disease direction. "
        "Higher = more aggressive erasure.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def _resolve_device(s: str) -> torch.device:
    if s != "auto":
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_image_paths(data_root: Path) -> list[tuple[Path, int]]:
    """Collect (image_path, mayo_score) pairs from directory structure."""
    samples: list[tuple[Path, int]] = []
    for cls_name in sorted(os.listdir(data_root)):
        cls_dir = data_root / cls_name
        if not cls_dir.is_dir():
            continue
        mayo = int(cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                samples.append((cls_dir / fname, mayo))
    labels_set = sorted(set(m for _, m in samples))
    print(f"Found {len(samples)} images across classes {labels_set}")
    return samples


@torch.no_grad()
def _extract_embeddings(
    module: DiffusionModuleWithIP,
    samples: list[tuple[Path, int]],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """
    Extract image projection embeddings for all samples.

    Returns:
        embeddings: (N, num_tokens, D)  -- image tokens per sample
        labels:     (N,)                -- Mayo scores
    """
    clip_processor = CLIPImageProcessor.from_pretrained(
        module.diff_cfg.image_encoder_path
    )

    all_embeds: list[Tensor] = []
    all_labels: list[Tensor] = []

    for start in tqdm(range(0, len(samples), batch_size), desc="Extracting embeddings"):
        batch_samples = samples[start : start + batch_size]

        clip_tensors = []
        mayo_scores = []
        for img_path, mayo in batch_samples:
            pil_img = Image.open(img_path).convert("RGB")
            clip_input = clip_processor(
                images=pil_img, return_tensors="pt", do_rescale=True
            )
            clip_tensors.append(clip_input.pixel_values.squeeze(0))
            mayo_scores.append(mayo)

        clip_batch = torch.stack(clip_tensors).to(device)
        # Get projected image embeddings (B, num_tokens, D)
        img_embeds = module._get_image_embeds(clip_batch)  # (B, 16, 768)
        all_embeds.append(img_embeds.cpu())
        all_labels.append(torch.tensor(mayo_scores, dtype=torch.float32))

    return torch.cat(all_embeds, dim=0), torch.cat(all_labels, dim=0)


def compute_leace_projection(
    embeddings: Tensor,
    labels: Tensor,
    rank: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute the LEACE null-space projection matrix.

    LEACE finds the linear subspace most predictive of the concept
    (Mayo score) and projects embeddings onto its orthogonal complement.

    We work on flattened tokens: (N, num_tokens * D) so the projection
    captures cross-token disease patterns too.

    Algorithm:
    1. Center embeddings: X_c = X - mu
    2. Compute class-conditional means for each Mayo score
    3. Form the between-class scatter matrix via class means
    4. SVD → top `rank` directions = disease subspace
    5. P_null = I - V_r @ V_r^T  (project out disease directions)

    After projection: a linear classifier can NO LONGER predict Mayo
    from the image embeddings.

    Args:
        embeddings: (N, T, D) image embeddings per sample (T tokens, D dims)
        labels:     (N,) Mayo scores (0, 1, 2, 3)
        rank:       Number of directions to erase

    Returns:
        P_null:   (T*D, T*D) projection matrix
        mu:       (T*D,)     mean used for centering
        mayo_dir: (T*D, rank) erased directions
    """
    # Flatten tokens for joint analysis
    N, T, D = embeddings.shape
    X = embeddings.view(N, T * D).float()  # (N, T*D)

    # 1. Center
    mu = X.mean(dim=0)  # (T*D,)
    X_c = X - mu

    # 2. Class-conditional means (between-class scatter)
    unique_labels = labels.unique().sort().values
    class_means = []
    class_counts = []
    for lbl in unique_labels:
        mask = labels == lbl
        class_means.append(X_c[mask].mean(dim=0))
        class_counts.append(mask.sum().item())

    print(f"\nClass counts: {dict(zip(unique_labels.tolist(), class_counts))}")

    # 3. Weighted mean matrix: each row = sqrt(n_k) * mean_k
    M = torch.stack(class_means, dim=0)  # (K, T*D)
    W = torch.sqrt(torch.tensor(class_counts, dtype=torch.float32)).unsqueeze(
        1
    )  # (K, 1)
    M_weighted = M * W  # (K, T*D)

    # 4. SVD to find top disease directions
    U, S, Vh = torch.linalg.svd(M_weighted, full_matrices=False)
    # Vh: (K, T*D) — rows are right singular vectors
    mayo_dir = Vh[:rank].T  # (T*D, rank)

    print(f"Disease direction singular values: {S[:min(5, len(S))].tolist()}")
    print(f"Erasing top {rank} direction(s)")
    explained = (S[:rank] ** 2).sum() / (S**2).sum()
    print(f"Explained variance ratio: {explained:.4f}")

    # 5. Null-space projection: P = I - V_r @ V_r^T
    TD = T * D
    P_null = torch.eye(TD) - mayo_dir @ mayo_dir.T

    # Verification: check that class means collapse after projection
    projected_means = M @ P_null.T
    dist_after = torch.cdist(projected_means, projected_means).max().item()
    dist_before = torch.cdist(M, M).max().item()
    print(f"Max inter-class mean distance BEFORE: {dist_before:.4f}")
    print(f"Max inter-class mean distance AFTER:  {dist_after:.4f}")

    return P_null, mu, mayo_dir


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")

    # Load model
    cfg = OmegaConf.load(args.config)
    module = DiffusionModuleWithIP.load_from_checkpoint(
        str(args.checkpoint), cfg=cfg, weights_only=False
    )
    module = module.to(device).to(torch.float32)
    module.eval()

    # Collect samples
    samples = _collect_image_paths(args.data_root)

    # Extract embeddings
    embeddings, labels = _extract_embeddings(module, samples, device, args.batch_size)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Labels distribution: {torch.bincount(labels.long())}")

    # Compute LEACE projection
    P_null, mu, mayo_dir = compute_leace_projection(embeddings, labels, rank=args.rank)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "P_null": P_null,  # (T*D, T*D)
            "mu": mu,  # (T*D,)
            "mayo_dir": mayo_dir,  # (T*D, rank)
            "rank": args.rank,
            "num_samples": len(samples),
            "num_tokens": embeddings.shape[1],
            "token_dim": embeddings.shape[2],
        },
        args.output,
    )
    print(f"\nSaved LEACE projection to {args.output}")
    print(f"  P_null shape: {P_null.shape}")
    print(
        f"  Embedding layout: {embeddings.shape[1]} tokens x {embeddings.shape[2]} dim"
    )


if __name__ == "__main__":
    main()
