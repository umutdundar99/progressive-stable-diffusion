"""
Comprehensive Evaluation Pipeline for IP-Adapter MES Progression Models.

For each source image with Mayo score X, generates images for the 3 other
MES classes (0-3 excl. X) so that every source image yields exactly
3 generated images.  Metrics are computed **per target class** and
**overall** (pooled).

Supports sweeping over multiple checkpoints x scale values.
Each combination creates a separate W&B run.

Scale interpretation per model type
------------------------------------
- ``use_routing_gates=True``  -> scale = steer_scale (delta pathway),
  guidance_scale forced to 1.0
- ``use_routing_gates=False`` -> scale = guidance_scale (CFG),
  steer_scale forced to 0.0.  Unconditional AOE obtained via
  ``get_negative_embedding`` (not zeros).

Metrics
-------
- FID   (Frechet Inception Distance)     -- distribution quality
- CMMD  (CLIP Maximum Mean Discrepancy)  -- distribution quality via CLIP
- IS    (Inception Score)                -- quality + diversity
- LPIPS (perceptual distance)            -- per-class gen->real similarity
- SSIM  (structural similarity)          -- per-class gen->real similarity

Usage
-----
    PYTHONPATH=. python -m src.pipelines.evaluation.evaluation_pipeline \
        --checkpoints  ckpt_noRG_noFP.ckpt  ckpt_RG_noFP.ckpt \
                       ckpt_noRG_FP.ckpt    ckpt_RG_FP.ckpt   \
        --configs      cfg_noRG.yaml cfg_RG.yaml cfg_noRG.yaml cfg_RG.yaml \
        --checkpoint-names noRG_noFP RG_noFP noRG_FP RG_FP \
        --scales 1.0 2.0 3.0 \
        --data-root data/limuc_cleaned/test data/limuc_cleaned/val \
        --batch-images 4
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel

from src.models.diffusion_module_ip import DiffusionModuleWithIP

# -- Metric library imports ---------------------------------------------------
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    HAS_TORCHMETRICS = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_TORCHMETRICS = False
    print(f"Warning: torchmetrics/torch-fidelity not installed: {e}")

try:
    import lpips as lpips_lib

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed.")

try:
    from skimage.metrics import structural_similarity as skimage_ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not installed.")

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed.")

ALL_MES_CLASSES = [0, 1, 2, 3]
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# == Data types ===============================================================


@dataclass
class GenerationJob:
    """One source image -> one target MES class."""

    source_path: Path
    source_label: int
    target_label: int


@dataclass
class EvalResult:
    """Holds per-class and overall metric values."""

    fid_per_class: Dict[int, float] = field(default_factory=dict)
    cmmd_per_class: Dict[int, float] = field(default_factory=dict)
    is_mean: float = -1.0
    is_std: float = -1.0
    lpips_per_class: Dict[int, float] = field(default_factory=dict)
    ssim_per_class: Dict[int, float] = field(default_factory=dict)
    overall_fid: float = -1.0
    overall_cmmd: float = -1.0
    overall_lpips: float = -1.0
    overall_ssim: float = -1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fid_per_class": {str(k): v for k, v in self.fid_per_class.items()},
            "cmmd_per_class": {str(k): v for k, v in self.cmmd_per_class.items()},
            "is_mean": self.is_mean,
            "is_std": self.is_std,
            "lpips_per_class": {str(k): v for k, v in self.lpips_per_class.items()},
            "ssim_per_class": {str(k): v for k, v in self.ssim_per_class.items()},
            "overall_fid": self.overall_fid,
            "overall_cmmd": self.overall_cmmd,
            "overall_lpips": self.overall_lpips,
            "overall_ssim": self.overall_ssim,
        }

    def flat_dict(self, prefix: str = "") -> Dict[str, float]:
        """Return a flat dict suitable for wandb.log."""
        d: Dict[str, float] = {}
        for cls, v in self.fid_per_class.items():
            d[f"{prefix}fid/class_{cls}"] = v
        for cls, v in self.cmmd_per_class.items():
            d[f"{prefix}cmmd/class_{cls}"] = v
        for cls, v in self.lpips_per_class.items():
            d[f"{prefix}lpips/class_{cls}"] = v
        for cls, v in self.ssim_per_class.items():
            d[f"{prefix}ssim/class_{cls}"] = v
        d[f"{prefix}fid/overall"] = self.overall_fid
        d[f"{prefix}cmmd/overall"] = self.overall_cmmd
        d[f"{prefix}lpips/overall"] = self.overall_lpips
        d[f"{prefix}ssim/overall"] = self.overall_ssim
        d[f"{prefix}is/mean"] = self.is_mean
        d[f"{prefix}is/std"] = self.is_std
        return d


# == CLI ======================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate IP-Adapter MES progression model(s)."
    )
    p.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="One or more checkpoint paths.",
    )
    p.add_argument(
        "--checkpoint-names",
        type=str,
        nargs="+",
        default=None,
        help="Short names for checkpoints (for W&B run id). "
        "Defaults to parent directory names.",
    )
    p.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Scale values to sweep. For routing-gates models this is "
        "steer_scale; for baseline models this is guidance_scale (CFG).",
    )
    p.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        default=[Path("configs/train_ip.yaml")],
        help="Config file(s). One per checkpoint, or a single config "
        "shared by all checkpoints.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        nargs="+",
        default=[
            Path("data/limuc_cleaned/test"),
            Path("data/limuc_cleaned/val"),
        ],
        help="One or more dataset roots with sub-folders 0/ 1/ 2/ 3/. "
        "Multiple paths are combined (e.g. test + val).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
    )
    p.add_argument(
        "--sampling-steps",
        type=int,
        default=50,
    )
    p.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--eta",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--no-blur",
        action="store_true",
        default=True,
        help="Skip blurring the structure image.",
    )
    p.add_argument(
        "--batch-images",
        type=int,
        default=4,
        help="Source images batched together.  Effective UNet batch = "
        "batch_images x 3 (3 target classes per source).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="ip-adapter-evaluation",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    p.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images to disk.",
    )
    p.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help="Max source images per class (0 = use all).",
    )
    p.add_argument(
        "--num-lpips-pairs",
        type=int,
        default=200,
        help="Number of random pairs for LPIPS / SSIM.",
    )
    return p.parse_args()


# == Helpers ==================================================================


def _resolve_device(s: str) -> torch.device:
    if s != "auto":
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_config(p: Path) -> DictConfig:
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return OmegaConf.load(p)


def _convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# == Image I/O (reusable CLIP processor, cached) =============================

_CLIP_PROCESSOR: Optional[CLIPImageProcessor] = None
_DISPLAY_TRANSFORM: Optional[transforms.Compose] = None


def _init_image_pipeline(target_size: int) -> None:
    global _CLIP_PROCESSOR, _DISPLAY_TRANSFORM
    _CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    _DISPLAY_TRANSFORM = transforms.Compose(
        [transforms.Resize((target_size, target_size)), transforms.ToTensor()]
    )


def _apply_gaussian_blur(
    images: Tensor, kernel_size: int = 15, sigma: float = 5.0
) -> Tensor:
    dev, dt = images.device, images.dtype
    x = torch.arange(kernel_size, device=dev, dtype=dt) - kernel_size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    gh = g.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    gv = g.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    pad = kernel_size // 2
    b = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    b = F.conv2d(b, gh, groups=3)
    b = F.conv2d(b, gv, groups=3)
    return b.clamp(0, 1)


def _load_structure_image(
    image_path: Path,
    device: torch.device,
    apply_blur: bool = False,
    blur_kernel_size: int = 7,
    blur_sigma: float = 2.0,
) -> Tensor:
    """Load one image -> CLIP-ready (1, 3, 224, 224) on *device*."""
    pil = Image.open(image_path).convert("RGB")
    display = _DISPLAY_TRANSFORM(pil)
    if apply_blur:
        display = _apply_gaussian_blur(
            display.unsqueeze(0), kernel_size=blur_kernel_size, sigma=blur_sigma
        ).squeeze(0)
    pil_out = transforms.ToPILImage()(display)
    clip_inputs = _CLIP_PROCESSOR(images=pil_out, return_tensors="pt", do_rescale=True)
    return clip_inputs.pixel_values.to(device)


def _load_real_images_for_class(
    data_dirs: List[Path],
    class_idx: int,
    image_size: int,
    max_images: int = 0,
) -> Tensor:
    """Load real images for a specific class from one or more data roots.

    Returns (N, 3, H, W) in [0,1].
    """
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    images = []
    for data_dir in data_dirs:
        class_dir = data_dir / str(class_idx)
        if not class_dir.exists():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                if max_images > 0 and len(images) >= max_images:
                    break
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(transform(img))
                except Exception as e:
                    print(f"  Warning: could not load {p}: {e}")
    if not images:
        raise ValueError(f"No images found for class {class_idx} in {data_dirs}")
    return torch.stack(images)


# == Conditioning & Sampling (matches inference_pipeline_ip.py exactly) =======


def _prepare_conditioning(
    module: DiffusionModuleWithIP,
    target_labels: Tensor,
    source_labels: Tensor,
    structure_images: Tensor,
    image_scale: float = 1.0,
    zero_aoe: bool = False,
) -> Tensor:
    """
    Prepare conditioning embeddings for evaluation sampling.

    When ``use_routing_gates=True``:
        3-segment: [Target_AOE | E_clean | Delta_AOE]

    When ``use_routing_gates=False``:
        2-segment: [AOE | Image]

    ``structure_images`` is (B, 3, 224, 224) -- one per sample.

    Args:
        zero_aoe: If True, replace AOE with ``get_negative_embedding``
                  (unconditional pass for CFG).  For Mayo 2, the
                  unconditional embedding is Mayo 0's base embeddings.
    """
    use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)

    t_aoe = module.ordinal_embedder(target_labels, is_training=False)
    if t_aoe.dim() == 2:
        t_aoe = t_aoe.unsqueeze(1)

    # CFG unconditional pass: replace AOE with negative conditioning
    if zero_aoe:
        t_aoe = module.ordinal_embedder.get_negative_embedding(
            target_labels, is_training=False
        )
        if t_aoe.dim() == 2:
            t_aoe = t_aoe.unsqueeze(1)

    s_aoe = module.ordinal_embedder(source_labels, is_training=False)
    if s_aoe.dim() == 2:
        s_aoe = s_aoe.unsqueeze(1)

    img_emb = module._get_image_embeds(structure_images)
    if module.feature_purifier is not None:
        img_emb = module.feature_purifier(img_emb, s_aoe)
    if image_scale != 1.0:
        img_emb = img_emb * image_scale

    if use_routing_gates:
        # 3-segment: [Target_AOE | E_clean | Delta_AOE]
        delta = module.ordinal_embedder.get_ordinal_delta_embedding(
            source_labels, target_labels
        )
        if delta.dim() == 2:
            delta = delta.unsqueeze(1)
        return torch.cat([t_aoe, img_emb, delta], dim=1)
    else:
        # 2-segment: [AOE | Image]
        return torch.cat([t_aoe, img_emb], dim=1)


def _set_delta_scale(module: DiffusionModuleWithIP, scale: float) -> None:
    """Set delta_scale on routing-gates processors (no-op for baseline processors)."""
    for _, mod in module.unet.unet.named_modules():
        if hasattr(mod, "processor") and hasattr(mod.processor, "delta_scale"):
            mod.processor.delta_scale = scale


@torch.inference_mode()
def _ddim_sample_batched(
    module: DiffusionModuleWithIP,
    target_labels: Tensor,
    source_labels: Tensor,
    structure_images: Tensor,
    sampling_steps: int,
    device: torch.device,
    eta: float = 0.0,
    image_scale: float = 1.0,
    steer_scale: float = 0.0,
    guidance_scale: float = 1.0,
) -> Tensor:
    """
    DDIM sampling with mode-aware disease control.

    **Routing-gates** (``use_routing_gates=True``):
        Single conditional pass.  ``steer_scale`` controls delta pathway.
        ``guidance_scale`` is ignored.

    **Baseline** (``use_routing_gates=False``):
        CFG dual-pass when ``guidance_scale != 1.0``::

            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        Unconditional AOE uses ``get_negative_embedding`` (not zeros).
    """
    use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)
    do_cfg = (not use_routing_gates) and (guidance_scale != 1.0)

    B = target_labels.shape[0]
    H = module.cfg.dataset.image_size
    T = module.diff_cfg.num_train_timesteps
    C = module.cfg.model.latent_channels

    latents = torch.randn(B, C, H // 8, H // 8, device=device, dtype=torch.float32)
    ac = module.alphas_cumprod
    ts = torch.linspace(T - 1, 0, steps=sampling_steps, dtype=torch.long, device=device)

    embed_cond = _prepare_conditioning(
        module,
        target_labels,
        source_labels,
        structure_images,
        image_scale=image_scale,
    )

    embed_uncond = None
    if do_cfg:
        embed_uncond = _prepare_conditioning(
            module,
            target_labels,
            source_labels,
            structure_images,
            image_scale=image_scale,
            zero_aoe=True,
        )

    _set_delta_scale(module, steer_scale)

    for i, t_s in enumerate(ts):
        ti = int(t_s.item())
        t = torch.full((B,), ti, dtype=torch.long, device=device)

        if do_cfg:
            eps_cond = module(latents, t, embed_cond)
            eps_uncond = module(latents, t, embed_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = module(latents, t, embed_cond)

        ab = ac[ti].to(device=device, dtype=latents.dtype)
        sa = torch.sqrt(ab)
        so = torch.sqrt(1.0 - ab)

        x0 = ((latents - so * eps) / sa).clamp(-4.0, 4.0)

        if i == sampling_steps - 1:
            latents = x0
            continue

        tp = int(ts[i + 1].item())
        abp = ac[tp].to(device=device, dtype=latents.dtype)
        sap = torch.sqrt(abp)
        sop = torch.sqrt(1.0 - abp)

        if eta == 0.0:
            latents = sap * x0 + sop * eps
        else:
            sigma = eta * torch.sqrt((1 - abp) / (1 - ab) * (1 - ab / abp))
            noise = torch.randn_like(latents)
            latents = sap * x0 + torch.sqrt(1 - abp - sigma**2) * eps + sigma * noise

    return latents


@torch.inference_mode()
def _decode_latents(module: DiffusionModuleWithIP, latents: Tensor) -> Tensor:
    """Decode latents -> [0, 1] RGB (B, 3, H, W) on CPU."""
    scaled = latents / module.diff_cfg.latent_scale
    decoded = module.vae.decode(scaled)
    imgs = decoded.sample if hasattr(decoded, "sample") else decoded
    imgs = imgs.float().clamp(-1.0, 1.0)
    return ((imgs + 1.0) / 2.0).clamp(0.0, 1.0).cpu()


# == Metrics ==================================================================


def compute_fid(real: Tensor, fake: Tensor, device: torch.device) -> float:
    """FID between real and generated images (both [0, 1] float, NCHW)."""
    if not HAS_TORCHMETRICS:
        return -1.0
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    bs = 32
    for i in range(0, len(real), bs):
        fid_metric.update(real[i : i + bs].to(device), real=True)
    for i in range(0, len(fake), bs):
        fid_metric.update(fake[i : i + bs].to(device), real=False)
    return fid_metric.compute().item()


def compute_inception_score(
    images: Tensor, device: torch.device
) -> Tuple[float, float]:
    """Inception Score for generated images."""
    if not HAS_TORCHMETRICS:
        return -1.0, -1.0
    inception = InceptionScore(normalize=True).to(device)
    bs = 32
    for i in range(0, len(images), bs):
        inception.update(images[i : i + bs].to(device))
    mean, std = inception.compute()
    return mean.item(), std.item()


def compute_lpips(
    real: Tensor,
    fake: Tensor,
    device: torch.device,
    num_pairs: int = 200,
) -> float:
    """Average LPIPS between random (real, generated) pairs."""
    if not HAS_LPIPS:
        return -1.0
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    n_real, n_fake = len(real), len(fake)
    if n_real == 0 or n_fake == 0:
        return -1.0

    num_pairs = min(num_pairs, n_real, n_fake)
    real_idx = torch.randperm(n_real)[:num_pairs]
    fake_idx = torch.randperm(n_fake)[:num_pairs]

    dists = []
    with torch.no_grad():
        for a, b in zip(real_idx, fake_idx):
            r = real[a : a + 1].to(device) * 2 - 1  # [0,1] -> [-1,1]
            f = fake[b : b + 1].to(device) * 2 - 1
            dists.append(lpips_fn(r, f).item())
    return float(np.mean(dists))


def compute_ssim(
    real: Tensor,
    fake: Tensor,
    num_pairs: int = 200,
) -> float:
    """Average SSIM between random (real, generated) pairs."""
    if not HAS_SKIMAGE:
        return -1.0
    n_real, n_fake = len(real), len(fake)
    if n_real == 0 or n_fake == 0:
        return -1.0
    num_pairs = min(num_pairs, n_real, n_fake)

    vals = []
    for _ in range(num_pairs):
        ri = np.random.randint(0, n_real)
        fi = np.random.randint(0, n_fake)
        r_np = real[ri].permute(1, 2, 0).numpy()
        f_np = fake[fi].permute(1, 2, 0).numpy()
        vals.append(skimage_ssim(r_np, f_np, channel_axis=2, data_range=1.0))
    return float(np.mean(vals))


# -- CMMD (CLIP Maximum Mean Discrepancy) ------------------------------------


def _extract_clip_features(
    images: Tensor,
    clip_model: CLIPModel,
    clip_processor: CLIPImageProcessor,
    device: torch.device,
    batch_size: int = 32,
) -> Tensor:
    """Extract CLIP image features.  Returns (N, D) normalised vectors."""
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        # Convert tensors to PIL for CLIP processing
        pil_images = []
        for img in batch:
            arr = (img.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
            pil_images.append(Image.fromarray(arr))

        inputs = clip_processor(images=pil_images, return_tensors="pt", do_rescale=True)
        pixel_values = inputs.pixel_values.to(device)

        with torch.no_grad():
            vision_out = clip_model.vision_model(pixel_values=pixel_values)
            feats = clip_model.visual_projection(vision_out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def _mmd_rbf(x: Tensor, y: Tensor, sigmas: Optional[List[float]] = None) -> float:
    """
    Compute MMD-squared with sum of RBF kernels (unbiased estimator).

    Args:
        x: (N, D) features for distribution P (real)
        y: (M, D) features for distribution Q (generated)
        sigmas: kernel bandwidths.

    Returns:
        MMD-squared estimate (float).
    """
    if sigmas is None:
        sigmas = [0.1, 1.0, 10.0, 100.0]

    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return -1.0

    # Squared pairwise distances
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    mmd2 = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma**2)
        kxx = torch.exp(-gamma * xx)
        kyy = torch.exp(-gamma * yy)
        kxy = torch.exp(-gamma * xy)

        # Unbiased: exclude diagonal for k(x,x) and k(y,y)
        kxx_sum = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
        kyy_sum = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
        kxy_sum = kxy.sum() / (n * m)

        mmd2 += float(kxx_sum + kyy_sum - 2 * kxy_sum)

    return mmd2


def compute_cmmd(
    real: Tensor,
    fake: Tensor,
    device: torch.device,
    max_samples: int = 1000,
) -> float:
    """
    Compute CMMD between real and generated image sets.

    Uses CLIP ViT-L/14 features + RBF kernel MMD.
    """
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Sub-sample if too large
    if len(real) > max_samples:
        idx = torch.randperm(len(real))[:max_samples]
        real = real[idx]
    if len(fake) > max_samples:
        idx = torch.randperm(len(fake))[:max_samples]
        fake = fake[idx]

    real_feats = _extract_clip_features(real, clip_model, clip_proc, device)
    fake_feats = _extract_clip_features(fake, clip_model, clip_proc, device)

    # Free GPU memory
    del clip_model
    torch.cuda.empty_cache()

    return _mmd_rbf(real_feats, fake_feats)


# == Generation engine -- balanced, batched ===================================


def _collect_jobs(
    data_roots: List[Path], max_per_class: int = 0
) -> List[GenerationJob]:
    """
    Collect generation jobs from one or more data roots so that every
    source image produces images for the 3 *other* MES classes.
    """
    jobs: List[GenerationJob] = []
    for data_root in data_roots:
        for cls in ALL_MES_CLASSES:
            cls_dir = data_root / str(cls)
            if not cls_dir.is_dir():
                continue
            paths = sorted(
                p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if max_per_class > 0:
                paths = paths[:max_per_class]
            for p in paths:
                for target in ALL_MES_CLASSES:
                    if target != cls:
                        jobs.append(GenerationJob(p, cls, target))
    return jobs


@torch.inference_mode()
def generate_all(
    module: DiffusionModuleWithIP,
    jobs: List[GenerationJob],
    cfg: DictConfig,
    device: torch.device,
    batch_images: int = 4,
    sampling_steps: int = 50,
    image_scale: float = 1.0,
    steer_scale: float = 0.0,
    guidance_scale: float = 1.0,
    eta: float = 0.0,
    apply_blur: bool = False,
    seed: int = 42,
) -> Dict[int, Tensor]:
    """
    Generate all jobs, returning {target_class: (N, 3, H, W)} tensors.

    Batches multiple jobs together for efficient GPU utilisation.
    Model is NOT reloaded -- it must be on *device* and in eval mode.

    For baseline models (``use_routing_gates=False``) ``guidance_scale``
    controls CFG strength.  For routing-gates models ``steer_scale``
    controls the delta pathway.
    """
    blur_cfg = cfg.model
    bk = getattr(blur_cfg, "blur_kernel_size", 7)
    bs_sigma = getattr(blur_cfg, "blur_sigma", 2.0)

    # Sort jobs by source path so same-source jobs are adjacent
    jobs_sorted = sorted(jobs, key=lambda j: (str(j.source_path), j.target_label))

    # Bucket by source path
    source_buckets: OrderedDict[Path, List[GenerationJob]] = OrderedDict()
    for j in jobs_sorted:
        source_buckets.setdefault(j.source_path, []).append(j)

    source_list = list(source_buckets.items())
    total_jobs = len(jobs)

    # Pre-allocate result lists per target class
    result_lists: Dict[int, List[Tensor]] = {c: [] for c in ALL_MES_CLASSES}

    pbar = tqdm(total=total_jobs, desc="Generating", unit="img")
    src_idx = 0

    while src_idx < len(source_list):
        batch_sources = source_list[src_idx : src_idx + batch_images]
        src_idx += len(batch_sources)

        _set_seed(seed)

        # Flatten into one big batch
        all_targets: List[int] = []
        all_sources: List[float] = []
        all_structs: List[Tensor] = []
        job_target_labels: List[int] = []

        for src_path, src_jobs in batch_sources:
            struct = _load_structure_image(
                src_path,
                device,
                apply_blur=apply_blur,
                blur_kernel_size=bk,
                blur_sigma=bs_sigma,
            )  # (1, 3, 224, 224)
            for j in src_jobs:
                all_targets.append(j.target_label)
                all_sources.append(float(j.source_label))
                all_structs.append(struct)
                job_target_labels.append(j.target_label)

        B = len(all_targets)
        target_labels = torch.tensor(all_targets, dtype=torch.float32, device=device)
        source_labels = torch.tensor(all_sources, dtype=torch.float32, device=device)
        structure_images = torch.cat(all_structs, dim=0)  # (B, 3, 224, 224)

        latents = _ddim_sample_batched(
            module,
            target_labels,
            source_labels,
            structure_images,
            sampling_steps=sampling_steps,
            device=device,
            eta=eta,
            image_scale=image_scale,
            steer_scale=steer_scale,
            guidance_scale=guidance_scale,
        )
        images = _decode_latents(module, latents)  # (B, 3, H, W) on CPU

        for k in range(B):
            result_lists[job_target_labels[k]].append(images[k : k + 1])

        pbar.update(B)

    pbar.close()

    # Concatenate per target class
    results: Dict[int, Tensor] = {}
    for cls in ALL_MES_CLASSES:
        if result_lists[cls]:
            results[cls] = torch.cat(result_lists[cls], dim=0)
        else:
            results[cls] = torch.zeros(
                0, 3, cfg.dataset.image_size, cfg.dataset.image_size
            )

    return results


# == Main evaluation loop =====================================================


def evaluate_one_run(
    generated: Dict[int, Tensor],
    real: Dict[int, Tensor],
    device: torch.device,
    num_lpips_pairs: int = 200,
) -> EvalResult:
    """Compute all metrics for one steer-scale run."""
    result = EvalResult()

    # -- Per-class metrics ----------------------------------------------------
    all_gen = []
    all_real = []

    for cls in ALL_MES_CLASSES:
        gen_cls = generated.get(cls)
        real_cls = real.get(cls)
        if gen_cls is None or len(gen_cls) == 0:
            print(f"  Class {cls}: no generated images, skipping.")
            continue
        if real_cls is None or len(real_cls) == 0:
            print(f"  Class {cls}: no real images, skipping.")
            continue

        print(f"  Class {cls}: {len(gen_cls)} generated vs {len(real_cls)} real")

        # FID
        print("    Computing FID ...")
        result.fid_per_class[cls] = compute_fid(real_cls, gen_cls, device)
        print(f"    FID = {result.fid_per_class[cls]:.2f}")

        # CMMD
        print("    Computing CMMD ...")
        result.cmmd_per_class[cls] = compute_cmmd(real_cls, gen_cls, device)
        print(f"    CMMD = {result.cmmd_per_class[cls]:.6f}")

        # LPIPS
        print("    Computing LPIPS ...")
        result.lpips_per_class[cls] = compute_lpips(
            real_cls, gen_cls, device, num_pairs=num_lpips_pairs
        )
        print(f"    LPIPS = {result.lpips_per_class[cls]:.4f}")

        # SSIM
        print("    Computing SSIM ...")
        result.ssim_per_class[cls] = compute_ssim(
            real_cls, gen_cls, num_pairs=num_lpips_pairs
        )
        print(f"    SSIM = {result.ssim_per_class[cls]:.4f}")

        all_gen.append(gen_cls)
        all_real.append(real_cls)

    # -- Overall metrics ------------------------------------------------------
    if all_gen and all_real:
        gen_all = torch.cat(all_gen, dim=0)
        real_all = torch.cat(all_real, dim=0)

        print(f"  Overall: {len(gen_all)} generated vs {len(real_all)} real")

        print("  Computing overall FID ...")
        result.overall_fid = compute_fid(real_all, gen_all, device)
        print(f"    Overall FID = {result.overall_fid:.2f}")

        print("  Computing overall CMMD ...")
        result.overall_cmmd = compute_cmmd(real_all, gen_all, device)
        print(f"    Overall CMMD = {result.overall_cmmd:.6f}")

        print("  Computing overall LPIPS ...")
        result.overall_lpips = compute_lpips(
            real_all, gen_all, device, num_pairs=num_lpips_pairs * 2
        )
        print(f"    Overall LPIPS = {result.overall_lpips:.4f}")

        print("  Computing overall SSIM ...")
        result.overall_ssim = compute_ssim(
            real_all, gen_all, num_pairs=num_lpips_pairs * 2
        )
        print(f"    Overall SSIM = {result.overall_ssim:.4f}")

        # IS (on generated only)
        print("  Computing Inception Score ...")
        result.is_mean, result.is_std = compute_inception_score(gen_all, device)
        print(f"    IS = {result.is_mean:.2f} +/- {result.is_std:.2f}")

    return result


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)

    n_ckpts = len(args.checkpoints)

    # -- Resolve per-checkpoint configs ----------------------------------------
    if len(args.configs) == 1:
        configs = [_load_config(args.configs[0])] * n_ckpts
    elif len(args.configs) == n_ckpts:
        configs = [_load_config(p) for p in args.configs]
    else:
        raise ValueError(
            f"--configs must be 1 (shared) or {n_ckpts} (per checkpoint), "
            f"got {len(args.configs)}"
        )

    image_size = configs[0].dataset.image_size

    # Resolve checkpoint names
    ckpt_names = args.checkpoint_names
    if ckpt_names is None:
        ckpt_names = [p.parent.parent.name for p in args.checkpoints]
    if len(ckpt_names) != n_ckpts:
        raise ValueError(
            f"Number of checkpoint names ({len(ckpt_names)}) must match "
            f"number of checkpoints ({n_ckpts})"
        )

    # Init image pipeline
    _init_image_pipeline(image_size)

    # -- Collect generation jobs (test + val combined) -------------------------
    data_roots = args.data_root  # List[Path] now
    print(f"\nCollecting images from {[str(d) for d in data_roots]} ...")
    jobs = _collect_jobs(data_roots, max_per_class=args.max_images_per_class)
    n_sources = len(set(j.source_path for j in jobs))
    print(
        f"  {n_sources} source images -> {len(jobs)} generation jobs " f"(3 per source)"
    )

    # -- Load real images (done once, combined from all roots) -----------------
    print("\nLoading real images ...")
    real_images: Dict[int, Tensor] = {}
    for cls in ALL_MES_CLASSES:
        try:
            real_images[cls] = _load_real_images_for_class(data_roots, cls, image_size)
            print(f"  Class {cls}: {len(real_images[cls])} real images")
        except (FileNotFoundError, ValueError) as e:
            print(f"  Class {cls}: {e}")

    # -- Main loop: checkpoint x scale -----------------------------------------
    all_results: Dict[str, EvalResult] = {}

    for ckpt_idx, (ckpt_path, ckpt_name) in enumerate(
        zip(args.checkpoints, ckpt_names)
    ):
        ckpt_cfg = configs[ckpt_idx]

        print(f"\n{'=' * 60}")
        print(f"CHECKPOINT: {ckpt_name}  ({ckpt_path})")
        print(f"{'=' * 60}")

        # Load model ONCE per checkpoint
        print("Loading model ...")
        module = DiffusionModuleWithIP.load_from_checkpoint(
            str(ckpt_path),
            cfg=ckpt_cfg,
            weights_only=False,
        )
        module = module.to(device).to(torch.float32)
        module.eval()

        use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)
        use_feature_purifier = module.feature_purifier is not None
        mode_str = (
            f"routing_gates={use_routing_gates}, "
            f"feature_purifier={use_feature_purifier}"
        )
        print(f"  Mode: {mode_str}")

        for scale_val in args.scales:
            # Interpret scale based on model mode
            if use_routing_gates:
                steer_scale = scale_val
                guidance_scale = 1.0
                scale_label = f"steer{scale_val:.2f}"
            else:
                steer_scale = 0.0
                guidance_scale = scale_val
                scale_label = f"cfg{scale_val:.2f}"

            run_name = f"{ckpt_name}_{scale_label}"
            print(
                f"\n--- Scale: {scale_val}  "
                f"(steer={steer_scale}, cfg={guidance_scale})  "
                f"run: {run_name} ---"
            )

            # Init W&B run
            wb_run = None
            if HAS_WANDB and not args.no_wandb:
                wb_run = wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config={
                        "checkpoint": str(ckpt_path),
                        "checkpoint_name": ckpt_name,
                        "use_routing_gates": use_routing_gates,
                        "use_feature_purifier": use_feature_purifier,
                        "scale": scale_val,
                        "steer_scale": steer_scale,
                        "guidance_scale": guidance_scale,
                        "image_scale": args.image_scale,
                        "sampling_steps": args.sampling_steps,
                        "eta": args.eta,
                        "seed": args.seed,
                        "batch_images": args.batch_images,
                        "num_source_images": n_sources,
                        "num_jobs": len(jobs),
                        "blur": not args.no_blur,
                        "data_roots": [str(d) for d in data_roots],
                    },
                    reinit=True,
                )

            t0 = time.time()

            # Generate
            generated = generate_all(
                module=module,
                jobs=jobs,
                cfg=ckpt_cfg,
                device=device,
                batch_images=args.batch_images,
                sampling_steps=args.sampling_steps,
                image_scale=args.image_scale,
                steer_scale=steer_scale,
                guidance_scale=guidance_scale,
                eta=args.eta,
                apply_blur=not args.no_blur,
                seed=args.seed,
            )
            gen_time = time.time() - t0
            total_gen = sum(len(v) for v in generated.values())
            print(
                f"  Generated {total_gen} images in {gen_time:.1f}s "
                f"({total_gen / max(gen_time, 0.001):.1f} img/s)"
            )

            # Save images if requested
            if args.save_images:
                save_dir = args.output_dir / run_name
                for cls, imgs in generated.items():
                    cls_dir = save_dir / str(cls)
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    for idx, img in enumerate(imgs):
                        arr = (img.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
                        Image.fromarray(arr).save(cls_dir / f"gen_{idx:04d}.png")

            # Compute metrics
            print("\nComputing metrics ...")
            result = evaluate_one_run(
                generated,
                real_images,
                device,
                num_lpips_pairs=args.num_lpips_pairs,
            )
            elapsed = time.time() - t0

            # Log to W&B
            if wb_run is not None:
                log_dict = result.flat_dict()
                log_dict["generation_time_s"] = gen_time
                log_dict["total_time_s"] = elapsed
                log_dict["num_generated"] = total_gen
                wandb.log(log_dict)
                wandb.finish()

            # Save JSON
            run_dir = args.output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            result_path = run_dir / "metrics.json"
            with open(result_path, "w") as f:
                json.dump(
                    _convert_to_serializable(
                        {
                            "checkpoint": str(ckpt_path),
                            "checkpoint_name": ckpt_name,
                            "use_routing_gates": use_routing_gates,
                            "use_feature_purifier": use_feature_purifier,
                            "scale": scale_val,
                            "steer_scale": steer_scale,
                            "guidance_scale": guidance_scale,
                            "metrics": result.to_dict(),
                            "elapsed_s": elapsed,
                        }
                    ),
                    f,
                    indent=2,
                )
            print(f"  Saved metrics to {result_path}")

            all_results[run_name] = result

            # Print summary
            print(f"\n  SUMMARY ({run_name}):")
            print(f"    Overall FID  = {result.overall_fid:.2f}")
            print(f"    Overall CMMD = {result.overall_cmmd:.6f}")
            print(f"    Overall LPIPS= {result.overall_lpips:.4f}")
            print(f"    Overall SSIM = {result.overall_ssim:.4f}")
            print(f"    IS           = {result.is_mean:.2f} +/- {result.is_std:.2f}")
            for cls in ALL_MES_CLASSES:
                if cls in result.fid_per_class:
                    print(
                        f"    Class {cls}: FID={result.fid_per_class[cls]:.2f}  "
                        f"CMMD={result.cmmd_per_class.get(cls, -1):.6f}  "
                        f"LPIPS={result.lpips_per_class.get(cls, -1):.4f}  "
                        f"SSIM={result.ssim_per_class.get(cls, -1):.4f}"
                    )
            print(f"    Time: {elapsed:.1f}s")

        # Unload model to free VRAM before next checkpoint
        del module
        torch.cuda.empty_cache()

    # -- Final comparison table ------------------------------------------------
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print(f"{'=' * 70}")
    header = (
        f"{'Run':<40} {'FID':>8} {'CMMD':>10} {'IS':>8} " f"{'LPIPS':>8} {'SSIM':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        print(
            f"{name:<40} "
            f"{r.overall_fid:>8.2f} "
            f"{r.overall_cmmd:>10.6f} "
            f"{r.is_mean:>8.2f} "
            f"{r.overall_lpips:>8.4f} "
            f"{r.overall_ssim:>8.4f}"
        )
    print(f"{'=' * 70}")

    # Save comparison JSON
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(
            _convert_to_serializable(
                {name: r.to_dict() for name, r in all_results.items()}
            ),
            f,
            indent=2,
        )
    print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
