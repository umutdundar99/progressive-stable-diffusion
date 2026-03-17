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
- FID       (Frechet Inception Distance)          -- distribution quality
- CMMD      (CLIP Maximum Mean Discrepancy)       -- distribution quality via CLIP
- Precision (Improved P&R, Kynkäänniemi 2019)     -- manifold quality
- Recall    (Improved P&R, Kynkäänniemi 2019)     -- manifold coverage

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
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision import models as tv_models
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel

from src.models.diffusion_module_ip import DiffusionModuleWithIP

try:
    from torchmetrics.image.fid import FrechetInceptionDistance

    HAS_TORCHMETRICS = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_TORCHMETRICS = False
    print(f"Warning: torchmetrics/torch-fidelity not installed: {e}")

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed.")

ALL_MES_CLASSES = [0, 1, 2, 3]
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class GenerationJob:
    """One source image -> one target MES class."""

    source_path: Path
    source_label: int
    target_label: int


@dataclass
class EvalResult:
    """Holds per-class and overall metric values.

    Overall metrics are computed via class-balanced subsampling
    (min(real, gen) per class), repeated over multiple seeds.
    The ``*_std`` fields report standard deviation across seeds.
    """

    fid_per_class: Dict[int, float] = field(default_factory=dict)
    cmmd_per_class: Dict[int, float] = field(default_factory=dict)
    ipr_precision_per_class: Dict[int, float] = field(default_factory=dict)
    ipr_recall_per_class: Dict[int, float] = field(default_factory=dict)
    overall_fid: float = -1.0
    overall_fid_std: float = 0.0
    overall_cmmd: float = -1.0
    overall_cmmd_std: float = 0.0
    overall_ipr_precision: float = -1.0
    overall_ipr_precision_std: float = 0.0
    overall_ipr_recall: float = -1.0
    overall_ipr_recall_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fid_per_class": {str(k): v for k, v in self.fid_per_class.items()},
            "cmmd_per_class": {str(k): v for k, v in self.cmmd_per_class.items()},
            "ipr_precision_per_class": {
                str(k): v for k, v in self.ipr_precision_per_class.items()
            },
            "ipr_recall_per_class": {
                str(k): v for k, v in self.ipr_recall_per_class.items()
            },
            "overall_fid": self.overall_fid,
            "overall_fid_std": self.overall_fid_std,
            "overall_cmmd": self.overall_cmmd,
            "overall_cmmd_std": self.overall_cmmd_std,
            "overall_ipr_precision": self.overall_ipr_precision,
            "overall_ipr_precision_std": self.overall_ipr_precision_std,
            "overall_ipr_recall": self.overall_ipr_recall,
            "overall_ipr_recall_std": self.overall_ipr_recall_std,
        }

    def flat_dict(self, prefix: str = "") -> Dict[str, float]:
        """Return a flat dict suitable for wandb.log."""
        d: Dict[str, float] = {}
        for cls, v in self.fid_per_class.items():
            d[f"{prefix}fid/class_{cls}"] = v
        for cls, v in self.cmmd_per_class.items():
            d[f"{prefix}cmmd/class_{cls}"] = v
        for cls, v in self.ipr_precision_per_class.items():
            d[f"{prefix}ipr_precision/class_{cls}"] = v
        for cls, v in self.ipr_recall_per_class.items():
            d[f"{prefix}ipr_recall/class_{cls}"] = v
        d[f"{prefix}fid/overall"] = self.overall_fid
        d[f"{prefix}fid/overall_std"] = self.overall_fid_std
        d[f"{prefix}cmmd/overall"] = self.overall_cmmd
        d[f"{prefix}cmmd/overall_std"] = self.overall_cmmd_std
        d[f"{prefix}ipr_precision/overall"] = self.overall_ipr_precision
        d[f"{prefix}ipr_precision/overall_std"] = self.overall_ipr_precision_std
        d[f"{prefix}ipr_recall/overall"] = self.overall_ipr_recall
        d[f"{prefix}ipr_recall/overall_std"] = self.overall_ipr_recall_std
        return d


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
        default="ip-adapter-evaluation-final",
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
        "--fp16",
        action="store_true",
        help="Run diffusion model with mixed-precision (autocast) for "
        "~2x faster generation, matching training's 16-mixed. "
        "Metric backbones always use float32.",
    )
    p.add_argument(
        "--compile-unet",
        action="store_true",
        help="Apply torch.compile() to the UNet for faster generation "
        "(requires PyTorch 2.x; first run includes compilation overhead).",
    )
    p.add_argument(
        "--overall-seeds",
        type=int,
        default=5,
        help="Number of random seeds for class-balanced overall metric "
        "subsampling.  Reports mean +/- std.  (default: 5)",
    )
    return p.parse_args()


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
        3-segment: [Source_AOE | E_clean | Delta_AOE]

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
        delta = module.ordinal_embedder.get_ordinal_delta_embedding(
            source_labels, target_labels
        )
        if delta.dim() == 2:
            delta = delta.unsqueeze(1)
        return torch.cat([s_aoe, img_emb, delta], dim=1)
    else:
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


def compute_fid(
    real: Tensor,
    fake: Tensor,
    device: torch.device,
    fid_metric: Optional["FrechetInceptionDistance"] = None,
) -> float:
    """FID between real and generated images (both [0, 1] float, NCHW).

    If *fid_metric* is provided it will be reset and reused (avoids
    recreating the InceptionV3 backbone each call).
    """
    if not HAS_TORCHMETRICS:
        return -1.0
    if fid_metric is None:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    else:
        fid_metric.reset()
    bs = 32
    for i in range(0, len(real), bs):
        fid_metric.update(real[i : i + bs].to(device), real=True)
    for i in range(0, len(fake), bs):
        fid_metric.update(fake[i : i + bs].to(device), real=False)
    return fid_metric.compute().item()


def _extract_clip_features(
    images: Tensor,
    clip_model: CLIPModel,
    clip_processor: CLIPImageProcessor,
    device: torch.device,
    batch_size: int = 64,
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

    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    mmd2 = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma**2)
        kxx = torch.exp(-gamma * xx)
        kyy = torch.exp(-gamma * yy)
        kxy = torch.exp(-gamma * xy)

        kxx_sum = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
        kyy_sum = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
        kxy_sum = kxy.sum() / (n * m)

        mmd2 += float(kxx_sum + kyy_sum - 2 * kxy_sum)

    return mmd2


def compute_cmmd(
    real: Tensor,
    fake: Tensor,
    device: torch.device,
    clip_model: Optional[CLIPModel] = None,
    clip_proc: Optional[CLIPImageProcessor] = None,
) -> float:
    """
    Compute CMMD between real and generated image sets.

    Uses CLIP ViT-L/14 features + RBF kernel MMD.
    All samples are used (no sub-sampling) for deterministic results.

    If *clip_model* / *clip_proc* are provided they are reused (avoids
    loading the 1.7 GB checkpoint on every call).
    """
    own_model = clip_model is None
    if own_model:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(
            device
        )
        clip_model.eval()
    if clip_proc is None:
        clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    real_feats = _extract_clip_features(real, clip_model, clip_proc, device)
    fake_feats = _extract_clip_features(fake, clip_model, clip_proc, device)

    if own_model:
        del clip_model
        torch.cuda.empty_cache()

    return _mmd_rbf(real_feats, fake_feats)


def _build_vgg16_feature_extractor(device: torch.device) -> torch.nn.Module:
    """Build a VGG16 feature extractor (up to second-to-last FC layer).

    Returns features of dimension 4096, matching the original IPR paper.
    """
    vgg = tv_models.vgg16(weights="IMAGENET1K_V1").to(device)
    vgg.eval()

    vgg.classifier = torch.nn.Sequential(*list(vgg.classifier.children())[:-1])
    return vgg


@torch.inference_mode()
def _extract_vgg_features(
    images: Tensor,
    vgg: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> Tensor:
    """Extract VGG16 features for a set of images.

    Args:
        images: (N, 3, H, W) in [0, 1].

    Returns:
        (N, 4096) feature tensor on CPU.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    resize = transforms.Resize((224, 224), antialias=True)
    all_feats: List[Tensor] = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        batch = torch.stack([normalize(resize(img)) for img in batch])
        feats = vgg(batch)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def _compute_ipr_from_features(
    real_feats: Tensor, fake_feats: Tensor, k: int = 3
) -> Tuple[float, float]:
    """Compute Improved Precision & Recall (Kynkäänniemi et al., 2019).

    Algorithm
    ---------
    For each distribution (real / generated), define its *manifold* as the
    union of hyperspheres centred at every sample with radius equal to the
    distance to its k-th nearest neighbour **within the same set**.

    - **Precision** = fraction of generated samples that fall inside the
      real manifold.  (quality)
    - **Recall** = fraction of real samples that fall inside the generated
      manifold.  (coverage / diversity)

    Reference: arXiv 1904.06991

    Args:
        real_feats:  (N, D) feature matrix for real images.
        fake_feats:  (M, D) feature matrix for generated images.
        k: neighbourhood size (default 3, as in the paper).

    Returns:
        (precision, recall) tuple of floats in [0, 1].
    """
    n = real_feats.shape[0]
    m = fake_feats.shape[0]
    if n < k + 1 or m < k + 1:
        return -1.0, -1.0

    real_dists = torch.cdist(real_feats, real_feats, p=2)  # (N, N)
    real_knn_radii = real_dists.topk(k + 1, largest=False).values[:, -1]

    fake_dists = torch.cdist(fake_feats, fake_feats, p=2)  # (M, M)
    fake_knn_radii = fake_dists.topk(k + 1, largest=False).values[:, -1]

    cross_dist = torch.cdist(fake_feats, real_feats, p=2)  # (M, N)

    precision = float(
        (cross_dist <= real_knn_radii.unsqueeze(0)).any(dim=1).float().mean().item()
    )

    recall = float(
        (cross_dist.T <= fake_knn_radii.unsqueeze(0)).any(dim=1).float().mean().item()
    )

    return precision, recall


def compute_improved_precision_recall(
    real: Tensor,
    fake: Tensor,
    device: torch.device,
    k: int = 3,
    max_samples: int = 10000,
    vgg: Optional[torch.nn.Module] = None,
) -> Tuple[float, float]:
    """End-to-end Improved Precision & Recall.

    Extracts VGG16 features, then computes the manifold-based metric.

    Args:
        real:  (N, 3, H, W) real images in [0, 1].
        fake:  (M, 3, H, W) generated images in [0, 1].
        device: torch device.
        k: k-NN neighbourhood size (default 3).
        max_samples: cap to avoid OOM on large datasets.
        vgg: pre-loaded VGG16 feature extractor (reused if given).

    Returns:
        (precision, recall).
    """
    if len(real) < k + 1 or len(fake) < k + 1:
        return -1.0, -1.0

    own_vgg = vgg is None
    if own_vgg:
        vgg = _build_vgg16_feature_extractor(device)

    # Sub-sample if necessary
    if len(real) > max_samples:
        idx = torch.randperm(len(real))[:max_samples]
        real = real[idx]
    if len(fake) > max_samples:
        idx = torch.randperm(len(fake))[:max_samples]
        fake = fake[idx]

    real_feats = _extract_vgg_features(real, vgg, device)
    fake_feats = _extract_vgg_features(fake, vgg, device)

    if own_vgg:
        del vgg
        torch.cuda.empty_cache()

    return _compute_ipr_from_features(real_feats, fake_feats, k=k)


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
    use_fp16: bool = False,
) -> Dict[int, Tensor]:
    """
    Generate all jobs, returning {target_class: (N, 3, H, W)} tensors.

    Batches multiple jobs together for efficient GPU utilisation.
    Model is NOT reloaded -- it must be on *device* and in eval mode.

    When *use_fp16* is True, DDIM sampling runs inside
    ``torch.autocast("cuda", dtype=torch.float16)`` — matching the
    ``16-mixed`` precision used during training.
    """
    blur_cfg = cfg.model
    bk = getattr(blur_cfg, "blur_kernel_size", 7)
    bs_sigma = getattr(blur_cfg, "blur_sigma", 2.0)

    jobs_sorted = sorted(jobs, key=lambda j: (str(j.source_path), j.target_label))

    source_buckets: OrderedDict[Path, List[GenerationJob]] = OrderedDict()
    for j in jobs_sorted:
        source_buckets.setdefault(j.source_path, []).append(j)

    source_list = list(source_buckets.items())
    total_jobs = len(jobs)

    result_lists: Dict[int, List[Tensor]] = {c: [] for c in ALL_MES_CLASSES}

    # Set seed ONCE before all generation (not per-batch)
    _set_seed(seed)

    pbar = tqdm(total=total_jobs, desc="Generating", unit="img")
    src_idx = 0

    while src_idx < len(source_list):
        batch_sources = source_list[src_idx : src_idx + batch_images]
        src_idx += len(batch_sources)

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
            )
            for j in src_jobs:
                all_targets.append(j.target_label)
                all_sources.append(float(j.source_label))
                all_structs.append(struct)
                job_target_labels.append(j.target_label)

        B = len(all_targets)
        target_labels = torch.tensor(all_targets, dtype=torch.float32, device=device)
        source_labels = torch.tensor(all_sources, dtype=torch.float32, device=device)
        structure_images = torch.cat(all_structs, dim=0)

        ctx = torch.autocast("cuda", dtype=torch.float16) if use_fp16 else nullcontext()
        with ctx:
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
            images = _decode_latents(module, latents)

        for k in range(B):
            result_lists[job_target_labels[k]].append(images[k : k + 1])

        pbar.update(B)

    pbar.close()

    results: Dict[int, Tensor] = {}
    for cls in ALL_MES_CLASSES:
        if result_lists[cls]:
            results[cls] = torch.cat(result_lists[cls], dim=0)
        else:
            results[cls] = torch.zeros(
                0, 3, cfg.dataset.image_size, cfg.dataset.image_size
            )

    return results


def evaluate_one_run(
    generated: Dict[int, Tensor],
    real: Dict[int, Tensor],
    device: torch.device,
    seed: int = 42,
    overall_seeds: int = 5,
    fid_metric: Optional["FrechetInceptionDistance"] = None,
    clip_model: Optional[CLIPModel] = None,
    clip_proc: Optional[CLIPImageProcessor] = None,
    vgg: Optional[torch.nn.Module] = None,
) -> EvalResult:
    """Compute all metrics for one steer-scale run.

    Pre-loaded metric backbones (*fid_metric*, *clip_model*, *vgg*) are
    reused across calls to avoid redundant model loading.

    Features are extracted once per class and concatenated for overall
    metrics, avoiding redundant forward passes.

    Overall metrics use **class-balanced subsampling**: for each class,
    ``min(real_count, gen_count)`` samples are drawn from the larger set.
    This is repeated *overall_seeds* times with different random seeds
    to report mean +/- std.
    """

    _set_seed(seed)

    own_fid = fid_metric is None
    if own_fid and HAS_TORCHMETRICS:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    own_clip = clip_model is None
    if own_clip:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(
            device
        )
        clip_model.eval()
    if clip_proc is None:
        clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    own_vgg = vgg is None
    if own_vgg:
        vgg = _build_vgg16_feature_extractor(device)

    result = EvalResult()

    clip_feats_real: Dict[int, Tensor] = {}
    clip_feats_gen: Dict[int, Tensor] = {}
    vgg_feats_real: Dict[int, Tensor] = {}
    vgg_feats_gen: Dict[int, Tensor] = {}

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
        result.fid_per_class[cls] = compute_fid(
            real_cls, gen_cls, device, fid_metric=fid_metric
        )
        print(f"    FID = {result.fid_per_class[cls]:.2f}")

        # CMMD — extract CLIP features (cached for overall)
        print("    Extracting CLIP features & computing CMMD ...")
        rf = _extract_clip_features(real_cls, clip_model, clip_proc, device)
        gf = _extract_clip_features(gen_cls, clip_model, clip_proc, device)
        clip_feats_real[cls] = rf
        clip_feats_gen[cls] = gf
        result.cmmd_per_class[cls] = _mmd_rbf(rf, gf)
        print(f"    CMMD = {result.cmmd_per_class[cls]:.6f}")

        # Improved Precision & Recall — extract VGG features (cached for overall)
        print("    Extracting VGG features & computing IPR ...")
        rvgg = _extract_vgg_features(real_cls, vgg, device)
        gvgg = _extract_vgg_features(gen_cls, vgg, device)
        vgg_feats_real[cls] = rvgg
        vgg_feats_gen[cls] = gvgg
        ipr_p, ipr_r = _compute_ipr_from_features(rvgg, gvgg, k=3)
        result.ipr_precision_per_class[cls] = ipr_p
        result.ipr_recall_per_class[cls] = ipr_r
        print(f"    IPR Precision = {ipr_p:.4f}")
        print(f"    IPR Recall    = {ipr_r:.4f}")

    active_classes = [
        c for c in ALL_MES_CLASSES if c in clip_feats_real and c in clip_feats_gen
    ]
    if active_classes:
        per_class_n: Dict[int, int] = {}
        for cls in active_classes:
            n_real = len(clip_feats_real[cls])
            n_gen = len(clip_feats_gen[cls])
            per_class_n[cls] = min(n_real, n_gen)

        total_balanced = sum(per_class_n.values())
        print(
            f"\n  Overall (class-balanced): {total_balanced} samples per set "
            f"({', '.join(f'MES{c}={per_class_n[c]}' for c in active_classes)})"
        )
        print(f"  Running {overall_seeds} subsampling seeds ...")

        fid_runs: List[float] = []
        cmmd_runs: List[float] = []
        ipr_p_runs: List[float] = []
        ipr_r_runs: List[float] = []

        for si, sub_seed in enumerate(range(seed, seed + overall_seeds)):
            rng = torch.Generator().manual_seed(sub_seed)

            # Subsample features and raw images per class, then concat
            sub_clip_real_parts: List[Tensor] = []
            sub_clip_gen_parts: List[Tensor] = []
            sub_vgg_real_parts: List[Tensor] = []
            sub_vgg_gen_parts: List[Tensor] = []
            sub_raw_real_parts: List[Tensor] = []
            sub_raw_gen_parts: List[Tensor] = []

            for cls in active_classes:
                n = per_class_n[cls]
                nr = len(clip_feats_real[cls])
                ng = len(clip_feats_gen[cls])

                # Subsample the larger set; keep the smaller set intact
                if nr > n:
                    idx_r = torch.randperm(nr, generator=rng)[:n]
                else:
                    idx_r = torch.arange(nr)
                if ng > n:
                    idx_g = torch.randperm(ng, generator=rng)[:n]
                else:
                    idx_g = torch.arange(ng)

                sub_clip_real_parts.append(clip_feats_real[cls][idx_r])
                sub_clip_gen_parts.append(clip_feats_gen[cls][idx_g])
                sub_vgg_real_parts.append(vgg_feats_real[cls][idx_r])
                sub_vgg_gen_parts.append(vgg_feats_gen[cls][idx_g])
                sub_raw_real_parts.append(real[cls][idx_r])
                sub_raw_gen_parts.append(generated[cls][idx_g])

            sub_real_imgs = torch.cat(sub_raw_real_parts, dim=0)
            sub_gen_imgs = torch.cat(sub_raw_gen_parts, dim=0)
            sub_clip_real = torch.cat(sub_clip_real_parts, dim=0)
            sub_clip_gen = torch.cat(sub_clip_gen_parts, dim=0)
            sub_vgg_real = torch.cat(sub_vgg_real_parts, dim=0)
            sub_vgg_gen = torch.cat(sub_vgg_gen_parts, dim=0)

            # FID (needs raw images — InceptionV3 feature extraction)
            fid_val = compute_fid(
                sub_real_imgs, sub_gen_imgs, device, fid_metric=fid_metric
            )
            fid_runs.append(fid_val)

            # CMMD (from cached CLIP features)
            cmmd_val = _mmd_rbf(sub_clip_real, sub_clip_gen)
            cmmd_runs.append(cmmd_val)

            # IPR (from cached VGG features)
            p_val, r_val = _compute_ipr_from_features(sub_vgg_real, sub_vgg_gen, k=3)
            ipr_p_runs.append(p_val)
            ipr_r_runs.append(r_val)

            print(
                f"    seed {sub_seed}: FID={fid_val:.2f}  CMMD={cmmd_val:.6f}  "
                f"P={p_val:.4f}  R={r_val:.4f}"
            )

        result.overall_fid = float(np.mean(fid_runs))
        result.overall_fid_std = float(np.std(fid_runs))
        result.overall_cmmd = float(np.mean(cmmd_runs))
        result.overall_cmmd_std = float(np.std(cmmd_runs))
        result.overall_ipr_precision = float(np.mean(ipr_p_runs))
        result.overall_ipr_precision_std = float(np.std(ipr_p_runs))
        result.overall_ipr_recall = float(np.mean(ipr_r_runs))
        result.overall_ipr_recall_std = float(np.std(ipr_r_runs))

        print(
            f"  Overall FID  = {result.overall_fid:.2f} ± {result.overall_fid_std:.2f}"
        )
        print(
            f"  Overall CMMD = {result.overall_cmmd:.6f} ± {result.overall_cmmd_std:.6f}"
        )
        print(
            f"  Overall IPR Precision = {result.overall_ipr_precision:.4f} ± {result.overall_ipr_precision_std:.4f}"
        )
        print(
            f"  Overall IPR Recall    = {result.overall_ipr_recall:.4f} ± {result.overall_ipr_recall_std:.4f}"
        )

    if own_clip:
        del clip_model
    if own_vgg:
        del vgg
    if own_fid:
        del fid_metric
    torch.cuda.empty_cache()

    return result


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)

    n_ckpts = len(args.checkpoints)

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

    ckpt_names = args.checkpoint_names
    if ckpt_names is None:
        ckpt_names = [p.parent.parent.name for p in args.checkpoints]
    if len(ckpt_names) != n_ckpts:
        raise ValueError(
            f"Number of checkpoint names ({len(ckpt_names)}) must match "
            f"number of checkpoints ({n_ckpts})"
        )

    _init_image_pipeline(image_size)

    data_roots = args.data_root  # List[Path] now
    print(f"\nCollecting images from {[str(d) for d in data_roots]} ...")
    jobs = _collect_jobs(data_roots, max_per_class=args.max_images_per_class)
    n_sources = len(set(j.source_path for j in jobs))
    print(
        f"  {n_sources} source images -> {len(jobs)} generation jobs " f"(3 per source)"
    )

    print("\nLoading real images ...")
    real_images: Dict[int, Tensor] = {}
    for cls in ALL_MES_CLASSES:
        try:
            real_images[cls] = _load_real_images_for_class(data_roots, cls, image_size)
            print(f"  Class {cls}: {len(real_images[cls])} real images")
        except (FileNotFoundError, ValueError) as e:
            print(f"  Class {cls}: {e}")

    print("\nPre-loading metric backbones ...")
    fid_metric = None
    if HAS_TORCHMETRICS:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        print("  FID (InceptionV3) loaded")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    clip_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print("  CLIP ViT-L/14 loaded")

    vgg = _build_vgg16_feature_extractor(device)
    print("  VGG16 loaded")

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
            str(ckpt_path), cfg=ckpt_cfg, weights_only=False, strict=False
        )
        module = module.to(device).to(torch.float32)
        module.eval()

        if args.compile_unet:
            print("  Compiling UNet with torch.compile() ...")
            module.unet.unet = torch.compile(module.unet.unet)

        use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)
        use_feature_purifier = module.feature_purifier is not None
        mode_str = (
            f"routing_gates={use_routing_gates}, "
            f"feature_purifier={use_feature_purifier}"
        )
        print(f"  Mode: {mode_str}")

        for scale_val in args.scales:
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
                    group="final",
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
                use_fp16=args.fp16,
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
                seed=args.seed,
                overall_seeds=args.overall_seeds,
                fid_metric=fid_metric,
                clip_model=clip_model,
                clip_proc=clip_proc,
                vgg=vgg,
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
            print(
                f"    Overall FID  = {result.overall_fid:.2f} ± {result.overall_fid_std:.2f}"
            )
            print(
                f"    Overall CMMD = {result.overall_cmmd:.6f} ± {result.overall_cmmd_std:.6f}"
            )
            print(
                f"    IPR Prec     = {result.overall_ipr_precision:.4f} ± {result.overall_ipr_precision_std:.4f}"
            )
            print(
                f"    IPR Recall   = {result.overall_ipr_recall:.4f} ± {result.overall_ipr_recall_std:.4f}"
            )
            for cls in ALL_MES_CLASSES:
                if cls in result.fid_per_class:
                    print(
                        f"    Class {cls}: FID={result.fid_per_class[cls]:.2f}  "
                        f"CMMD={result.cmmd_per_class.get(cls, -1):.6f}  "
                        f"P={result.ipr_precision_per_class.get(cls, -1):.4f}  "
                        f"R={result.ipr_recall_per_class.get(cls, -1):.4f}"
                    )
            print(f"    Time: {elapsed:.1f}s")

        del module
        torch.cuda.empty_cache()

    print(f"\n{'=' * 100}")
    print("COMPARISON TABLE")
    print(f"{'=' * 100}")
    header = f"{'Run':<40} {'FID':>14} {'CMMD':>16} " f"{'IPR_P':>14} {'IPR_R':>14}"
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        fid_s = f"{r.overall_fid:.2f}±{r.overall_fid_std:.2f}"
        cmmd_s = f"{r.overall_cmmd:.6f}±{r.overall_cmmd_std:.6f}"
        p_s = f"{r.overall_ipr_precision:.4f}±{r.overall_ipr_precision_std:.4f}"
        r_s = f"{r.overall_ipr_recall:.4f}±{r.overall_ipr_recall_std:.4f}"
        print(
            f"{name:<40} " f"{fid_s:>14} " f"{cmmd_s:>16} " f"{p_s:>14} " f"{r_s:>14}"
        )
    print(f"{'=' * 100}")

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

    del fid_metric, clip_model, clip_proc, vgg
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
