"""
Balanced Data Augmentation via MES Progression Generation  (optimised).

For each training image in limuc_cleaned/train/, generate synthetic images
for the missing MES classes to create a balanced dataset.

Optimisations over the naive version:
  1.  **Batched generation**  – multiple source images are processed together
      so the UNet sees a large batch per DDIM step.
  2.  **FP16 inference**      – model runs in float16 (matches training precision).
  3.  **CLIP processor cache** – loaded once, not per image.
  4.  **cuDNN benchmark ON**  – faster convolutions for fixed-size tensors.
  5.  **Async BMP saving**    – images are written to disk in a background
      thread-pool so the GPU is never idle waiting on I/O.
  6.  **torch.compile (opt-in)** – set ``--compile`` to JIT-compile the UNet.

Usage:
    python -m src.pipelines.inference.inference_pipeline_ip_data_augment \
        --checkpoint prog-disease-generation-ip/uqqx9kg9/checkpoints/last.ckpt \
        --config configs/train_ip.yaml \
        --steer-scale 0.5 --image-scale 1 \
        --batch-images 4
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor

from src.models.diffusion_module_ip import DiffusionModuleWithIP

ALL_MES_CLASSES = [0, 1, 2, 3]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate balanced training data via MES progression."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/train_ip.yaml"))
    p.add_argument("--data-root", type=Path, default=Path("data/limuc_cleaned"))
    p.add_argument(
        "--output-root", type=Path, default=Path("data/limuc_cleaned_balanced")
    )
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--image-scale", type=float, default=1.0)
    p.add_argument("--steer-scale", type=float, default=0.5)
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG guidance scale (baseline mode only). "
        "Default reads from config or 1.0.",
    )
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--batch-images",
        type=int,
        default=4,
        help="Number of source images to batch together. Each source produces "
        "3 target images, so effective UNet batch = batch_images × 3. "
        "Increase for faster throughput, decrease if OOM. Default 4 → 12 per step.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Apply torch.compile to the UNet (one-time warmup cost, then faster).",
    )
    p.add_argument(
        "--save-workers",
        type=int,
        default=4,
        help="Number of threads for async BMP writing.",
    )
    return p.parse_args()


def _resolve_device(s: str) -> torch.device:
    if s != "auto":
        return torch.device(s)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_config(p: Path) -> DictConfig:
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return OmegaConf.load(p)


_CLIP_PROCESSOR: CLIPImageProcessor | None = None
_DISPLAY_TRANSFORM: transforms.Compose | None = None


def _init_image_pipeline(target_size: int) -> None:
    """Initialise reusable CLIP processor & torchvision transform (called once)."""
    global _CLIP_PROCESSOR, _DISPLAY_TRANSFORM
    _CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    _DISPLAY_TRANSFORM = transforms.Compose(
        [transforms.Resize((target_size, target_size)), transforms.ToTensor()]
    )


def _load_structure_image(
    image_path: Path,
    device: torch.device,
) -> Tensor:
    """Load one image → CLIP-ready tensor (1, 3, 224, 224) on *device*."""
    pil = Image.open(image_path).convert("RGB")
    display = _DISPLAY_TRANSFORM(pil)

    pil_out = transforms.ToPILImage()(display)
    clip_inputs = _CLIP_PROCESSOR(images=pil_out, return_tensors="pt", do_rescale=True)
    return clip_inputs.pixel_values.to(device)


def _tensor_to_bmp(image_tensor: Tensor, save_path: Path) -> None:
    """Save (3, H, W) float [0,1] tensor as .bmp  (called in worker thread)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (image_tensor.permute(1, 2, 0).mul(255).to(torch.uint8)).numpy()
    Image.fromarray(arr).save(save_path, format="BMP")


def _prepare_conditioning(
    module: DiffusionModuleWithIP,
    target_labels: Tensor,
    source_labels: Tensor,
    structure_images: Tensor,
    image_scale: float = 1.0,
    zero_aoe: bool = False,
) -> Tensor:
    """Prepare conditioning for a **batch** of (source, target) pairs.

    ``structure_images`` is (B, 3, 224, 224) -- one per sample.

    When ``use_routing_gates=True``:
        3-segment: [source_aoe | image_embeds | delta_embeds]
    When ``use_routing_gates=False``:
        2-segment: [aoe | image_embeds]

    Args:
        zero_aoe: If True, replace AOE with ``get_negative_embedding``
                  (unconditional pass for CFG).
    """
    use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)

    # Target AOE
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

    # Source AOE (for purifier)
    s_aoe = module.ordinal_embedder(source_labels, is_training=False)
    if s_aoe.dim() == 2:
        s_aoe = s_aoe.unsqueeze(1)

    # Image embeddings
    img_emb = module._get_image_embeds(structure_images)
    if module.feature_purifier is not None:
        img_emb = module.feature_purifier(img_emb, s_aoe)
    if image_scale != 1.0:
        img_emb = img_emb * image_scale

    if use_routing_gates:
        # 3-segment: [Source_AOE | E_clean | Delta_AOE]
        delta = module.ordinal_embedder.get_ordinal_delta_embedding(
            source_labels, target_labels
        )
        if delta.dim() == 2:
            delta = delta.unsqueeze(1)
        return torch.cat([s_aoe, img_emb, delta], dim=1)
    else:
        # 2-segment: [AOE | Image]
        return torch.cat([t_aoe, img_emb], dim=1)


def _set_delta_scale_on_processors(module: DiffusionModuleWithIP, s: float) -> None:
    for _, mod in module.unet.unet.named_modules():
        if hasattr(mod, "processor") and hasattr(mod.processor, "delta_scale"):
            mod.processor.delta_scale = s


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
    """DDIM sampling for an arbitrary-sized batch of (source, targets) pairs.

    Routing-gates mode: single pass (guidance_scale ignored).
    Baseline mode: CFG dual-pass when guidance_scale != 1.0.
    Unconditional AOE uses ``get_negative_embedding``.
    """
    use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)
    do_cfg = (not use_routing_gates) and (guidance_scale != 1.0)

    model_dtype = next(module.unet.parameters()).dtype

    B = target_labels.shape[0]
    H = module.cfg.dataset.image_size
    T = module.diff_cfg.num_train_timesteps
    C = module.cfg.model.latent_channels

    latents = torch.randn(B, C, H // 8, H // 8, device=device, dtype=model_dtype)
    ac = module.alphas_cumprod
    ts = torch.linspace(T - 1, 0, steps=sampling_steps, dtype=torch.long, device=device)

    embed_cond = _prepare_conditioning(
        module,
        target_labels,
        source_labels,
        structure_images,
        image_scale=image_scale,
    ).to(model_dtype)

    embed_uncond = None
    if do_cfg:
        embed_uncond = _prepare_conditioning(
            module,
            target_labels,
            source_labels,
            structure_images,
            image_scale=image_scale,
            zero_aoe=True,
        ).to(model_dtype)

    _set_delta_scale_on_processors(module, steer_scale)

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
    """Decode latents → [0, 1] RGB (B, 3, H, W) in fp32 on CPU for saving."""

    scaled = (
        latents.to(next(module.vae.parameters()).dtype) / module.diff_cfg.latent_scale
    )
    decoded = module.vae.decode(scaled)
    imgs = decoded.sample if hasattr(decoded, "sample") else decoded
    imgs = imgs.float().clamp(-1.0, 1.0)
    return ((imgs + 1.0) / 2.0).clamp(0.0, 1.0).cpu()


def _collect_pending_jobs(data_root: Path, train_dst: Path) -> list[dict]:
    """Return list of jobs: {path, stem, source_mes, targets: list[int]}.

    Only includes targets that haven't been generated yet (resume-friendly).
    """
    train_dir = data_root / "train"
    jobs: list[dict] = []
    for mes in ALL_MES_CLASSES:
        class_dir = train_dir / str(mes)
        if not class_dir.exists():
            continue
        for img in sorted(class_dir.glob("*.bmp")):
            targets = []
            for tc in ALL_MES_CLASSES:
                if tc == mes:
                    continue
                out = train_dst / str(tc) / f"{img.stem}_generated.bmp"
                if not out.exists():
                    targets.append(tc)
            if targets:
                jobs.append(
                    {
                        "path": img,
                        "stem": img.stem,
                        "source_mes": mes,
                        "targets": targets,
                    }
                )
    return jobs


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)

    data_root: Path = args.data_root
    output_root: Path = args.output_root
    train_src = data_root / "train"
    train_dst = output_root / "train"

    print(f"🔧 Device: {device}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📂 Source: {data_root}  →  Output: {output_root}")
    print(
        f"⚡ batch_images={args.batch_images}  compile={args.compile}  "
        f"save_workers={args.save_workers}"
    )

    cfg = _load_config(args.config)

    _init_image_pipeline(cfg.dataset.image_size)

    print("📦 Loading model...")
    module = DiffusionModuleWithIP.load_from_checkpoint(
        str(args.checkpoint),
        cfg=cfg,
        weights_only=False,
    )
    module = module.to(device)

    use_fp16 = device.type == "cuda"
    if use_fp16:
        module = module.half()
        print("   Using float16 inference")
    else:
        module = module.float()

    module.eval()

    use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)
    if args.guidance_scale is not None:
        guidance_scale = args.guidance_scale
    else:
        guidance_scale = getattr(module.diff_cfg, "guidance_scale", 1.0)
    if use_routing_gates:
        guidance_scale = 1.0
    print(
        f"   use_routing_gates={use_routing_gates}  " f"guidance_scale={guidance_scale}"
    )

    print(f"Doing cfg is {bool(not use_routing_gates and guidance_scale != 1.0)}")

    torch.backends.cudnn.benchmark = True

    if args.compile and hasattr(torch, "compile"):
        print("   Compiling UNet (one-time warmup) ...")
        module.unet.unet = torch.compile(module.unet.unet, mode="reduce-overhead")

    print("\n📋 Step 1: Copying original training images ...")
    for mes in ALL_MES_CLASSES:
        src_d = train_src / str(mes)
        dst_d = train_dst / str(mes)
        dst_d.mkdir(parents=True, exist_ok=True)
        if not src_d.exists():
            continue
        for f in src_d.glob("*.bmp"):
            dst = dst_d / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

    for split in ("val", "test"):
        s, d = data_root / split, output_root / split
        if s.exists() and not d.exists():
            print(f"   Copying {split}/ ...")
            shutil.copytree(s, d)

    print("\n📋 Step 2: Collecting pending generation jobs ...")
    jobs = _collect_pending_jobs(data_root, train_dst)

    total_targets = sum(len(j["targets"]) for j in jobs)
    print(
        f"   Pending images to generate: {total_targets}  "
        f"(from {len(jobs)} source images)"
    )
    if total_targets == 0:
        print("   Nothing to do — all images already generated.")
        return

    print("\n🎨 Step 3: Generating (batched) ...")
    generated_counts = {m: 0 for m in ALL_MES_CLASSES}
    saver = ThreadPoolExecutor(max_workers=args.save_workers)
    futures = []

    batch_images = args.batch_images
    pbar = tqdm(total=total_targets, desc="Generating", unit="img")

    idx = 0
    while idx < len(jobs):
        batch_jobs = jobs[idx : idx + batch_images]
        idx += len(batch_jobs)

        all_targets: list[int] = []
        all_sources: list[float] = []
        all_structs: list[Tensor] = []
        all_stems: list[str] = []
        all_target_mes: list[int] = []

        for job in batch_jobs:
            struct = _load_structure_image(
                job["path"],
                device,
            )
            for tc in job["targets"]:
                all_targets.append(tc)
                all_sources.append(float(job["source_mes"]))
                all_structs.append(struct)  # will be cat'd
                all_stems.append(job["stem"])
                all_target_mes.append(tc)

        B = len(all_targets)
        model_dtype = next(module.unet.parameters()).dtype
        target_labels = torch.tensor(all_targets, dtype=model_dtype, device=device)
        source_labels = torch.tensor(all_sources, dtype=model_dtype, device=device)
        structure_images = torch.cat(all_structs, dim=0)

        latents = _ddim_sample_batched(
            module,
            target_labels,
            source_labels,
            structure_images,
            sampling_steps=args.sampling_steps,
            device=device,
            eta=args.eta,
            image_scale=args.image_scale,
            steer_scale=args.steer_scale,
            guidance_scale=guidance_scale,
        )

        images = _decode_latents(module, latents)

        for k in range(B):
            stem = all_stems[k]
            tmes = all_target_mes[k]
            out_path = train_dst / str(tmes) / f"{stem}_generated.bmp"
            fut = saver.submit(_tensor_to_bmp, images[k], out_path)
            futures.append(fut)
            generated_counts[tmes] += 1

        pbar.update(B)

    pbar.close()

    for f in futures:
        f.result()
    saver.shutdown(wait=True)

    print("\n✅ Data augmentation complete!")
    print(f"   Generated per class: {generated_counts}")

    final = {}
    for mes in ALL_MES_CLASSES:
        d = train_dst / str(mes)
        final[mes] = len(list(d.glob("*.bmp"))) if d.exists() else 0
    print(f"   Final class distribution: {final}")
    print(f"   Output: {output_root}")


if __name__ == "__main__":
    main()
