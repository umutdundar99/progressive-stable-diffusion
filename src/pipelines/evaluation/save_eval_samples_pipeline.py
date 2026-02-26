"""
Save generated evaluation samples for multiple checkpoints and scales.

This script does NOT compute metrics. It only generates and saves images.

Use case:
- Pick a fixed subset from test split (e.g. 10 images per MES class => 40 sources)
- For each source image, generate 3 target classes (excluding source class)
- Repeat for each checkpoint and each scale value

Scale interpretation (same as evaluation_pipeline.py)
-----------------------------------------------------
- use_routing_gates=True  -> scale = steer_scale, guidance_scale=1.0
- use_routing_gates=False -> scale = guidance_scale, steer_scale=0.0
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from src.models.diffusion_module_ip import DiffusionModuleWithIP
from src.pipelines.evaluation.evaluation_pipeline import (
    ALL_MES_CLASSES,
    IMAGE_EXTENSIONS,
    GenerationJob,
    _ddim_sample_batched,
    _decode_latents,
    _init_image_pipeline,
    _load_config,
    _load_structure_image,
    _resolve_device,
    _set_seed,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate and save fixed evaluation samples for multiple models/scales."
    )
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--checkpoint-names", type=str, nargs="+", default=None)
    p.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        required=True,
        help="One config per checkpoint, or a single shared config.",
    )
    p.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/limuc_cleaned/test"),
        help="Test split root containing class folders 0/1/2/3.",
    )
    p.add_argument(
        "--num-per-class",
        type=int,
        default=10,
        help="How many source images to select from each class.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/eval_samples"))
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--image-scale", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--batch-images", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--no-blur",
        action="store_true",
        default=True,
        help="Skip blurring the structure image.",
    )
    return p.parse_args()


def _select_sources(data_root: Path, num_per_class: int) -> Dict[int, List[Path]]:
    selected: Dict[int, List[Path]] = {}
    for cls in ALL_MES_CLASSES:
        class_dir = data_root / str(cls)
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        paths = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(paths) < num_per_class:
            raise ValueError(
                f"Class {cls} has only {len(paths)} images, requested {num_per_class}."
            )
        selected[cls] = paths[:num_per_class]
    return selected


def _build_jobs(selected_sources: Dict[int, List[Path]]) -> List[GenerationJob]:
    jobs: List[GenerationJob] = []
    for source_label, paths in selected_sources.items():
        for source_path in paths:
            for target_label in ALL_MES_CLASSES:
                if target_label != source_label:
                    jobs.append(
                        GenerationJob(
                            source_path=source_path,
                            source_label=source_label,
                            target_label=target_label,
                        )
                    )
    return jobs


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = img.permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
    return Image.fromarray(arr)


def _build_side_by_side(
    source_pil: Image.Image,
    generated_by_target: Dict[int, torch.Tensor],
) -> Image.Image:
    labels = ["ORIGINAL"]
    tiles = [source_pil]

    for target in sorted(generated_by_target.keys()):
        labels.append(f"TARGET {target}")
        tiles.append(_tensor_to_pil(generated_by_target[target]))

    tile_w = max(t.width for t in tiles)
    tile_h = max(t.height for t in tiles)
    label_h = 28
    gap = 6
    width = len(tiles) * tile_w + (len(tiles) - 1) * gap
    height = tile_h + label_h
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)

    x = 0
    for tile, label in zip(tiles, labels):
        if tile.size != (tile_w, tile_h):
            tile = tile.resize((tile_w, tile_h))
        canvas.paste(tile, (x, label_h))

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        draw.text(
            (x + max((tile_w - text_w) // 2, 0), 6),
            label,
            fill=(20, 20, 20),
            font=font,
        )
        x += tile_w + gap

    return canvas


@torch.inference_mode()
def _generate_and_save(
    module: DiffusionModuleWithIP,
    jobs: List[GenerationJob],
    cfg: DictConfig,
    device: torch.device,
    save_root: Path,
    batch_images: int,
    sampling_steps: int,
    image_scale: float,
    steer_scale: float,
    guidance_scale: float,
    eta: float,
    apply_blur: bool,
    seed: int,
) -> None:
    blur_cfg = cfg.model
    blur_kernel_size = getattr(blur_cfg, "blur_kernel_size", 7)
    blur_sigma = getattr(blur_cfg, "blur_sigma", 2.0)

    jobs_sorted = sorted(jobs, key=lambda j: (str(j.source_path), j.target_label))

    source_buckets: OrderedDict[Path, List[GenerationJob]] = OrderedDict()
    for j in jobs_sorted:
        source_buckets.setdefault(j.source_path, []).append(j)

    source_list = list(source_buckets.items())
    src_idx = 0

    _set_seed(seed)

    while src_idx < len(source_list):
        batch_sources = source_list[src_idx : src_idx + batch_images]
        src_idx += len(batch_sources)

        all_targets: List[int] = []
        all_sources: List[float] = []
        all_structs: List[torch.Tensor] = []
        all_jobs: List[GenerationJob] = []

        for src_path, src_jobs in batch_sources:
            struct = _load_structure_image(
                src_path,
                device,
                apply_blur=apply_blur,
                blur_kernel_size=blur_kernel_size,
                blur_sigma=blur_sigma,
            )
            for j in src_jobs:
                all_targets.append(j.target_label)
                all_sources.append(float(j.source_label))
                all_structs.append(struct)
                all_jobs.append(j)

        target_labels = torch.tensor(all_targets, dtype=torch.float32, device=device)
        source_labels = torch.tensor(all_sources, dtype=torch.float32, device=device)
        structure_images = torch.cat(all_structs, dim=0)

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

        grouped_generated: Dict[Path, Dict[int, torch.Tensor]] = {}
        grouped_source_label: Dict[Path, int] = {}
        for idx, job in enumerate(all_jobs):
            grouped_generated.setdefault(job.source_path, {})[job.target_label] = images[idx]
            grouped_source_label[job.source_path] = job.source_label

        for src_path, by_target in grouped_generated.items():
            src_img = Image.open(src_path).convert("RGB")
            src_img = src_img.resize((cfg.dataset.image_size, cfg.dataset.image_size))

            side_by_side = _build_side_by_side(src_img, by_target)
            src_cls = grouped_source_label[src_path]
            out_path = save_root / f"{src_path.stem}.png"
            if out_path.exists():
                out_path = save_root / f"{src_path.stem}_source_{src_cls}.png"
            side_by_side.save(out_path)


def _resolve_checkpoint_names(checkpoints: List[Path], provided: List[str] | None) -> List[str]:
    if provided is None:
        return [p.parent.parent.name for p in checkpoints]
    if len(provided) != len(checkpoints):
        raise ValueError(
            f"checkpoint-names count ({len(provided)}) must match checkpoints ({len(checkpoints)})."
        )
    return provided


def _resolve_configs(config_paths: List[Path], n_ckpts: int) -> List[DictConfig]:
    if len(config_paths) == 1:
        cfg = _load_config(config_paths[0])
        return [cfg] * n_ckpts
    if len(config_paths) == n_ckpts:
        return [_load_config(p) for p in config_paths]
    raise ValueError(
        f"configs must be either 1 or {n_ckpts}, got {len(config_paths)}"
    )


def _serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed)

    checkpoints = args.checkpoints
    n_ckpts = len(checkpoints)
    cfgs = _resolve_configs(args.configs, n_ckpts)
    ckpt_names = _resolve_checkpoint_names(checkpoints, args.checkpoint_names)

    image_size = cfgs[0].dataset.image_size
    _init_image_pipeline(image_size)

    selected_sources = _select_sources(args.data_root, args.num_per_class)
    jobs = _build_jobs(selected_sources)

    print(f"Selected {sum(len(v) for v in selected_sources.values())} source images.")
    print(f"Total generation jobs: {len(jobs)} (3 per source).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "data_root": str(args.data_root),
        "num_per_class": args.num_per_class,
        "source_counts": {str(k): len(v) for k, v in selected_sources.items()},
        "sources": {str(k): [str(p) for p in v] for k, v in selected_sources.items()},
        "runs": [],
    }

    for ckpt, name, cfg in zip(checkpoints, ckpt_names, cfgs):
        print(f"\nLoading model: {name} ({ckpt})")
        module = DiffusionModuleWithIP.load_from_checkpoint(
            str(ckpt), cfg=cfg, weights_only=False
        )
        module = module.to(device).to(torch.float32)
        module.eval()

        use_routing_gates = getattr(module.diff_cfg, "use_routing_gates", True)

        for scale in args.scales:
            if use_routing_gates:
                steer_scale = scale
                guidance_scale = 1.0
                scale_label = f"steer_{scale:.2f}"
            else:
                steer_scale = 0.0
                guidance_scale = scale
                scale_label = f"cfg_{scale:.2f}"

            run_dir = args.output_dir / name / scale_label
            run_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"  Generating {name} @ {scale_label} "
                f"(steer={steer_scale}, cfg={guidance_scale})"
            )

            _generate_and_save(
                module=module,
                jobs=jobs,
                cfg=cfg,
                device=device,
                save_root=run_dir,
                batch_images=args.batch_images,
                sampling_steps=args.sampling_steps,
                image_scale=args.image_scale,
                steer_scale=steer_scale,
                guidance_scale=guidance_scale,
                eta=args.eta,
                apply_blur=not args.no_blur,
                seed=args.seed,
            )

            manifest["runs"].append(
                {
                    "checkpoint": str(ckpt),
                    "checkpoint_name": name,
                    "use_routing_gates": bool(use_routing_gates),
                    "scale": float(scale),
                    "steer_scale": float(steer_scale),
                    "guidance_scale": float(guidance_scale),
                    "output_dir": str(run_dir),
                }
            )

        del module
        torch.cuda.empty_cache()

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(_serializable(manifest), f, indent=2)
    print(f"\nDone. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
