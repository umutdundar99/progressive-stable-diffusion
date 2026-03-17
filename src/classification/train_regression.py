"""
Training script for MES Regression with ResNet.

Trains a ResNet-18 regression model on real LIMUC data.
Used as a "judge" to evaluate synthetic image quality.

Usage:
    python -m src.classification.train_regression \
        --config configs/train_classifier_regression.yaml

    # Test-only with existing checkpoint:
    python -m src.classification.train_regression \
        --config configs/train_classifier_regression.yaml \
        --test-only --resume outputs/classifier_regression/last.ckpt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

torch.load = partial(torch.load, weights_only=False)

from src.classification.dataset import MESDataModule
from src.classification.model_regression import ResNetRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet regressor for MES scoring (judge model)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_classifier_regression.yaml"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument(
        "--seed", type=int, help="Override dataset balancing seed only."
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path, args: argparse.Namespace) -> DictConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.epochs is not None:
        cfg.training.max_epochs = args.epochs
    if args.model is not None:
        cfg.model.name = args.model
    if args.data_root is not None:
        cfg.dataset.data_root = str(args.data_root)
    if args.experiment_name is not None:
        cfg.logging.experiment_name = args.experiment_name
    return cfg


def setup_callbacks(cfg: DictConfig) -> list:
    callbacks = []

    checkpoint_dir = Path(cfg.checkpoint.dirpath)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = checkpoint_dir / timestamp

    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=cfg.checkpoint.filename,
            monitor=cfg.logging.monitor_metric,
            mode=cfg.logging.monitor_mode,
            save_top_k=cfg.logging.save_top_k,
            save_last=True,
            verbose=True,
        )
    )

    callbacks.append(
        EarlyStopping(
            monitor=cfg.logging.monitor_metric,
            mode=cfg.logging.monitor_mode,
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            verbose=True,
        )
    )

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    callbacks.append(TQDMProgressBar(refresh_rate=10))

    return callbacks


def setup_logger(cfg: DictConfig) -> WandbLogger | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.logging.experiment_name}_{timestamp}"

    logger = WandbLogger(
        project=cfg.logging.project_name,
        group=cfg.logging.group_name,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=False,
        offline=cfg.logging.offline,
    )
    return logger


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)

    print("=" * 60)
    print("MES Regression Training (Judge Model)")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    pl.seed_everything(cfg.seed, workers=True)

    print("\n📦 Setting up data module...")
    data_module = MESDataModule(
        data_root=cfg.dataset.data_root,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        augmentation_cfg=OmegaConf.to_container(cfg.augmentation, resolve=True),
        use_weighted_sampler=False,
        balance_seed=args.seed if args.seed is not None else 4,
    )

    print("\n🧠 Setting up regression model...")
    if args.resume and not args.test_only:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        model = ResNetRegressor.load_from_checkpoint(str(args.resume), cfg=cfg)
    else:
        model = ResNetRegressor(cfg=cfg)

    print(f"   Model: {cfg.model.name}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)

    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=True,
        enable_progress_bar=True,
    )

    if args.test_only:
        if args.resume is None:
            raise ValueError("--resume must be specified for --test-only mode")
        print("\n🧪 Running test evaluation...")
        model = ResNetRegressor.load_from_checkpoint(str(args.resume), cfg=cfg)
        data_module.setup("test")
        trainer.test(model, datamodule=data_module)
    else:
        print("\n🚀 Starting training...")
        data_module.setup("fit")
        trainer.fit(model, datamodule=data_module)

        print("\n🧪 Running final test evaluation...")
        data_module.setup("test")
        trainer.test(
            model, datamodule=data_module, ckpt_path="best", weights_only=False
        )

    print("\n✅ Regression training completed!")


if __name__ == "__main__":
    main()
