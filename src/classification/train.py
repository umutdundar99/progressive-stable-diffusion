"""
Training script for MES Classification with ResNet.

Usage:
    python -m src.classification.train --config configs/train_classifier.yaml

    # Override config values:
    python -m src.classification.train --config configs/train_classifier.yaml \
        --batch-size 64 --lr 0.0001 --epochs 50
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
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

# Fix for PyTorch 2.6+ checkpoint loading - set weights_only=False by default
# This is safe since we trust our own checkpoints
torch.load = partial(torch.load, weights_only=False)

from src.classification.dataset import MESDataModule
from src.classification.model import ResNetClassifier


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ResNet classifier for MES classification."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_classifier.yaml"),
        help="Path to configuration file.",
    )

    # Override arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        dest="learning_rate",
        help="Override learning rate from config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override max epochs from config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        help="Override model architecture.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override data root directory.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Override experiment name for logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run testing on a trained model.",
    )

    return parser.parse_args()


def load_config(config_path: Path, args: argparse.Namespace) -> DictConfig:
    """Load and merge configuration with command line overrides."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    # Apply command line overrides
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
    if args.seed is not None:
        cfg.seed = args.seed

    return cfg


def setup_callbacks(cfg: DictConfig) -> list:
    """Set up training callbacks."""
    callbacks = []

    # Model checkpoint
    checkpoint_dir = Path(cfg.checkpoint.dirpath)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = checkpoint_dir / timestamp

    # callbacks.append(
    #     ModelCheckpoint(
    #         dirpath=checkpoint_dir,
    #         filename=cfg.checkpoint.filename,
    #         monitor=cfg.logging.monitor_metric,
    #         mode=cfg.logging.monitor_mode,
    #         save_top_k=cfg.logging.save_top_k,
    #         save_last=cfg.checkpoint.save_last,
    #         verbose=True,
    #     )
    # )

    # Early stopping
    if cfg.get("early_stopping"):
        callbacks.append(
            EarlyStopping(
                monitor=cfg.logging.monitor_metric,
                mode=cfg.logging.monitor_mode,
                patience=cfg.early_stopping.patience,
                min_delta=cfg.early_stopping.min_delta,
                verbose=True,
            )
        )

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Progress bar
    callbacks.append(TQDMProgressBar(refresh_rate=10))

    return callbacks


def setup_logger(cfg: DictConfig) -> WandbLogger | None:
    """Set up wandb logger."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.logging.experiment_name}_{timestamp}"

    logger = WandbLogger(
        project=cfg.logging.project_name,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=False,
        offline=cfg.logging.offline,
    )

    return logger


def main() -> None:
    """Main training function."""
    args = parse_args()
    cfg = load_config(args.config, args)

    # Print configuration
    print("=" * 60)
    print("MES Classification Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Setup data module
    print("\n📦 Setting up data module...")
    data_module = MESDataModule(
        data_root=cfg.dataset.data_root,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        augmentation_cfg=OmegaConf.to_container(cfg.augmentation, resolve=True),
        use_weighted_sampler=True,
    )

    # Setup data to get class weights
    data_module.setup("fit")
    class_weights = data_module.get_class_weights()

    if class_weights is not None:
        print(f"\n⚖️  Class weights: {class_weights.tolist()}")

    # Use config-specified class weights if provided
    if cfg.training.class_weights is not None:
        class_weights = torch.tensor(cfg.training.class_weights, dtype=torch.float32)
        print(f"📝 Using config-specified class weights: {class_weights.tolist()}")

    # Setup model
    print("\n🧠 Setting up model...")
    if args.resume and not args.test_only:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        model = ResNetClassifier.load_from_checkpoint(
            str(args.resume),
            cfg=cfg,
            class_weights=class_weights,
        )
    else:
        model = ResNetClassifier(cfg=cfg, class_weights=class_weights)

    print(f"   Model: {cfg.model.name}")
    print(f"   Pretrained: {cfg.model.pretrained}")
    print(f"   Num classes: {cfg.model.num_classes}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Setup callbacks and logger
    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)

    # Setup trainer
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
        # Test only mode
        if args.resume is None:
            raise ValueError("--resume must be specified for --test-only mode")

        print("\n🧪 Running test evaluation...")
        model = ResNetClassifier.load_from_checkpoint(
            str(args.resume),
            cfg=cfg,
            class_weights=class_weights,
        )
        data_module.setup("test")
        trainer.test(model, datamodule=data_module)
    else:
        # Training mode
        print("\n🚀 Starting training...")
        trainer.fit(model, datamodule=data_module)

        # Test after training
        print("\n🧪 Running final test evaluation...")
        data_module.setup("test")
        trainer.test(model, datamodule=data_module, ckpt_path="best")

    # Save final config
    # if logger is not None:
    # config_save_path = Path(callbacks[0].dirpath) / "config.yaml"
    # OmegaConf.save(cfg, config_save_path)
    # print(f"\n💾 Config saved to: {config_save_path}")

    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
