"""Training pipeline for IP-Adapter with Ordinal Disease Conditioning.

This pipeline trains a diffusion model with dual conditioning:
1. AOE (Additive Ordinal Embedding) for disease severity
2. Image features for patient-specific anatomical structure

Supports resuming from a checkpoint via `training.resume_checkpoint`:
- null  → train from scratch
- path  → resume from that .ckpt file (weights, optimizer, LR scheduler, epoch, etc.)
"""

from __future__ import annotations

from pathlib import Path

import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src.callbacks.ema_callback import EMAWeightAveraging
from src.data.datamodule_ip import OrdinalIPDataModule
from src.models.diffusion_module_ip import DiffusionModuleWithIP


def _resolve_checkpoint_path(cfg: DictConfig) -> str | None:
    """Resolve the checkpoint path from config.

    Returns:
        Absolute path string to the checkpoint, or None to train from scratch.
    """
    ckpt_path = cfg.training.get("resume_checkpoint", None)
    if ckpt_path is None:
        return None

    ckpt_path = str(ckpt_path)

    if ckpt_path.lower() == "last":
        return "last"

    resolved = Path(ckpt_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {resolved}\n"
            "Set training.resume_checkpoint=null to train from scratch."
        )
    return str(resolved)


@hydra.main(version_base="1.4", config_path="../../../configs", config_name="train_ip")
def main(cfg: DictConfig) -> None:
    """Main training function for IP-Adapter."""

    ckpt_path = _resolve_checkpoint_path(cfg)
    if ckpt_path is not None:
        print(f"🔄 Resuming training from checkpoint: {ckpt_path}")
    else:
        print("🆕 Starting training from scratch.")

    datamodule = OrdinalIPDataModule(cfg)

    diffusion = DiffusionModuleWithIP(cfg)

    resume_wandb = (
        "must" if ckpt_path is not None and cfg.wandb.get("run_id", None) else "allow"
    )
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        id=cfg.wandb.get("run_id", None),
        resume=resume_wandb,
        log_model=False,
        offline=cfg.wandb.offline,
        group=cfg.wandb.get("group", None),
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=50,
        save_last=True,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        filename="ip-ddpm-epoch{epoch:04d}",
    )

    ema_callback = EMAWeightAveraging(
        decay=cfg.training.ema_decay,
        update_starting_at_step=cfg.training.update_starting_at_step,
        update_every_n_steps=cfg.training.update_every_n_steps,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        ema_callback,
        checkpoint_callback,
        lr_monitor,
    ]

    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        precision=cfg.training.precision,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy="ddp_find_unused_parameters_false"
        if cfg.training.strategy == "ddp"
        else cfg.training.strategy,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=False,
        enable_checkpointing=True,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        benchmark=True,
    )

    trainer.fit(
        model=diffusion, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False
    )


if __name__ == "__main__":
    main()
