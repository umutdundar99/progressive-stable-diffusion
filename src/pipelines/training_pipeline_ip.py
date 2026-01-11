"""
Training pipeline for IP-Adapter with Ordinal Disease Conditioning.

This pipeline trains a diffusion model with dual conditioning:
1. AOE (Additive Ordinal Embedding) for disease severity
2. Image features for patient-specific anatomical structure
"""

from __future__ import annotations

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


@hydra.main(version_base="1.4", config_path="../../configs", config_name="train_ip")
def main(cfg: DictConfig) -> None:
    """Main training function for IP-Adapter."""

    datamodule = OrdinalIPDataModule(cfg)

    diffusion = DiffusionModuleWithIP(cfg)

    # Logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        log_model=False,
        offline=cfg.wandb.offline,
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

    trainer.fit(model=diffusion, datamodule=datamodule)


if __name__ == "__main__":
    main()
