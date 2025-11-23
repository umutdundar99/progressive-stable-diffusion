"""
Training pipeline integrating Hydra, Lightning, and WandB.
"""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

import hydra
from omegaconf import DictConfig

from src.data.datamodule import OrdinalDataModule
from src.models.diffusion_module import DiffusionModule


class EMACallback(Callback):
    """Optional EMA diagnostics callback."""

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: DiffusionModule):
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return
        # If EMA metrics needed, implement pl_module._compute_ema_norm()
        trainer.logger.experiment.log(
            {"ema/update_step": trainer.global_step}
        )


@hydra.main(version_base="1.4", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.training.seed, workers=True)


    datamodule = OrdinalDataModule(cfg)
    diffusion = DiffusionModule(cfg)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        log_model=False,
        offline=cfg.wandb.offline,
    )


    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        every_n_train_steps=cfg.training.val_check_interval,
        filename="ddpm-step{step:06d}",
        save_last=True,
        monitor=None,  # step-based checkpointing, no metric needed
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        EMACallback(),
        checkpoint_callback,
        lr_monitor,
    ]


    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_steps=cfg.training.max_steps,
        val_check_interval=cfg.training.val_check_interval,
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
    )


    trainer.fit(model=diffusion, datamodule=datamodule)


if __name__ == "__main__":
    main()
