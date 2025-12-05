"""
Training pipeline integrating Hydra, Lightning, and WandB.
"""

from __future__ import annotations

import lightning as pl
from lightning import Trainer

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    )

from src.callbacks.ema_callback import EMAWeightAveraging

from lightning.pytorch.loggers import WandbLogger
# from lightning.loggers import WandbLogger

import hydra
from omegaconf import DictConfig

from src.data.datamodule import OrdinalDataModule
from src.models.diffusion_module import DiffusionModule


@hydra.main(version_base="1.4", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.training.seed, workers=True)
    datamodule = OrdinalDataModule(cfg)
    diffusion = DiffusionModule(cfg)

    wandb_logger = WandbLogger(
    project=cfg.wandb.project,
    name=cfg.wandb.run_name,
    log_model="all" if not cfg.wandb.offline else False,
    offline=cfg.wandb.offline,
)

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,            
        save_last=True,             
        save_top_k=1,                
        save_on_train_epoch_end=True,
        filename="ddpm-epoch{epoch:04d}"
    )

    ema_callback= EMAWeightAveraging(
        decay=cfg.training.ema_decay,
        update_starting_at_step=cfg.training.update_starting_at_step,
        update_every_n_steps =cfg.training.update_every_n_steps

    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        ema_callback,
        checkpoint_callback,
        lr_monitor,
    ]


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
    )


    trainer.fit(model=diffusion, datamodule=datamodule)


if __name__ == "__main__":
    main()
