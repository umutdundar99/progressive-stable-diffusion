"""
ResNet Regression Module for MES Scoring.

PyTorch Lightning module implementing ResNet-based regressor
for continuous MES prediction. Used as a "judge" to evaluate
synthetic image quality independently of classification.

Metrics: RMSE, MAE, Accuracy (rounded), QWK (rounded).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

NUM_CLASSES = 4  # for QWK / accuracy rounding


class ResNetRegressor(pl.LightningModule):
    """
    ResNet-based regressor for continuous MES prediction.

    Outputs a single scalar (predicted MES value) and computes:
        - RMSE: Root Mean Squared Error
        - MAE:  Mean Absolute Error
        - Accuracy:  after rounding prediction to nearest integer
        - QWK:  Quadratic Weighted Kappa after rounding
    """

    RESNET_VARIANTS = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.num_classes = cfg.model.get("num_classes", NUM_CLASSES)

        # Build model
        self.model = self._build_model()

        # Internal accumulators for epoch-level metrics
        for prefix in ("train", "val", "test"):
            self.register_buffer(f"_{prefix}_sum_se", torch.tensor(0.0))
            self.register_buffer(f"_{prefix}_sum_ae", torch.tensor(0.0))
            self.register_buffer(
                f"_{prefix}_correct", torch.tensor(0, dtype=torch.long)
            )
            self.register_buffer(f"_{prefix}_total", torch.tensor(0, dtype=torch.long))

        # Store predictions for QWK computation
        self._test_preds: List[Tensor] = []
        self._test_targets: List[Tensor] = []
        self._val_preds: List[Tensor] = []
        self._val_targets: List[Tensor] = []

    def _build_model(self) -> nn.Module:
        model_name = self.cfg.model.name.lower()
        if model_name not in self.RESNET_VARIANTS:
            raise ValueError(f"Unknown model: {model_name}")

        weights = "IMAGENET1K_V1" if self.cfg.model.pretrained else None
        model = self.RESNET_VARIANTS[model_name](weights=weights)

        num_features = model.fc.in_features
        dropout = self.cfg.model.get("dropout", 0.0)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 1),
        )
        return model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass → (B, 1) raw predictions."""
        return self.model(x)

    # ── Step logic ────────────────────────────────────────────────────────────

    def _step(self, batch: Tuple[Tensor, Tensor], prefix: str) -> Tensor:
        images, targets = batch
        targets = targets.float()  # ensure float for regression
        preds = self(images).squeeze(-1)  # (B,)

        loss = F.mse_loss(preds, targets)

        # Accumulate metrics
        se = (preds - targets).pow(2).sum()
        ae = (preds - targets).abs().sum()
        rounded = preds.clamp(0, 3).round().long()
        int_targets = targets.round().long()
        correct = (rounded == int_targets).sum()

        getattr(self, f"_{prefix}_sum_se").add_(se.detach())
        getattr(self, f"_{prefix}_sum_ae").add_(ae.detach())
        getattr(self, f"_{prefix}_correct").add_(correct.detach())
        getattr(self, f"_{prefix}_total").add_(len(targets))

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss = self._step(batch, "val")
        preds = self(images).squeeze(-1)
        self._val_preds.append(preds.detach())
        self._val_targets.append(targets.detach().float())
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        loss = self._step(batch, "test")
        preds = self(images).squeeze(-1)
        self._test_preds.append(preds.detach())
        self._test_targets.append(targets.detach().float())
        return loss

    # ── Epoch-end logging ─────────────────────────────────────────────────────

    def _log_epoch_metrics(self, prefix: str) -> None:
        total = getattr(self, f"_{prefix}_total").float()
        if total == 0:
            return

        rmse = torch.sqrt(getattr(self, f"_{prefix}_sum_se") / total)
        mae = getattr(self, f"_{prefix}_sum_ae") / total
        acc = getattr(self, f"_{prefix}_correct").float() / total

        self.log(f"{prefix}/rmse", rmse, prog_bar=(prefix == "val"))
        self.log(f"{prefix}/mae", mae)
        self.log(f"{prefix}/accuracy", acc, prog_bar=(prefix == "val"))

        # QWK from stored predictions
        preds_list = getattr(self, f"_{prefix}_preds", [])
        targets_list = getattr(self, f"_{prefix}_targets", [])
        if preds_list:
            all_preds = torch.cat(preds_list)
            all_targets = torch.cat(targets_list)
            qwk = self._compute_qwk(all_preds, all_targets)
            self.log(f"{prefix}/qwk", qwk, prog_bar=(prefix == "val"))

        # Reset
        getattr(self, f"_{prefix}_sum_se").zero_()
        getattr(self, f"_{prefix}_sum_ae").zero_()
        getattr(self, f"_{prefix}_correct").zero_()
        getattr(self, f"_{prefix}_total").zero_()

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")
        self._val_preds.clear()
        self._val_targets.clear()

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")
        self._test_preds.clear()
        self._test_targets.clear()

    # ── QWK ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_qwk(preds: Tensor, targets: Tensor) -> Tensor:
        """Compute Quadratic Weighted Kappa from continuous predictions.

        Both preds and targets are rounded to nearest int ∈ {0,1,2,3}.
        """
        N = NUM_CLASSES
        p = preds.clamp(0, N - 1).round().long()
        t = targets.clamp(0, N - 1).round().long()

        # Confusion matrix
        O = torch.zeros(N, N, device=preds.device)
        for i in range(len(p)):
            O[t[i], p[i]] += 1

        # Weight matrix W[i,j] = (i-j)^2 / (N-1)^2
        W = torch.zeros(N, N, device=preds.device)
        for i in range(N):
            for j in range(N):
                W[i, j] = (i - j) ** 2 / (N - 1) ** 2

        # Expected matrix
        hist_t = O.sum(dim=1)
        hist_p = O.sum(dim=0)
        total = O.sum()
        E = hist_t.unsqueeze(1) * hist_p.unsqueeze(0) / total

        num = (W * O).sum()
        den = (W * E).sum()

        if den == 0:
            return torch.tensor(1.0, device=preds.device)
        return 1.0 - num / den

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> Dict[str, Any]:
        lr = self.cfg.training.learning_rate
        wd = self.cfg.training.weight_decay
        optimizer = AdamW(self.parameters(), lr=lr, weight_decay=wd)

        scheduler_name = self.cfg.training.get("scheduler", "cosine").lower()
        if scheduler_name == "cosine":
            sp = self.cfg.training.get("scheduler_params", {})
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=sp.get("T_max", self.cfg.training.max_epochs),
                eta_min=sp.get("eta_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return {"optimizer": optimizer}
