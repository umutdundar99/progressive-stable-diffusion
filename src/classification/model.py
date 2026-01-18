"""
ResNet Classification Module for MES Scoring.

PyTorch Lightning module implementing ResNet-based classifier
with comprehensive metrics logging for medical image classification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torchmetrics import (
    AUROC,
    Accuracy,
    CohenKappa,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    Specificity,
)
from torchmetrics.classification import MulticlassCalibrationError
from torchvision import models


class ResNetClassifier(pl.LightningModule):
    """
    ResNet-based classifier for MES (Mayo Endoscopic Score) classification.

    Features:
        - Pretrained ResNet backbone (18, 34, 50, 101)
        - Comprehensive metrics: Accuracy, Precision, Recall, F1, QWK, AUROC
        - Per-class metrics tracking
        - Configurable optimizer and scheduler
        - Label smoothing support
        - Optional backbone freezing
    """

    RESNET_VARIANTS = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }

    def __init__(
        self,
        cfg: DictConfig,
        class_weights: Optional[Tensor] = None,
    ) -> None:
        """
        Initialize the classifier.

        Args:
            cfg: Configuration object containing model and training parameters
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.cfg = cfg
        self.num_classes = cfg.model.num_classes

        # Build model
        self.model = self._build_model()

        # Loss function
        self.class_weights = class_weights
        self.label_smoothing = cfg.training.get("label_smoothing", 0.0)

        # Metrics for training
        self._setup_metrics("train")
        self._setup_metrics("val")
        self._setup_metrics("test")

        # For confusion matrix logging
        self.val_preds: List[Tensor] = []
        self.val_targets: List[Tensor] = []
        self.test_preds: List[Tensor] = []
        self.test_targets: List[Tensor] = []

        # Track current epoch for backbone unfreezing
        self.freeze_backbone_epochs = cfg.model.get("freeze_backbone_epochs", 0)

    def _build_model(self) -> nn.Module:
        """Build ResNet model with custom classifier head."""
        model_name = self.cfg.model.name.lower()

        if model_name not in self.RESNET_VARIANTS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choose from: {list(self.RESNET_VARIANTS.keys())}"
            )

        # Load pretrained model
        weights = "IMAGENET1K_V1" if self.cfg.model.pretrained else None
        model = self.RESNET_VARIANTS[model_name](weights=weights)

        # Get the number of features from the last layer
        num_features = model.fc.in_features

        # Replace classifier head
        dropout = self.cfg.model.get("dropout", 0.0)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, self.num_classes),
        )

        return model

    def _setup_metrics(self, prefix: str) -> None:
        """Set up metrics for a given stage (train/val/test)."""
        # Basic metrics
        setattr(
            self,
            f"{prefix}_accuracy",
            Accuracy(task="multiclass", num_classes=self.num_classes, average="macro"),
        )
        setattr(
            self,
            f"{prefix}_accuracy_micro",
            Accuracy(task="multiclass", num_classes=self.num_classes, average="micro"),
        )

        # Per-class accuracy
        setattr(
            self,
            f"{prefix}_accuracy_per_class",
            Accuracy(task="multiclass", num_classes=self.num_classes, average=None),
        )

        # Precision, Recall, F1 (macro)
        setattr(
            self,
            f"{prefix}_precision",
            Precision(task="multiclass", num_classes=self.num_classes, average="macro"),
        )
        setattr(
            self,
            f"{prefix}_recall",
            Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
        )
        setattr(
            self,
            f"{prefix}_f1",
            F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
        )

        # Weighted metrics
        setattr(
            self,
            f"{prefix}_precision_weighted",
            Precision(
                task="multiclass", num_classes=self.num_classes, average="weighted"
            ),
        )
        setattr(
            self,
            f"{prefix}_recall_weighted",
            Recall(task="multiclass", num_classes=self.num_classes, average="weighted"),
        )
        setattr(
            self,
            f"{prefix}_f1_weighted",
            F1Score(
                task="multiclass", num_classes=self.num_classes, average="weighted"
            ),
        )

        # Per-class precision, recall, f1
        setattr(
            self,
            f"{prefix}_precision_per_class",
            Precision(task="multiclass", num_classes=self.num_classes, average=None),
        )
        setattr(
            self,
            f"{prefix}_recall_per_class",
            Recall(task="multiclass", num_classes=self.num_classes, average=None),
        )
        setattr(
            self,
            f"{prefix}_f1_per_class",
            F1Score(task="multiclass", num_classes=self.num_classes, average=None),
        )

        # Specificity
        setattr(
            self,
            f"{prefix}_specificity",
            Specificity(
                task="multiclass", num_classes=self.num_classes, average="macro"
            ),
        )

        # Cohen's Kappa (Quadratic Weighted Kappa for ordinal data)
        setattr(
            self,
            f"{prefix}_qwk",
            CohenKappa(
                task="multiclass", num_classes=self.num_classes, weights="quadratic"
            ),
        )
        setattr(
            self,
            f"{prefix}_kappa_linear",
            CohenKappa(
                task="multiclass", num_classes=self.num_classes, weights="linear"
            ),
        )

        # AUROC (one-vs-rest)
        setattr(
            self,
            f"{prefix}_auroc",
            AUROC(task="multiclass", num_classes=self.num_classes, average="macro"),
        )
        setattr(
            self,
            f"{prefix}_auroc_weighted",
            AUROC(task="multiclass", num_classes=self.num_classes, average="weighted"),
        )

        # Confusion Matrix
        setattr(
            self,
            f"{prefix}_confusion_matrix",
            ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
        )

        # Expected Calibration Error
        setattr(
            self,
            f"{prefix}_ece",
            MulticlassCalibrationError(
                num_classes=self.num_classes, n_bins=10, norm="l1"
            ),
        )

    def _get_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute cross-entropy loss with optional label smoothing and class weights."""
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None

        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        """Handle backbone unfreezing at specified epoch."""
        if self.freeze_backbone_epochs > 0:
            if self.current_epoch < self.freeze_backbone_epochs:
                # Freeze backbone
                for name, param in self.model.named_parameters():
                    if "fc" not in name:
                        param.requires_grad = False
            elif self.current_epoch == self.freeze_backbone_epochs:
                # Unfreeze backbone
                print(f"🔓 Unfreezing backbone at epoch {self.current_epoch}")
                for param in self.model.parameters():
                    param.requires_grad = True

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        images, targets = batch
        logits = self(images)
        loss = self._get_loss(logits, targets)

        # Get predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Update metrics
        self._update_metrics("train", preds, targets, probs)

        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log metrics at end of training epoch."""
        self._log_metrics("train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        images, targets = batch
        logits = self(images)
        loss = self._get_loss(logits, targets)

        # Get predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Update metrics
        self._update_metrics("val", preds, targets, probs)

        # Store for confusion matrix
        self.val_preds.append(preds)
        self.val_targets.append(targets)

        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Log metrics and confusion matrix at end of validation epoch."""
        self._log_metrics("val")
        self._log_confusion_matrix("val", self.val_preds, self.val_targets)

        # Clear stored predictions
        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        images, targets = batch
        logits = self(images)
        loss = self._get_loss(logits, targets)

        # Get predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Update metrics
        self._update_metrics("test", preds, targets, probs)

        # Store for confusion matrix
        self.test_preds.append(preds)
        self.test_targets.append(targets)

        # Log loss
        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Log metrics and confusion matrix at end of test epoch."""
        self._log_metrics("test")
        self._log_confusion_matrix("test", self.test_preds, self.test_targets)

        # Clear stored predictions
        self.test_preds.clear()
        self.test_targets.clear()

    def _update_metrics(
        self,
        prefix: str,
        preds: Tensor,
        targets: Tensor,
        probs: Tensor,
    ) -> None:
        """Update all metrics for a given stage."""
        # Basic metrics
        getattr(self, f"{prefix}_accuracy").update(preds, targets)
        getattr(self, f"{prefix}_accuracy_micro").update(preds, targets)
        getattr(self, f"{prefix}_accuracy_per_class").update(preds, targets)

        # Precision, Recall, F1
        getattr(self, f"{prefix}_precision").update(preds, targets)
        getattr(self, f"{prefix}_recall").update(preds, targets)
        getattr(self, f"{prefix}_f1").update(preds, targets)

        # Weighted metrics
        getattr(self, f"{prefix}_precision_weighted").update(preds, targets)
        getattr(self, f"{prefix}_recall_weighted").update(preds, targets)
        getattr(self, f"{prefix}_f1_weighted").update(preds, targets)

        # Per-class
        getattr(self, f"{prefix}_precision_per_class").update(preds, targets)
        getattr(self, f"{prefix}_recall_per_class").update(preds, targets)
        getattr(self, f"{prefix}_f1_per_class").update(preds, targets)

        # Other metrics
        getattr(self, f"{prefix}_specificity").update(preds, targets)
        getattr(self, f"{prefix}_qwk").update(preds, targets)
        getattr(self, f"{prefix}_kappa_linear").update(preds, targets)
        getattr(self, f"{prefix}_auroc").update(probs, targets)
        getattr(self, f"{prefix}_auroc_weighted").update(probs, targets)
        getattr(self, f"{prefix}_ece").update(probs, targets)

    def _log_metrics(self, prefix: str) -> None:
        """Log all metrics for a given stage."""
        # Basic metrics
        self.log(
            f"{prefix}/accuracy_macro",
            getattr(self, f"{prefix}_accuracy").compute(),
            prog_bar=(prefix == "val"),
        )
        self.log(
            f"{prefix}/accuracy_micro",
            getattr(self, f"{prefix}_accuracy_micro").compute(),
        )

        # Per-class accuracy
        per_class_acc = getattr(self, f"{prefix}_accuracy_per_class").compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f"{prefix}/accuracy_class_{i}", acc)

        # Precision, Recall, F1
        self.log(
            f"{prefix}/precision_macro", getattr(self, f"{prefix}_precision").compute()
        )
        self.log(f"{prefix}/recall_macro", getattr(self, f"{prefix}_recall").compute())
        self.log(f"{prefix}/f1_macro", getattr(self, f"{prefix}_f1").compute())

        # Weighted metrics
        self.log(
            f"{prefix}/precision_weighted",
            getattr(self, f"{prefix}_precision_weighted").compute(),
        )
        self.log(
            f"{prefix}/recall_weighted",
            getattr(self, f"{prefix}_recall_weighted").compute(),
        )
        self.log(
            f"{prefix}/f1_weighted", getattr(self, f"{prefix}_f1_weighted").compute()
        )

        # Per-class precision, recall, f1
        per_class_prec = getattr(self, f"{prefix}_precision_per_class").compute()
        per_class_rec = getattr(self, f"{prefix}_recall_per_class").compute()
        per_class_f1 = getattr(self, f"{prefix}_f1_per_class").compute()
        for i in range(self.num_classes):
            self.log(f"{prefix}/precision_class_{i}", per_class_prec[i])
            self.log(f"{prefix}/recall_class_{i}", per_class_rec[i])
            self.log(f"{prefix}/f1_class_{i}", per_class_f1[i])

        # Other metrics
        self.log(
            f"{prefix}/specificity", getattr(self, f"{prefix}_specificity").compute()
        )

        # QWK - most important for ordinal classification!
        qwk_value = getattr(self, f"{prefix}_qwk").compute()
        self.log(f"{prefix}/qwk", qwk_value, prog_bar=(prefix == "val"))
        self.log(
            f"{prefix}/kappa_linear", getattr(self, f"{prefix}_kappa_linear").compute()
        )

        # AUROC
        self.log(f"{prefix}/auroc_macro", getattr(self, f"{prefix}_auroc").compute())
        self.log(
            f"{prefix}/auroc_weighted",
            getattr(self, f"{prefix}_auroc_weighted").compute(),
        )

        # ECE
        self.log(f"{prefix}/ece", getattr(self, f"{prefix}_ece").compute())

        # Reset all metrics
        self._reset_metrics(prefix)

    def _reset_metrics(self, prefix: str) -> None:
        """Reset all metrics for a given stage."""
        metric_names = [
            "accuracy",
            "accuracy_micro",
            "accuracy_per_class",
            "precision",
            "recall",
            "f1",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "precision_per_class",
            "recall_per_class",
            "f1_per_class",
            "specificity",
            "qwk",
            "kappa_linear",
            "auroc",
            "auroc_weighted",
            "ece",
        ]
        for name in metric_names:
            getattr(self, f"{prefix}_{name}").reset()

    def _log_confusion_matrix(
        self, prefix: str, preds_list: List[Tensor], targets_list: List[Tensor]
    ) -> None:
        """Log confusion matrix to wandb."""
        if not preds_list:
            return

        preds = torch.cat(preds_list)
        targets = torch.cat(targets_list)

        cm = getattr(self, f"{prefix}_confusion_matrix")
        cm.update(preds, targets)
        confusion_matrix = cm.compute().cpu().numpy()
        cm.reset()

        # Log as wandb table or image
        if self.logger and hasattr(self.logger, "experiment"):
            try:
                import wandb

                # Create wandb confusion matrix
                class_names = [f"MES {i}" for i in range(self.num_classes)]
                wandb_cm = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=targets.cpu().numpy(),
                    preds=preds.cpu().numpy(),
                    class_names=class_names,
                )
                self.logger.experiment.log({f"{prefix}/confusion_matrix": wandb_cm})
            except Exception as e:
                print(f"Warning: Could not log confusion matrix to wandb: {e}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        optimizer_name = self.cfg.training.optimizer.lower()
        lr = self.cfg.training.learning_rate
        weight_decay = self.cfg.training.weight_decay

        if optimizer_name == "adam":
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler
        scheduler_name = self.cfg.training.get("scheduler", "none").lower()
        scheduler_params = self.cfg.training.get("scheduler_params", {})

        if scheduler_name == "none":
            return {"optimizer": optimizer}

        if scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get("T_max", self.cfg.training.max_epochs),
                eta_min=scheduler_params.get("eta_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        if scheduler_name == "step":
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_params.get("step_size", 30),
                gamma=scheduler_params.get("gamma", 0.1),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        if scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=scheduler_params.get("factor", 0.1),
                patience=scheduler_params.get("patience", 10),
                min_lr=scheduler_params.get("min_lr", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg.logging.monitor_metric,
                    "interval": "epoch",
                },
            }

        raise ValueError(f"Unknown scheduler: {scheduler_name}")
