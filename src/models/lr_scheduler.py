"""
Lightweight learning rate scheduler to avoid pl_bolts dependency.
"""

from __future__ import annotations

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs.
        max_epochs: Total number of epochs.
        warmup_start_lr: Starting learning rate for warmup.
        eta_min: Minimum learning rate after cosine decay.
        last_epoch: The index of the last epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.max_epochs = max(1, int(max_epochs))
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        current_epoch = self.last_epoch
        base_lrs = self.base_lrs

        if self.warmup_epochs > 0 and current_epoch < self.warmup_epochs:
            warmup_progress = current_epoch / float(self.warmup_epochs)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_progress
                for base_lr in base_lrs
            ]

        cosine_epoch = current_epoch - self.warmup_epochs
        cosine_total = max(1, self.max_epochs - self.warmup_epochs)
        cosine_progress = min(cosine_epoch / float(cosine_total), 1.0)

        return [
            self.eta_min
            + (base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            for base_lr in base_lrs
        ]
