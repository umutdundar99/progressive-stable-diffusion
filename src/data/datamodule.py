"""
Data pipeline for LIMUC ordinal disease dataset.

"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms


class LIMUCDataset(Dataset):
    """
    ImageFolder-based dataset returning images and continuous MES labels.

    Folder structure:
        root/
            0/
            1/
            2/
            3/

    Labels are mapped automatically by ImageFolder:
        folder name -> class index (0,1,2,3)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        continuous: bool = True,
    ) -> None:
        self.continuous = continuous

        self.dataset = datasets.ImageFolder(
            root=root,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, float]:
        image, label = self.dataset[idx]
        label_value = float(label) if self.continuous else int(label)
        return image, label_value

    @property
    def class_counts(self) -> List[int]:
        """Return number of samples per class for balanced sampling."""
        counts: Dict[int, int] = {}
        for _, label in self.dataset.samples:
            counts[label] = counts.get(label, 0) + 1
        return [counts[i] for i in sorted(counts.keys())]


class OrdinalDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset_path = cfg.dataset.dataset_path

        self.image_size = cfg.dataset.image_size
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.augmentation = cfg.dataset.augmentation
        self.sampler = cfg.dataset.sampler

        # Advanced data loading options
        self.persistent_workers = getattr(cfg.dataset, "persistent_workers", True)
        self.prefetch_factor = getattr(cfg.dataset, "prefetch_factor", 2)

        self._train_dataset: Optional[LIMUCDataset] = None
        self._val_dataset: Optional[LIMUCDataset] = None

    def _build_transforms(self, train: bool) -> transforms.Compose:
        ops = []

        ops.append(transforms.CenterCrop(self.augmentation["center_crop"]))
        ops.append(transforms.Resize((self.image_size, self.image_size)))

        if train:
            if self.augmentation.get("flip", False):
                ops.append(transforms.RandomHorizontalFlip(p=0.5))
            if self.augmentation.get("rotation", 0) > 0:
                ops.append(transforms.RandomRotation(self.augmentation["rotation"]))
            if self.augmentation.get("color_jitter", False):
                ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))

        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

        return transforms.Compose(ops)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_dataset = LIMUCDataset(
                os.path.join(self.dataset_path, "train"),
                transform=self._build_transforms(train=True),
            )
            self._val_dataset = LIMUCDataset(
                os.path.join(self.dataset_path, "val"),
                transform=self._build_transforms(train=False),
            )
            self._test_dataset = LIMUCDataset(
                os.path.join(self.dataset_path, "test"),
                transform=self._build_transforms(train=False),
            )

    def _build_sampler(self, dataset: LIMUCDataset) -> Optional[WeightedRandomSampler]:
        if self.sampler != "class_balanced":
            return None

        counts = torch.tensor(dataset.class_counts, dtype=torch.float32)  # [4]
        weights = 1.0 / (counts + 1e-8)

        all_labels = torch.tensor(
            [label for _, label in dataset.dataset.samples], dtype=torch.long
        )
        sample_weights = weights[all_labels]

        return WeightedRandomSampler(
            sample_weights.tolist(), num_samples=len(sample_weights), replacement=True
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self.setup()

        assert self._train_dataset is not None
        sampler = self._build_sampler(self._train_dataset)

        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,  # Avoid issues with batch norm on small last batches
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            self.setup()

        assert self._val_dataset is not None

        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_dataset is None:
            self.setup()

        assert self._test_dataset is not None

        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
