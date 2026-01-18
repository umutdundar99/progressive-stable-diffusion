"""
Dataset and DataModule for MES Classification.

Provides PyTorch Dataset and Lightning DataModule for training
ResNet classifier on LIMUC endoscopy images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MESClassificationDataset(Dataset):
    """
    Dataset for MES (Mayo Endoscopic Score) classification.

    Expects folder structure:
        data_root/
            train/
                0/  # MES 0 images
                1/  # MES 1 images
                2/  # MES 2 images
                3/  # MES 3 images
            val/
                ...
            test/
                ...
    """

    VALID_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            data_root: Root directory containing train/val/test folders
            split: One of 'train', 'val', 'test'
            transform: Optional transforms to apply to images
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        self.split_dir = self.data_root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        self._load_samples()

    def _load_samples(self) -> None:
        """Load all image paths and their labels."""
        for class_idx in range(4):
            class_dir = self.split_dir / str(class_idx)
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_path, class_idx))
                    self.class_counts[class_idx] += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.split_dir}")

        print(f"[{self.split}] Loaded {len(self.samples)} images")
        print(f"[{self.split}] Class distribution: {self.class_counts}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> Tensor:
        """Compute inverse frequency class weights for balanced training."""
        total = sum(self.class_counts.values())
        weights = []
        for i in range(4):
            count = self.class_counts[i]
            if count > 0:
                weights.append(total / (4 * count))
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return [class_weights[label].item() for _, label in self.samples]


class MESDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MES classification.

    Handles data loading, transforms, and sampling for training,
    validation, and testing.
    """

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation_cfg: Optional[Dict[str, Any]] = None,
        use_weighted_sampler: bool = True,
    ) -> None:
        """
        Initialize the DataModule.

        Args:
            data_root: Root directory containing train/val/test folders
            image_size: Target image size
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            augmentation_cfg: Augmentation configuration from config file
            use_weighted_sampler: Use weighted random sampler for class balance
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_root = Path(data_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augmentation_cfg = augmentation_cfg or {}
        self.use_weighted_sampler = use_weighted_sampler

        self.train_dataset: MESClassificationDataset
        self.val_dataset: MESClassificationDataset
        self.test_dataset: MESClassificationDataset

    def _build_train_transforms(self) -> transforms.Compose:
        """Build augmentation pipeline for training."""
        train_cfg = self.augmentation_cfg.get("train", {})

        transform_list = []

        # Random resized crop
        if "random_resized_crop" in train_cfg:
            rrc_cfg = train_cfg["random_resized_crop"]
            transform_list.append(
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=tuple(rrc_cfg.get("scale", [0.8, 1.0])),
                    ratio=tuple(rrc_cfg.get("ratio", [0.9, 1.1])),
                )
            )
        else:
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))

        # Random horizontal flip
        if "random_horizontal_flip" in train_cfg:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=train_cfg["random_horizontal_flip"])
            )

        # Random vertical flip
        if "random_vertical_flip" in train_cfg:
            transform_list.append(
                transforms.RandomVerticalFlip(p=train_cfg["random_vertical_flip"])
            )

        # Random rotation with reflect fill to avoid black borders
        if "random_rotation" in train_cfg:
            transform_list.append(
                transforms.RandomRotation(
                    degrees=train_cfg["random_rotation"],
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,  # Will be replaced by custom padding below
                )
            )

        # Random affine with fill to reduce black borders
        if "random_affine" in train_cfg:
            affine_cfg = train_cfg["random_affine"]
            transform_list.append(
                transforms.RandomAffine(
                    degrees=affine_cfg.get("degrees", 0),
                    translate=tuple(affine_cfg.get("translate", [0, 0])),
                    scale=tuple(affine_cfg.get("scale", [1.0, 1.0])),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,  # Black fill, but we'll use a wrapper for reflect
                )
            )

        # Color jitter
        if "color_jitter" in train_cfg:
            cj_cfg = train_cfg["color_jitter"]
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj_cfg.get("brightness", 0),
                    contrast=cj_cfg.get("contrast", 0),
                    saturation=cj_cfg.get("saturation", 0),
                    hue=cj_cfg.get("hue", 0),
                )
            )

        # Gaussian blur
        if "gaussian_blur" in train_cfg:
            blur_cfg = train_cfg["gaussian_blur"]
            transform_list.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=blur_cfg.get("kernel_size", 5),
                            sigma=tuple(blur_cfg.get("sigma", [0.1, 2.0])),
                        )
                    ],
                    p=blur_cfg.get("probability", 0.5),
                )
            )

        # To tensor and normalize
        transform_list.append(transforms.ToTensor())
        transform_list.append(
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        )

        # Random erasing (after ToTensor)
        if "random_erasing" in train_cfg:
            transform_list.append(
                transforms.RandomErasing(p=train_cfg["random_erasing"])
            )

        return transforms.Compose(transform_list)

    def _build_eval_transforms(self) -> transforms.Compose:
        """Build transforms for validation/test."""
        eval_cfg = self.augmentation_cfg.get("eval", {})

        transform_list = []

        # Resize
        resize_size = eval_cfg.get("resize", 256)
        transform_list.append(transforms.Resize(resize_size))

        # Center crop
        crop_size = eval_cfg.get("center_crop", self.image_size)
        transform_list.append(transforms.CenterCrop(crop_size))

        # To tensor and normalize
        transform_list.append(transforms.ToTensor())
        transform_list.append(
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        )

        return transforms.Compose(transform_list)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = MESClassificationDataset(
                data_root=self.data_root,
                split="train",
                transform=self._build_train_transforms(),
            )
            self.val_dataset = MESClassificationDataset(
                data_root=self.data_root,
                split="val",
                transform=self._build_eval_transforms(),
            )

        if stage == "test" or stage is None:
            self.test_dataset = MESClassificationDataset(
                data_root=self.data_root,
                split="test",
                transform=self._build_eval_transforms(),
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with optional weighted sampling."""
        shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_class_weights(self) -> Optional[Tensor]:
        """Get class weights from training dataset."""
        if self.train_dataset is not None:
            return self.train_dataset.get_class_weights()
        return None
