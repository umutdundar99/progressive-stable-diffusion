"""
Data pipeline for IP-Adapter training with structure images.

This datamodule returns:
1. Original endoscopy images (for training target)
2. Mayo score labels (for AOE conditioning)
3. Structure images (processed by CLIPImageProcessor for anatomical conditioning)
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from lightning import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from transformers import CLIPImageProcessor


class LIMUCIPDataset(Dataset):
    """
    Dataset returning images, labels, and structure images for IP-Adapter training.

    Returns:
        image: Original endoscopy image (B, 3, H, W) in [-1, 1] for SD
        label: Mayo score as float
        clip_image: CLIP-processed image for anatomical conditioning
    """

    def __init__(
        self,
        root: str,
        pil_augment: Callable,
        config: Dict,
        continuous: bool = True,
        image_size: Optional[int] = None,
        clip_image_processor: Optional[CLIPImageProcessor] = None,
    ) -> None:
        self.continuous = continuous
        self.image_size = image_size or 256
        self.root = root
        self.pil_augment = pil_augment  # Flip, rotation etc. at PIL level

        self.clip_image_processor = clip_image_processor or CLIPImageProcessor()

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        self._load_samples()

    def _load_samples(self) -> None:
        """Load image paths and labels from directory structure."""
        classes = sorted(
            [
                d
                for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d))
            ]
        )

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_dir = os.path.join(self.root, cls_name)
            cls_idx = self.class_to_idx[cls_name]

            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                ):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, float, Tensor]:
        img_path, label = self.samples[idx]
        label_value = float(label) if self.continuous else int(label)
        raw_image = Image.open(img_path).convert("RGB")

        if self.pil_augment:
            raw_image = self.pil_augment(raw_image)

        resized_pil = transforms.Resize((self.image_size, self.image_size))(raw_image)
        raw_tensor = transforms.ToTensor()(resized_pil)
        image = transforms.Normalize([0.5] * 3, [0.5] * 3)(raw_tensor)

        clip_image = self.clip_image_processor(
            images=raw_tensor.permute(1, 2, 0).cpu().numpy(),
            return_tensors="pt",
            do_rescale=False,
        ).pixel_values.squeeze(0)

        return image, label_value, clip_image

    @property
    def class_counts(self) -> List[int]:
        """Return number of samples per class for balanced sampling."""
        counts: Dict[int, int] = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return [counts[i] for i in sorted(counts.keys())]


class OrdinalIPDataModule(LightningDataModule):
    """DataModule for IP-Adapter training with structure images."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset_path = cfg.dataset.dataset_path
        self.image_size = cfg.dataset.image_size
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.augmentation = cfg.dataset.augmentation
        self.sampler = cfg.dataset.sampler

        clip_model_path = getattr(
            cfg.model, "image_encoder_path", "openai/clip-vit-large-patch14"
        )
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_model_path)

        self.persistent_workers = getattr(cfg.dataset, "persistent_workers", True)
        self.prefetch_factor = getattr(cfg.dataset, "prefetch_factor", 2)

        self._train_dataset: Optional[LIMUCIPDataset] = None
        self._val_dataset: Optional[LIMUCIPDataset] = None

    def _build_pil_augment(self, train: bool) -> Optional[transforms.Compose]:
        """Build PIL-level augmentations (applied before both SD and CLIP transforms)."""
        if not train:
            return None

        ops = []

        # Center crop at PIL level
        if self.augmentation.get("center_crop"):
            ops.append(transforms.CenterCrop(self.augmentation["center_crop"]))

        if self.augmentation.get("flip", False):
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if self.augmentation.get("rotation", 0) > 0:
            ops.append(transforms.RandomRotation(self.augmentation["rotation"]))

        if self.augmentation.get("perspective", 0) > 0:
            ops.append(
                transforms.RandomPerspective(
                    distortion_scale=self.augmentation["perspective"], p=0.3
                )
            )

        return transforms.Compose(ops) if ops else None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_dataset = LIMUCIPDataset(
                os.path.join(self.dataset_path, "train"),
                pil_augment=self._build_pil_augment(train=True),
                clip_image_processor=self.clip_image_processor,
                image_size=self.image_size,
                config=self.cfg,
            )

    def _build_sampler(
        self, dataset: LIMUCIPDataset
    ) -> Optional[WeightedRandomSampler]:
        if self.sampler != "class_balanced":
            return None

        counts = torch.tensor(dataset.class_counts, dtype=torch.float32)
        weights = 1.0 / (counts + 1e-8)

        all_labels = torch.tensor(
            [label for _, label in dataset.samples], dtype=torch.long
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
            drop_last=True,
        )
