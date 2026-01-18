"""
MES Classification Module.

This module provides ResNet-based classification for Mayo Endoscopic Score (MES)
prediction on endoscopy images.
"""

from src.classification.dataset import MESClassificationDataset, MESDataModule
from src.classification.model import ResNetClassifier

__all__ = [
    "MESClassificationDataset",
    "MESDataModule",
    "ResNetClassifier",
]
