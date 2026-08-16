"""
EMNIST dataset loader.

EMNIST Balanced: 131,600 samples across 47 classes
(digits 0-9, uppercase A-Z, 11 merged lowercase letters).

This is the PRIMARY training dataset for the character classifier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.training.config import Config, EMNIST_BALANCED_LABELS, config as default_config

logger = logging.getLogger(__name__)

# EMNIST class counts per split (for reference)
EMNIST_SPLITS_INFO: Dict[str, Dict[str, int]] = {
    "balanced":  {"classes": 47,  "train": 112800, "test": 18800},
    "byclass":   {"classes": 62,  "train": 697932, "test": 116323},
    "bymerge":   {"classes": 47,  "train": 697932, "test": 116323},
    "digits":    {"classes": 10,  "train": 240000, "test": 40000},
    "letters":   {"classes": 26,  "train": 88800,  "test": 14800},
    "mnist":     {"classes": 10,  "train": 60000,  "test": 10000},
}


class RotateTransform:
    """
    Pickle-safe top-level transform class for rotating images.
    Replaces transforms.Lambda with an explicit callable class so that
    multiprocessing DataLoaders can serialize transforms without AttributeError under Python 3.13+.
    """
    def __init__(self, angle: int = -90) -> None:
        self.angle = angle

    def __call__(self, img):
        return transforms.functional.rotate(img, self.angle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angle={self.angle})"


def get_emnist_transforms(cfg: Config, augment: bool = False) -> transforms.Compose:
    """
    Build EMNIST transform pipeline.

    NOTE: EMNIST images are transposed relative to standard orientation —
    torchvision applies a 90° rotation + horizontal flip automatically when
    split='balanced', but we still explicitly correct it here for robustness.

    Args:
        cfg:     Project configuration.
        augment: Apply training-time augmentations.

    Returns:
        Composed transform pipeline.
    """
    pc = cfg.preprocessing
    transform_list: list = []

    # EMNIST orientation fix (images come rotated/flipped from the dataset; pickle-safe class used)
    transform_list.append(RotateTransform(angle=-90))
    transform_list.append(transforms.RandomHorizontalFlip(p=1.0))  # always flip back

    # Resize to target
    transform_list.append(transforms.Resize(pc.image_size))

    # Augmentations for training
    if augment:
        transform_list.extend([
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.85, 1.15),
                shear=5,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ])

    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(pc.normalize_mean, pc.normalize_std))

    return transforms.Compose(transform_list)


def load_emnist(
    cfg: Config = default_config,
    split: str | None = None,
    val_split: float | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, Dict[int, str]]:
    """
    Load EMNIST and return train / val / test DataLoaders.

    Args:
        cfg:        Project configuration.
        split:      EMNIST split override (defaults to cfg.dataset.emnist_split).
        val_split:  Override validation fraction.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes, label_map).
    """
    root = Path(cfg.dataset.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    emnist_split = split or cfg.dataset.emnist_split
    val_frac = val_split if val_split is not None else cfg.dataset.val_split
    tc = cfg.training

    split_info = EMNIST_SPLITS_INFO.get(emnist_split, {})
    num_classes = split_info.get("classes", 47)

    logger.info(
        "Loading EMNIST split='%s' (%d classes) from %s ...",
        emnist_split, num_classes, root,
    )

    # Training set with augmentation
    full_train = datasets.EMNIST(
        root=str(root),
        split=emnist_split,
        train=True,
        download=cfg.dataset.auto_download,
        transform=get_emnist_transforms(cfg, augment=True),
    )

    # Test set without augmentation
    test_set = datasets.EMNIST(
        root=str(root),
        split=emnist_split,
        train=False,
        download=cfg.dataset.auto_download,
        transform=get_emnist_transforms(cfg, augment=False),
    )

    # Split into train + val
    n_total = len(full_train)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.dataset.seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    logger.info(
        "EMNIST splits — train: %d | val: %d | test: %d",
        len(train_set), len(val_set), len(test_set),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
    )

    # Return the appropriate label map
    label_map = EMNIST_BALANCED_LABELS if emnist_split == "balanced" else {}

    return train_loader, val_loader, test_loader, num_classes, label_map
