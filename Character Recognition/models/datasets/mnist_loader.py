"""
MNIST dataset loader.

Downloads MNIST automatically via torchvision and returns
standardized PyTorch DataLoaders.

Label space: 0–9 (10 classes)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models.training.config import Config, config as default_config

logger = logging.getLogger(__name__)


def get_mnist_transforms(cfg: Config, augment: bool = False) -> transforms.Compose:
    """
    Build the transform pipeline for MNIST images.

    Args:
        cfg:      Project configuration.
        augment:  If True, apply training-time augmentations.

    Returns:
        Composed transform pipeline.
    """
    pc = cfg.preprocessing
    transform_list: list = []

    # Resize
    transform_list.append(transforms.Resize(pc.image_size))

    # Augmentations (training only)
    if augment:
        transform_list.extend([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ])

    # Tensor conversion + normalization
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(pc.normalize_mean, pc.normalize_std))

    return transforms.Compose(transform_list)


def load_mnist(
    cfg: Config = default_config,
    val_split: float | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load MNIST and return train / val / test DataLoaders.

    Args:
        cfg:        Project configuration.
        val_split:  Override validation fraction (defaults to cfg.dataset.val_split).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes).
    """
    root = Path(cfg.dataset.root_dir)
    root.mkdir(parents=True, exist_ok=True)

    val_frac = val_split if val_split is not None else cfg.dataset.val_split
    tc = cfg.training

    logger.info("Loading MNIST dataset from %s ...", root)

    # Download full training set
    full_train = datasets.MNIST(
        root=str(root),
        train=True,
        download=cfg.dataset.auto_download,
        transform=get_mnist_transforms(cfg, augment=True),
    )

    # Test set (no augmentation)
    test_set = datasets.MNIST(
        root=str(root),
        train=False,
        download=cfg.dataset.auto_download,
        transform=get_mnist_transforms(cfg, augment=False),
    )

    # Split training into train + val
    n_total = len(full_train)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.dataset.seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

    # Val set should not use augmentation — wrap with clean transform
    val_set.dataset.transform = get_mnist_transforms(cfg, augment=False)  # type: ignore[attr-defined]

    logger.info(
        "MNIST splits — train: %d | val: %d | test: %d",
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

    num_classes = 10
    return train_loader, val_loader, test_loader, num_classes
