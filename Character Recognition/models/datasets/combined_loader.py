"""
Combined dataset loader.

Merges MNIST, EMNIST, and IAM into a unified training set,
normalizing all label spaces to a common alphabet.

Label normalization strategy:
  - MNIST:  0–9   → unified class indices 0–9
  - EMNIST: 0–46  → unified class indices (already 0-based, includes digits)
  - IAM:    CTC character sequences → handled separately (word-level)

For character-level classification the unified label space uses
EMNIST Balanced's 47-class scheme as the canonical mapping.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from models.datasets.mnist_loader import load_mnist, get_mnist_transforms
from models.datasets.emnist_loader import load_emnist, get_emnist_transforms
from models.datasets.iam_loader import load_iam
from models.training.config import Config, EMNIST_BALANCED_LABELS, config as default_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label remapping wrapper
# ---------------------------------------------------------------------------

class LabelRemappedDataset(Dataset):
    """
    Wraps a classification dataset and remaps integer labels.

    Used to align MNIST labels (0–9) into the EMNIST Balanced
    label space, where digits occupy indices 0–9 (same mapping,
    so this is effectively a no-op for MNIST, but kept for clarity).
    """

    def __init__(self, dataset: Dataset, remap: Optional[Dict[int, int]] = None) -> None:
        self.dataset = dataset
        self.remap = remap or {}

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        if isinstance(label, torch.Tensor):
            label_val = label.item()
        else:
            label_val = int(label)
        remapped = self.remap.get(label_val, label_val)
        return image, torch.tensor(remapped, dtype=torch.long)


# ---------------------------------------------------------------------------
# CombinedDatasetInfo — returned alongside loaders for introspection
# ---------------------------------------------------------------------------

class CombinedDatasetInfo:
    """Metadata about the combined dataset."""

    def __init__(
        self,
        num_classes: int,
        label_map: Dict[int, str],
        sources: List[str],
        class_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.label_map = label_map
        self.sources = sources
        self.class_counts = class_counts or {}

    def __repr__(self) -> str:
        return (
            f"CombinedDatasetInfo("
            f"num_classes={self.num_classes}, "
            f"sources={self.sources}"
            f")"
        )


# ---------------------------------------------------------------------------
# Class weight computation (for handling class imbalance)
# ---------------------------------------------------------------------------

def compute_class_weights(
    dataset: Dataset,
    num_classes: int,
    max_samples: int = 50_000,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted random sampling
    or loss weighting.

    Args:
        dataset:     Classification dataset (returns image, label).
        num_classes: Total number of classes.
        max_samples: Cap samples scanned for speed (random subset if larger).

    Returns:
        Float tensor of shape (num_classes,) with inverse-frequency weights.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    n = min(len(dataset), max_samples)  # type: ignore[arg-type]

    indices = torch.randperm(len(dataset))[:n]  # type: ignore[arg-type]
    for idx in indices.tolist():
        _, label = dataset[idx]
        label_val = label.item() if isinstance(label, torch.Tensor) else int(label)
        if 0 <= label_val < num_classes:
            counts[label_val] += 1

    counts = torch.clamp(counts, min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights


# ---------------------------------------------------------------------------
# Main combined loader
# ---------------------------------------------------------------------------

def load_combined(
    cfg: Config = default_config,
    for_word_model: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, CombinedDatasetInfo]:
    """
    Load and combine datasets according to cfg.dataset flags.

    Args:
        cfg:            Project configuration.
        for_word_model: If True, returns IAM word-level loaders instead
                        of combined character-level loaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_info).
    """
    tc = cfg.training
    sources: List[str] = []

    # ------------------------------------------------------------------
    # Word-level mode: only IAM
    # ------------------------------------------------------------------
    if for_word_model:
        logger.info("Combined loader: word-level mode (IAM only).")
        train_loader, val_loader, test_loader = load_iam(cfg)
        info = CombinedDatasetInfo(
            num_classes=len(set("".join(EMNIST_BALANCED_LABELS.values()))),
            label_map=EMNIST_BALANCED_LABELS,
            sources=["iam"],
        )
        return train_loader, val_loader, test_loader, info

    # ------------------------------------------------------------------
    # Character-level mode: MNIST + EMNIST (+ optionally IAM chars)
    # ------------------------------------------------------------------
    train_datasets: List[Dataset] = []
    val_datasets: List[Dataset] = []
    test_datasets: List[Dataset] = []

    # --- EMNIST (primary) ---
    if cfg.dataset.use_emnist:
        logger.info("Loading EMNIST for combined dataset ...")
        e_train_loader, e_val_loader, e_test_loader, num_classes, label_map = load_emnist(cfg)
        train_datasets.append(e_train_loader.dataset)
        val_datasets.append(e_val_loader.dataset)
        test_datasets.append(e_test_loader.dataset)
        sources.append("emnist")
    else:
        num_classes = 10
        label_map = {i: str(i) for i in range(10)}

    # --- MNIST (supplement digits for extra robustness) ---
    if cfg.dataset.use_mnist:
        logger.info("Loading MNIST for combined dataset ...")
        m_train_loader, m_val_loader, m_test_loader, _ = load_mnist(cfg)
        # Remap MNIST labels — digits 0–9 map identically in EMNIST Balanced
        remap = {i: i for i in range(10)}
        train_datasets.append(LabelRemappedDataset(m_train_loader.dataset, remap))
        val_datasets.append(LabelRemappedDataset(m_val_loader.dataset, remap))
        test_datasets.append(LabelRemappedDataset(m_test_loader.dataset, remap))
        sources.append("mnist")

    # --- Combine ---
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    combined_test = ConcatDataset(test_datasets)

    logger.info(
        "Combined dataset sizes — train: %d | val: %d | test: %d | classes: %d",
        len(combined_train), len(combined_val), len(combined_test), num_classes,
    )

    # --- Compute class weights for loss weighting ---
    logger.info("Computing class weights ...")
    class_weights = compute_class_weights(combined_train, num_classes)
    logger.info("Class weights computed (min=%.4f, max=%.4f).", class_weights.min(), class_weights.max())

    train_loader = DataLoader(
        combined_train,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
    )
    test_loader = DataLoader(
        combined_test,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
    )

    info = CombinedDatasetInfo(
        num_classes=num_classes,
        label_map=label_map,
        sources=sources,
    )

    return train_loader, val_loader, test_loader, info
