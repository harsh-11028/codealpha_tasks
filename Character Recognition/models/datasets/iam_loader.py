"""
IAM Handwriting Dataset loader.

The IAM dataset contains segmented handwritten text (forms, lines, words, characters).
It requires registration at: https://fki.ifs.unibe.ch/databases/iam-handwriting-database

This loader supports:
  1. Automatic download using credentials from .env (IAM_USERNAME / IAM_PASSWORD)
  2. Manual download — place the dataset under datasets/raw/iam/ and run this loader.
  3. Mock dataset fallback — generates synthetic samples for testing purposes.

Dataset structure expected (after manual download):
    datasets/raw/iam/
        words/          (word-level PNG images)
        lines/          (line-level PNG images)
        words.txt       (ground truth annotations)
        lines.txt
"""

from __future__ import annotations

import io
import logging
import os
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from models.training.config import Config, config as default_config

logger = logging.getLogger(__name__)

IAM_BASE_URL = "https://fki.ifs.unibe.ch/DBs/iamDB"
IAM_WORDS_URL = f"{IAM_BASE_URL}/data/words.tgz"
IAM_LINES_URL = f"{IAM_BASE_URL}/data/lines.tgz"
IAM_GT_WORDS_URL = f"{IAM_BASE_URL}/data/ascii/words.txt"


# ---------------------------------------------------------------------------
# Ground truth parser
# ---------------------------------------------------------------------------

class IAMAnnotation:
    """Parsed ground truth annotation for a single IAM word."""

    __slots__ = ("word_id", "status", "gray_level", "x", "y", "w", "h", "tag", "text")

    def __init__(
        self,
        word_id: str,
        status: str,
        gray_level: int,
        x: int, y: int, w: int, h: int,
        tag: str,
        text: str,
    ) -> None:
        self.word_id = word_id
        self.status = status
        self.gray_level = gray_level
        self.x, self.y, self.w, self.h = x, y, w, h
        self.tag = tag
        self.text = text

    @classmethod
    def from_line(cls, line: str) -> Optional["IAMAnnotation"]:
        """Parse a single line from words.txt."""
        if line.startswith("#") or not line.strip():
            return None
        parts = line.split()
        if len(parts) < 9:
            return None
        try:
            return cls(
                word_id=parts[0],
                status=parts[1],
                gray_level=int(parts[2]),
                x=int(parts[3]),
                y=int(parts[4]),
                w=int(parts[5]),
                h=int(parts[6]),
                tag=parts[7],
                text=" ".join(parts[8:]),
            )
        except (ValueError, IndexError):
            return None

    def image_path(self, root: Path) -> Path:
        """Derive the image path from word_id convention: a01-000u-00-00."""
        parts = self.word_id.split("-")
        form_folder = parts[0]            # e.g. a01
        form_subfolder = f"{parts[0]}-{parts[1]}"  # e.g. a01-000u
        filename = f"{self.word_id}.png"
        return root / "words" / form_folder / form_subfolder / filename


def parse_iam_gt(gt_file: Path) -> List[IAMAnnotation]:
    """Parse the IAM words.txt ground truth file."""
    annotations: List[IAMAnnotation] = []
    with open(gt_file, encoding="utf-8") as f:
        for line in f:
            ann = IAMAnnotation.from_line(line.rstrip())
            if ann is not None and ann.status == "ok":
                annotations.append(ann)
    logger.info("Parsed %d valid IAM annotations from %s", len(annotations), gt_file)
    return annotations


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class IAMWordDataset(Dataset):
    """
    PyTorch Dataset for IAM word-level handwriting.

    Each sample is (image_tensor, text_label).
    The text label is a string of characters (for CTC training).
    """

    def __init__(
        self,
        annotations: List[IAMAnnotation],
        root: Path,
        transform: Optional[transforms.Compose] = None,
        charset: str = string.digits + string.ascii_letters,
    ) -> None:
        self.annotations = annotations
        self.root = root
        self.transform = transform
        self.charset = charset
        self.char_to_idx: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(charset)}
        self.idx_to_char: Dict[int, str] = {v: k for k, v in self.char_to_idx.items()}
        # Filter annotations to only those with existing image files
        self.annotations = [
            a for a in annotations
            if a.image_path(root).exists()
        ]
        logger.debug("IAMWordDataset: %d samples after filtering missing files.", len(self.annotations))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        ann = self.annotations[idx]
        img_path = ann.image_path(self.root)

        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)

        # Encode text as integer sequence for CTC
        encoded = torch.tensor(
            [self.char_to_idx.get(ch, 0) for ch in ann.text if ch in self.char_to_idx],
            dtype=torch.long,
        )

        return image, encoded, len(ann.text)

    def decode(self, indices: List[int]) -> str:
        """Decode CTC output indices back to string."""
        return "".join(self.idx_to_char.get(i, "") for i in indices if i > 0)


# ---------------------------------------------------------------------------
# Mock dataset for testing without IAM credentials
# ---------------------------------------------------------------------------

class MockIAMDataset(Dataset):
    """
    Synthetic word-image dataset that mimics IAM format.

    Generates PIL images of rendered text using a monospace font.
    Used for testing the pipeline when IAM is unavailable.
    """

    def __init__(
        self,
        size: int = 2000,
        image_size: Tuple[int, int] = (32, 128),
        transform: Optional[transforms.Compose] = None,
        charset: str = string.digits + string.ascii_letters,
        seed: int = 42,
    ) -> None:
        self.size = size
        self.image_size = image_size  # (H, W)
        self.transform = transform
        self.charset = charset
        self.char_to_idx: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(charset)}
        self.idx_to_char: Dict[int, str] = {v: k for k, v in self.char_to_idx.items()}

        rng = random.Random(seed)
        self.samples: List[str] = [
            "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
            for _ in range(size)
        ]

        logger.info("MockIAMDataset: generated %d synthetic word samples.", size)

    def _render_word(self, word: str) -> Image.Image:
        h, w = self.image_size
        img = Image.new("L", (w, h), color=255)
        draw = ImageDraw.Draw(img)
        # Use default PIL font (no external font needed)
        draw.text((4, 4), word, fill=0)
        return img

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        text = self.samples[idx]
        image = self._render_word(text)
        if self.transform:
            image = self.transform(image)
        encoded = torch.tensor(
            [self.char_to_idx.get(ch, 0) for ch in text if ch in self.char_to_idx],
            dtype=torch.long,
        )
        return image, encoded, len(text)


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def _download_iam(root: Path, username: str, password: str) -> bool:
    """
    Attempt to download the IAM dataset using provided credentials.
    Returns True on success, False on failure.
    """
    import tarfile

    session = requests.Session()
    session.auth = (username, password)

    for url, archive_name in [
        (IAM_WORDS_URL, "words.tgz"),
        (IAM_GT_WORDS_URL, "words.txt"),
    ]:
        dest = root / archive_name
        if dest.exists():
            logger.info("Skipping %s — already downloaded.", archive_name)
            continue

        logger.info("Downloading IAM file: %s", url)
        try:
            resp = session.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as exc:
            logger.error("Failed to download %s: %s", url, exc)
            return False

    # Extract archives
    for archive_name in ["words.tgz"]:
        archive = root / archive_name
        if archive.exists() and archive.suffix in (".tgz", ".gz"):
            logger.info("Extracting %s ...", archive_name)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(root)

    return True


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def get_word_transforms(cfg: Config, augment: bool = False) -> transforms.Compose:
    """Transform pipeline for word-level images (wider aspect ratio)."""
    pc = cfg.preprocessing
    transform_list: list = []
    transform_list.append(transforms.Resize(pc.word_image_size))
    if augment:
        transform_list.extend([
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(pc.normalize_mean, pc.normalize_std))
    return transforms.Compose(transform_list)


def load_iam(
    cfg: Config = default_config,
    use_mock: bool = False,
    val_split: float | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load IAM dataset and return train / val / test DataLoaders.

    Falls back to mock dataset if:
      - use_mock=True
      - IAM dataset not found and no credentials in env

    Args:
        cfg:        Project configuration.
        use_mock:   Force use of mock synthetic dataset.
        val_split:  Override validation fraction.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root = Path(cfg.dataset.root_dir) / "iam"
    root.mkdir(parents=True, exist_ok=True)

    val_frac = val_split if val_split is not None else cfg.dataset.val_split
    tc = cfg.training

    train_transform = get_word_transforms(cfg, augment=True)
    eval_transform = get_word_transforms(cfg, augment=False)

    gt_file = root / "words.txt"
    iam_available = gt_file.exists() and (root / "words").exists()

    if not iam_available and not use_mock:
        username = os.getenv("IAM_USERNAME", "")
        password = os.getenv("IAM_PASSWORD", "")
        if username and password:
            iam_available = _download_iam(root, username, password)

    if use_mock or not iam_available:
        logger.warning(
            "IAM dataset not available — using MockIAMDataset. "
            "Set IAM_USERNAME and IAM_PASSWORD in .env for the real dataset."
        )
        full_train = MockIAMDataset(
            size=2000,
            image_size=cfg.preprocessing.word_image_size,
            transform=train_transform,
            seed=cfg.dataset.seed,
        )
        test_dataset = MockIAMDataset(
            size=400,
            image_size=cfg.preprocessing.word_image_size,
            transform=eval_transform,
            seed=cfg.dataset.seed + 1,
        )
    else:
        annotations = parse_iam_gt(gt_file)
        n_test = int(len(annotations) * cfg.dataset.test_split)
        generator = random.Random(cfg.dataset.seed)
        generator.shuffle(annotations)
        test_annotations = annotations[:n_test]
        train_annotations = annotations[n_test:]

        full_train = IAMWordDataset(train_annotations, root, transform=train_transform)
        test_dataset = IAMWordDataset(test_annotations, root, transform=eval_transform)

    n_total = len(full_train)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val

    generator_torch = torch.Generator().manual_seed(cfg.dataset.seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator_torch)

    logger.info(
        "IAM splits — train: %d | val: %d | test: %d",
        len(train_set), len(val_set), len(test_dataset),
    )

    def collate_fn(batch):
        """Custom collate to handle variable-length CTC targets."""
        images, targets, lengths = zip(*batch)
        images = torch.stack(images, dim=0)
        targets = torch.cat(targets, dim=0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        return images, targets, lengths

    train_loader = DataLoader(
        train_set,
        batch_size=tc.batch_size,
        shuffle=True,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tc.batch_size,
        shuffle=False,
        num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
