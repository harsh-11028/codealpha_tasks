"""
models/handwriting/iam_hf_loader.py

IAM Handwriting Dataset loader via Hugging Face `datasets` library.
Uses the publicly available "Teklia/IAM-line" dataset — NO credentials needed.

Each sample:
    image  : PIL Image (grayscale handwritten text line)
    text   : str (ground-truth transcription)

Writer-independent split:
    Train  : 6,482 lines
    Val    :   976 lines
    Test   :   970 lines

Usage:
    from models.handwriting.iam_hf_loader import load_iam_hf
    train_ds, val_ds, test_ds = load_iam_hf()
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_iam_hf(
    cache_dir: Optional[str] = None,
) -> Tuple:
    """
    Download / load IAM line-level dataset from Hugging Face.

    Tries multiple sources in order:
      1. Teklia/IAM-line  (writer-independent official splits)
      2. fhswf/IAM        (alternative mirror)

    Returns:
        (train_dataset, val_dataset, test_dataset) as HuggingFace Dataset objects.
        Each item has keys: 'image' (PIL Image) and 'text' (str transcription).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` package is required. "
            "Install with: pip install datasets"
        )

    sources = [
        ("Teklia/IAM-line", None),         # primary — writer-independent splits
        ("fhswf/IAM", "line"),             # fallback with config
    ]

    last_error = None
    for ds_name, config_name in sources:
        try:
            logger.info("Attempting to load dataset: %s (config=%s)...", ds_name, config_name)
            kwargs: dict = {"cache_dir": cache_dir}
            if config_name:
                kwargs["name"] = config_name
            ds = load_dataset(ds_name, **kwargs)

            # Normalise split names
            train_key = "train"
            val_key   = next((k for k in ("validation", "val", "valid") if k in ds), None)
            test_key  = next((k for k in ("test",) if k in ds), None)

            train_ds = ds[train_key]
            val_ds   = ds[val_key] if val_key else ds[train_key].select(range(min(500, len(ds[train_key]))))
            test_ds  = ds[test_key] if test_key else ds[train_key].select(range(min(500, len(ds[train_key]))))

            # Normalise 'text' field — different datasets use different key names
            sample = train_ds[0]
            text_field = next(
                (k for k in ("text", "label", "transcript", "transcription", "ground_truth") if k in sample),
                None,
            )
            if text_field and text_field != "text":
                logger.info("Renaming column '%s' → 'text'", text_field)
                train_ds = train_ds.rename_column(text_field, "text")
                val_ds   = val_ds.rename_column(text_field, "text")
                test_ds  = test_ds.rename_column(text_field, "text")

            logger.info(
                "Dataset '%s' loaded — train: %d | val: %d | test: %d",
                ds_name, len(train_ds), len(val_ds), len(test_ds),
            )
            return train_ds, val_ds, test_ds

        except Exception as exc:
            logger.warning("Failed to load %s: %s — trying next source.", ds_name, exc)
            last_error = exc

    raise RuntimeError(
        f"Could not load any IAM dataset from Hugging Face. Last error: {last_error}"
    )


def inspect_sample(dataset, idx: int = 0) -> dict:
    """Return metadata about one sample for debugging."""
    sample = dataset[idx]
    return {
        "keys": list(sample.keys()),
        "text": sample.get("text", ""),
        "image_size": sample["image"].size if hasattr(sample.get("image"), "size") else None,
        "image_mode": sample["image"].mode if hasattr(sample.get("image"), "mode") else None,
    }
