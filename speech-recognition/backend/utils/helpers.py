"""
utils/helpers.py — General utility functions used across the project.
"""

import time
import uuid
import hashlib
from pathlib import Path
from typing import Any, Optional
import numpy as np


def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return str(uuid.uuid4())


def compute_file_hash(file_bytes: bytes, algorithm: str = "sha256") -> str:
    """Compute a hash of file bytes to detect duplicates."""
    h = hashlib.new(algorithm)
    h.update(file_bytes)
    return h.hexdigest()


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS.ms string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.label}: {self.elapsed * 1000:.2f}ms"


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of given size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def normalize_probabilities(probs: dict[str, float]) -> dict[str, float]:
    """Ensure probabilities sum to 1.0."""
    total = sum(probs.values())
    if total == 0:
        return {k: 1.0 / len(probs) for k in probs}
    return {k: v / total for k, v in probs.items()}
