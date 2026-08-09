"""
training/dataset.py — PyTorch Dataset for SER.

Reads the metadata.csv generated in Phase 3, and fetches features from the HDF5 store.
Features must be pre-computed with training/precompute_features.py before training.
If features are not present in HDF5 for a given file, they are computed on-the-fly
(slower, but allows partial feature stores).
"""

import ast
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.preprocessing.audio_processor import AudioProcessor
from ml.feature_extraction.extractor import FeatureExtractor
from training.config import Config, EMOTION_TO_IDX
from utils.logger import get_logger

logger = get_logger(__name__)


class SERDataset(Dataset):
    """
    Speech Emotion Recognition Dataset.

    Reads pre-extracted features from HDF5 (fast path) or computes them
    on-the-fly from raw audio (slow path / fallback).

    IMPORTANT: Open HDF5 in read-only mode.  The file is written ONCE by
    precompute_features.py and must not be touched during training to avoid
    data-corruption when multiple workers are used.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        split: str,
        config: Config,
        is_wav2vec2: bool = False,
    ):
        self.config = config
        self.is_wav2vec2 = is_wav2vec2

        # Filter by split
        self.df = metadata_df[metadata_df["split"] == split].reset_index(drop=True)
        logger.info(f"Loaded {len(self.df)} samples for {split} split.")

        self.h5_path = Path(config.features.feature_store_path)
        self._h5_ok = self.h5_path.exists()
        if not self._h5_ok:
            logger.warning(
                f"HDF5 not found at {self.h5_path}. "
                "Features will be computed on-the-fly (slow). "
                "Run 'python -m training.precompute_features' first for best performance."
            )

        # Fallback processors (used when feature is not in HDF5)
        self.audio_processor = AudioProcessor(config.audio)
        self.feature_extractor = FeatureExtractor(config)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        emotion = row["emotion"]
        file_id = Path(file_path).stem

        label_idx = EMOTION_TO_IDX.get(emotion, 0)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        # ── Wav2Vec2 expects raw audio (1D) ──────────────────────────────────
        if self.is_wav2vec2:
            audio, _ = self.audio_processor.process(file_path, denoise=False)
            return torch.FloatTensor(audio), label_tensor

        # ── Standard models: 2D feature tensor ───────────────────────────────
        features = self._load_features(file_id, file_path)
        return torch.FloatTensor(features), label_tensor

    def _load_features(self, file_id: str, file_path: str) -> np.ndarray:
        """Load features from HDF5 cache, or compute on-the-fly as fallback."""
        # Fast path: read from pre-computed HDF5 (read-only, safe for DataLoader workers)
        if self._h5_ok:
            try:
                with h5py.File(self.h5_path, "r") as f:
                    if "features" in f and file_id in f["features"]:
                        return f["features"][file_id][:]
            except Exception as e:
                logger.debug(f"HDF5 read failed for {file_id}: {e}")

        # Slow path: compute on-the-fly
        logger.debug(f"Computing features on-the-fly for {file_id}")
        audio, sr = self.audio_processor.process(file_path, denoise=False)
        return self.feature_extractor.extract(audio, sr)
