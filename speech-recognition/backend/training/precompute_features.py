"""
training/precompute_features.py — Single-process HDF5 feature pre-extraction.

WHY THIS IS NEEDED
------------------
The DataLoader uses num_workers > 0 (multi-process) to load data during training.
HDF5 files (the feature cache) are NOT safe for concurrent writes from multiple
processes. When the old code opened the HDF5 in 'append' mode inside the dataset
__getitem__ with 4 workers, all workers raced to write simultaneously, corrupting
features or causing all features to read as zeros.  The model then saw uniform
input across all samples and learned a constant (uniform) output distribution —
explaining the ~15% confidence identical predictions.

FIX
---
Run this script ONCE before training.  It extracts every audio file in a single
process and writes all features to HDF5.  Training then opens the file read-only
from the main process (no concurrent writes).

Usage
-----
  cd speech-emotion-recognition/backend
  source venv/bin/activate
  export PYTHONPATH=.
  python -m training.precompute_features          # uses data/processed/metadata.csv
  python -m training.precompute_features --csv path/to/custom/metadata.csv
"""

import sys
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from ml.feature_extraction.extractor import FeatureExtractor
from ml.preprocessing.audio_processor import AudioProcessor
from training.config import DEFAULT_CONFIG
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--csv",
    default=None,
    help="Path to metadata CSV. Defaults to config processed_data_dir / metadata.csv",
)
@click.option(
    "--out",
    default=None,
    help="Output HDF5 path. Defaults to config feature_store_path.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing HDF5 file.",
)
def main(csv, out, overwrite):
    """Pre-extract acoustic features for all audio files into a single HDF5 store."""
    setup_logger(log_level="INFO", log_file="logs/precompute.log")

    config = DEFAULT_CONFIG

    csv_path = Path(csv) if csv else Path(config.dataset.processed_data_dir) / "metadata.csv"
    h5_path = Path(out) if out else Path(config.features.feature_store_path)

    if not csv_path.exists():
        logger.error(f"metadata.csv not found at {csv_path}. Run dataset pipeline first.")
        sys.exit(1)

    if h5_path.exists():
        if overwrite:
            logger.info(f"Overwrite flag set — removing existing {h5_path}")
            h5_path.unlink()
        else:
            logger.info(
                f"HDF5 already exists at {h5_path}. "
                "Use --overwrite to rebuild. Skipping pre-extraction."
            )
            return

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    processor = AudioProcessor(config.audio)
    extractor = FeatureExtractor(config)

    ok = 0
    fail = 0
    feature_shape = None

    with h5py.File(h5_path, "w") as f:
        feat_grp = f.create_group("features")
        label_grp = f.create_group("labels")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            file_path = row["file_path"]
            emotion = row["emotion"]
            file_id = Path(file_path).stem

            # Skip duplicates
            if file_id in feat_grp:
                continue

            try:
                audio, sr = processor.process(file_path, denoise=False)  # no denoise = faster
                features = extractor.extract(audio, sr)          # shape: (N_feat, T)

                if feature_shape is None:
                    feature_shape = features.shape
                    logger.info(f"Feature shape detected: {feature_shape}")

                feat_grp.create_dataset(
                    file_id,
                    data=features.astype(np.float32),
                    compression="lzf",   # fast, lossless
                )
                label_grp.create_dataset(
                    file_id,
                    data=np.bytes_(emotion),
                )
                ok += 1

            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                fail += 1

    logger.info(
        f"✅ Done. Extracted {ok} files, skipped {fail} errors.\n"
        f"   HDF5 saved to: {h5_path}\n"
        f"   Feature shape: {feature_shape}\n"
        f"   File size: {h5_path.stat().st_size / 1024**2:.1f} MB"
    )


if __name__ == "__main__":
    main()
