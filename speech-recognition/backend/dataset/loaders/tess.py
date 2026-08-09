"""
dataset/loaders/tess.py — Parser for the TESS dataset.

Filename format: [Actor]_[Word]_[Emotion].wav
Example: OAF_back_angry.wav or YAF_back_angry.wav

OAF = Older Actor Female
YAF = Younger Actor Female
"""

from pathlib import Path
from typing import Iterator

from dataset.label_normalizer import normalize_label
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_dataset(raw_dir: Path) -> Iterator[dict]:
    """
    Scans the TESS directory and yields parsed file metadata.
    """
    if not raw_dir.exists():
        logger.warning(f"TESS directory not found: {raw_dir}")
        return

    wav_files = list(raw_dir.rglob("*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files in TESS")

    for file_path in wav_files:
        filename = file_path.stem
        parts = filename.split("_")
        
        if len(parts) != 3:
            logger.debug(f"Skipping malformed TESS filename: {filename}")
            continue

        actor, word, emotion = parts
        
        raw_emotion = emotion.lower()
        normalized_emotion = normalize_label("tess", raw_emotion)
        
        if not normalized_emotion:
            continue

        yield {
            "file_path": str(file_path.absolute()),
            "dataset": "tess",
            "raw_emotion": raw_emotion,
            "emotion": normalized_emotion,
            "actor_id": actor,
            "gender": "female",  # TESS only has two female actors
        }
