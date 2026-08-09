"""
dataset/loaders/savee.py — Parser for the SAVEE dataset.

Filename format: [Actor]_[Emotion][Number].wav
Example: DC_a01.wav, JE_h12.wav

Actors: DC, JE, JK, KL (All Male)
Emotions: a (angry), d (disgust), f (fear), h (happy), n (neutral), sa (sad), su (surprise)
"""

import re
from pathlib import Path
from typing import Iterator

from dataset.label_normalizer import normalize_label
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_dataset(raw_dir: Path) -> Iterator[dict]:
    """
    Scans the SAVEE directory and yields parsed file metadata.
    """
    if not raw_dir.exists():
        logger.warning(f"SAVEE directory not found: {raw_dir}")
        return

    wav_files = list(raw_dir.rglob("*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files in SAVEE")

    # Regex to extract emotion and number from e.g. "sa01" or "a15"
    pattern = re.compile(r"^([a-z]+)(\d+)$")

    for file_path in wav_files:
        filename = file_path.stem
        parts = filename.split("_")
        
        if len(parts) != 2:
            logger.debug(f"Skipping malformed SAVEE filename: {filename}")
            continue

        actor, emotion_code = parts
        
        match = pattern.match(emotion_code)
        if not match:
            logger.debug(f"Skipping malformed SAVEE emotion code: {emotion_code}")
            continue
            
        raw_emotion = match.group(1)
        
        normalized_emotion = normalize_label("savee", raw_emotion)
        
        if not normalized_emotion:
            continue

        yield {
            "file_path": str(file_path.absolute()),
            "dataset": "savee",
            "raw_emotion": raw_emotion,
            "emotion": normalized_emotion,
            "actor_id": actor,
            "gender": "male",  # SAVEE only has male actors
        }
