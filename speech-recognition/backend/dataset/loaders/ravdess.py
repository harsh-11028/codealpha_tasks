"""
dataset/loaders/ravdess.py — Parser for the RAVDESS dataset.

Filename identifiers (e.g., 03-01-06-01-02-01-12.wav):
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
"""

from pathlib import Path
from typing import Iterator

from dataset.label_normalizer import normalize_label
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_dataset(raw_dir: Path) -> Iterator[dict]:
    """
    Scans the RAVDESS directory and yields parsed file metadata.
    
    Args:
        raw_dir: Path to the root of the RAVDESS dataset.
        
    Yields:
        Dictionary containing file metadata and normalized labels.
    """
    if not raw_dir.exists():
        logger.warning(f"RAVDESS directory not found: {raw_dir}")
        return

    # Files are usually organized in folders like Actor_01, Actor_02, etc.
    # But we can just rglob all wav files
    wav_files = list(raw_dir.rglob("*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files in RAVDESS")

    for file_path in wav_files:
        filename = file_path.stem
        parts = filename.split("-")
        
        if len(parts) != 7:
            logger.debug(f"Skipping malformed RAVDESS filename: {filename}")
            continue

        # Extract components
        modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts
        
        # We generally only care about audio-only or audio extracted from video
        # But for SER, we can use any of them as long as it's speech/song audio
        
        raw_emotion = emotion
        normalized_emotion = normalize_label("ravdess", raw_emotion)
        
        if not normalized_emotion:
            continue

        actor_id = int(actor)
        gender = "female" if actor_id % 2 == 0 else "male"
        
        yield {
            "file_path": str(file_path.absolute()),
            "dataset": "ravdess",
            "raw_emotion": raw_emotion,
            "emotion": normalized_emotion,
            "actor_id": str(actor_id),
            "gender": gender,
            "intensity": "normal" if intensity == "01" else "strong",
            "vocal_channel": "speech" if vocal_channel == "01" else "song"
        }
