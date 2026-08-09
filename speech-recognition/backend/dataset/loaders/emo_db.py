"""
dataset/loaders/emo_db.py — Parser for the Berlin EMO-DB dataset.

Filename format: [Speaker(2 chars)][Text(3 chars)][Emotion(1 char)][Version(1 char)].wav
Example: 03a01Fa.wav

Emotions: W (Anger), L (Boredom->calm), E (Disgust), A (Fear), F (Happiness), T (Sadness), N (Neutral).
Speakers: 03, 08, 09, 10, 11, 12, 13, 14, 15, 16
Female speakers: 03, 08, 09, 13, 14
Male speakers: 10, 11, 12, 15, 16
"""

from pathlib import Path
from typing import Iterator

from dataset.label_normalizer import normalize_label
from utils.logger import get_logger

logger = get_logger(__name__)

FEMALE_SPEAKERS = {"03", "08", "09", "13", "14"}


def parse_dataset(raw_dir: Path) -> Iterator[dict]:
    """
    Scans the EMO-DB directory and yields parsed file metadata.
    """
    if not raw_dir.exists():
        logger.warning(f"EMO-DB directory not found: {raw_dir}")
        return

    wav_files = list(raw_dir.rglob("*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files in EMO-DB")

    for file_path in wav_files:
        filename = file_path.stem
        
        # EmoDB filenames are typically exactly 7 characters
        if len(filename) != 7:
            logger.debug(f"Skipping malformed EMO-DB filename: {filename}")
            continue

        speaker = filename[0:2]
        text_code = filename[2:5]
        emotion = filename[5:6]
        version = filename[6:7]
        
        raw_emotion = emotion
        normalized_emotion = normalize_label("emo_db", raw_emotion)
        
        if not normalized_emotion:
            continue

        gender = "female" if speaker in FEMALE_SPEAKERS else "male"

        yield {
            "file_path": str(file_path.absolute()),
            "dataset": "emo_db",
            "raw_emotion": raw_emotion,
            "emotion": normalized_emotion,
            "actor_id": speaker,
            "gender": gender,
        }
