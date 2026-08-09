"""
dataset/loaders/crema_d.py — Parser for the CREMA-D dataset.

Filename format: [ActorID]_[Sentence]_[Emotion]_[Intensity].wav
Example: 1001_DFA_ANG_XX.wav

ActorID: 1001 to 1091 (91 actors: 48 male, 43 female)
Emotions: ANG, DIS, FEA, HAP, NEU, SAD
Intensity: LO, MD, HI, XX (unspecified)
"""

from pathlib import Path
from typing import Iterator

from dataset.label_normalizer import normalize_label
from utils.logger import get_logger

logger = get_logger(__name__)

# Based on CREMA-D documentation, these actors are female
# 1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091
FEMALE_ACTORS = {
    1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021, 1024, 
    1025, 1028, 1029, 1030, 1037, 1043, 1046, 1047, 1049, 1052, 1053, 1054, 1055, 1056, 
    1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082, 1084, 1089, 1091
}


def parse_dataset(raw_dir: Path) -> Iterator[dict]:
    """
    Scans the CREMA-D directory and yields parsed file metadata.
    """
    if not raw_dir.exists():
        logger.warning(f"CREMA-D directory not found: {raw_dir}")
        return

    wav_files = list(raw_dir.rglob("*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files in CREMA-D")

    for file_path in wav_files:
        filename = file_path.stem
        parts = filename.split("_")
        
        if len(parts) != 4:
            logger.debug(f"Skipping malformed CREMA-D filename: {filename}")
            continue

        actor_id_str, sentence, emotion, intensity = parts
        
        raw_emotion = emotion
        normalized_emotion = normalize_label("crema_d", raw_emotion)
        
        if not normalized_emotion:
            continue
            
        try:
            actor_id = int(actor_id_str)
            gender = "female" if actor_id in FEMALE_ACTORS else "male"
        except ValueError:
            actor_id = actor_id_str
            gender = "unknown"

        yield {
            "file_path": str(file_path.absolute()),
            "dataset": "crema_d",
            "raw_emotion": raw_emotion,
            "emotion": normalized_emotion,
            "actor_id": str(actor_id),
            "gender": gender,
            "intensity": intensity.lower()
        }
