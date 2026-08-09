"""
dataset/label_normalizer.py — Unified label normalization across all datasets.
Maps dataset-specific raw emotion labels to the unified 8-class schema.
"""

from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# The target 8 classes as defined in training/config.py
TARGET_EMOTIONS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise"
}

# Mapping dictionaries for each dataset
# Note: Keys are typically derived from filenames by the respective loaders.

RAVDESS_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

TESS_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "ps": "surprise"  # pleasant surprise
}

SAVEE_MAP = {
    "n": "neutral",
    "h": "happy",
    "sa": "sad",
    "a": "angry",
    "f": "fear",
    "d": "disgust",
    "su": "surprise"
}

CREMA_D_MAP = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear",
    "DIS": "disgust"
}

EMO_DB_MAP = {
    "N": "neutral",
    "F": "happy",  # Freude (Happiness)
    "T": "sad",    # Trauer (Sadness)
    "W": "angry",  # Ärger (Anger)
    "A": "fear",   # Angst (Fear)
    "E": "disgust",# Ekel (Disgust)
    "L": "calm",   # Langeweile (Boredom) mapped to calm for balance
}


def normalize_label(dataset_name: str, raw_label: str) -> Optional[str]:
    """
    Converts a dataset-specific raw label to the unified emotion schema.

    Args:
        dataset_name: Identifier for the dataset (e.g., 'ravdess', 'tess')
        raw_label: The raw string extracted from the file/metadata

    Returns:
        The normalized emotion string, or None if unrecognized.
    """
    dataset_name = dataset_name.lower().strip()
    raw_label = raw_label.strip()

    normalized = None

    if dataset_name == "ravdess":
        normalized = RAVDESS_MAP.get(raw_label)
    elif dataset_name == "tess":
        normalized = TESS_MAP.get(raw_label.lower())
    elif dataset_name == "savee":
        normalized = SAVEE_MAP.get(raw_label.lower())
    elif dataset_name == "crema_d" or dataset_name == "cremad":
        normalized = CREMA_D_MAP.get(raw_label.upper())
    elif dataset_name == "emo_db" or dataset_name == "emodb":
        normalized = EMO_DB_MAP.get(raw_label.upper())
    else:
        logger.warning(f"Unknown dataset name for normalization: {dataset_name}")
        return None

    if normalized is None:
        logger.debug(f"Unmapped label '{raw_label}' in dataset '{dataset_name}'")

    return normalized
