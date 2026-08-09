"""
training/config.py — Central configuration for training pipeline.
All hyperparameters, paths, and settings in one place.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import torch


# ── Emotion label map (normalized across all datasets) ──────────────────────
EMOTION_LABELS: dict[int, str] = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fear",
    6: "disgust",
    7: "surprise",
}

EMOTION_TO_IDX: dict[str, int] = {v: k for k, v in EMOTION_LABELS.items()}

NUM_CLASSES: int = len(EMOTION_LABELS)

# Emoji map for UI display
EMOTION_EMOJI: dict[str, str] = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲",
}

# Color map for visualizations
EMOTION_COLORS: dict[str, str] = {
    "neutral": "#94a3b8",
    "calm": "#818cf8",
    "happy": "#fbbf24",
    "sad": "#60a5fa",
    "angry": "#f87171",
    "fear": "#a78bfa",
    "disgust": "#4ade80",
    "surprise": "#fb923c",
}


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 22050
    max_duration: float = 6.0          # seconds (pad/trim to this)
    hop_length: int = 512
    n_fft: int = 2048
    n_mfcc: int = 40
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0
    chunk_duration: float = 3.0        # for live streaming
    silence_threshold: float = 0.01    # amplitude below this = silence
    top_db: float = 60.0              # for silence trimming


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enabled: bool = True
    noise_factor: float = 0.005
    pitch_shift_steps: List[int] = field(default_factory=lambda: [-2, -1, 1, 2])
    time_stretch_rates: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.1, 1.2])
    gain_range: tuple = (0.7, 1.3)
    crop_fraction: float = 0.1        # max fraction to randomly crop


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    use_mfcc: bool = True
    use_delta_mfcc: bool = True
    use_delta2_mfcc: bool = True
    use_mel_spectrogram: bool = True
    use_chroma: bool = True
    use_zcr: bool = True
    use_spectral_centroid: bool = True
    use_spectral_contrast: bool = True
    use_tonnetz: bool = True
    use_rms: bool = True
    use_pitch: bool = True
    feature_store_path: str = "data/processed/features.h5"


@dataclass
class TrainingConfig:
    """Model training configuration."""
    # General
    seed: int = 42
    num_epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Optimizer
    optimizer: str = "adamw"           # adam | adamw | sgd
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9              # for SGD

    # Scheduler
    scheduler: str = "cosine"          # cosine | step | plateau | none
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_t_max: int = 50

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # Regularization
    dropout_rate: float = 0.4
    label_smoothing: float = 0.1

    # Validation
    val_split: float = 0.15
    test_split: float = 0.15
    n_folds: int = 5                   # for cross-validation
    stratified: bool = True

    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "saved_models"
    tensorboard_dir: str = "runs"

    # Mixed precision
    use_amp: bool = True               # automatic mixed precision

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""
    # CNN
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    cnn_pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])

    # LSTM / BiLSTM
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    bidirectional: bool = True

    # Attention
    attention_heads: int = 8
    attention_dim: int = 256

    # Wav2Vec2 (Transfer Learning)
    wav2vec2_model_name: str = "facebook/wav2vec2-base"
    wav2vec2_freeze_feature_encoder: bool = True
    wav2vec2_freeze_layers: int = 6    # freeze first N transformer layers


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""
    datasets: List[str] = field(default_factory=lambda: ["ravdess", "tess"])
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    cache_features: bool = True
    balance_classes: bool = True       # oversample minority classes


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Active model for inference
    active_model: str = "best"         # best | cnn | cnn_lstm | bilstm | cnn_attention | wav2vec2

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def emotion_labels(self) -> dict:
        return EMOTION_LABELS

    def to_dict(self) -> dict:
        """Serialize config to plain dict (for logging/saving)."""
        import dataclasses
        return dataclasses.asdict(self)


# ── Singleton default config ─────────────────────────────────────────────────
DEFAULT_CONFIG = Config()
