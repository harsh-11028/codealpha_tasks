"""
Central configuration for the OCR training pipeline.
Uses dataclasses for type-safety and environment-variable overrides.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import ssl

# Fix macOS Python framework SSL verification issues when downloading torchvision datasets (EMNIST/MNIST)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = MODELS_DIR / "datasets"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
TENSORBOARD_DIR = PROJECT_ROOT / "tensorboard_logs"


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Which datasets to load and how to merge them."""

    # Source flags
    use_mnist: bool = True
    use_emnist: bool = True
    use_iam: bool = False       # Requires manual download + credentials

    # EMNIST split: 'balanced' | 'letters' | 'digits' | 'byclass' | 'bymerge'
    emnist_split: str = "balanced"

    # Root directory where raw datasets are stored / downloaded
    root_dir: Path = DATASETS_DIR / "raw"

    # Output directory for preprocessed tensors (cached to disk)
    processed_dir: Path = DATASETS_DIR / "processed"

    # Validation split fraction (applied after combining datasets)
    val_split: float = 0.10

    # Test split fraction
    test_split: float = 0.10

    # Random seed for reproducibility
    seed: int = 42

    # Whether to download automatically (MNIST / EMNIST only)
    auto_download: bool = True


# ---------------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters."""

    # Target image dimensions fed into the model
    image_size: Tuple[int, int] = (32, 32)  # (H, W)

    # For CRNN / word-level models, wider input
    word_image_size: Tuple[int, int] = (32, 128)  # (H, W)

    # Grayscale conversion
    grayscale: bool = True

    # Binarization
    binarize: bool = True
    binarize_method: str = "otsu"  # 'otsu' | 'adaptive' | 'fixed'
    binarize_threshold: int = 128   # only used if method == 'fixed'

    # Noise removal (Gaussian blur kernel size — must be odd)
    denoise: bool = True
    denoise_kernel_size: int = 3

    # Deskew
    deskew: bool = True
    deskew_angle_limit: float = 15.0  # degrees

    # CLAHE contrast enhancement
    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # Morphological closing to fill gaps
    morphological_close: bool = True
    morph_kernel_size: int = 2

    # Sharpening
    sharpen: bool = False  # off by default; can cause noise amplification

    # Normalization (mean, std) — standard ImageNet values adapted for grayscale
    normalize_mean: Tuple[float, ...] = (0.5,)
    normalize_std: Tuple[float, ...] = (0.5,)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Hyperparameters and training loop settings."""

    # ----- Model -----
    # One of: 'cnn_basic' | 'cnn_batchnorm' | 'residual_cnn' | 'crnn' | 'vit'
    model_name: str = "crnn"

    # Number of output classes:
    #   47 for EMNIST Balanced (digits + upper + lower)
    #   62 for full alphanumeric
    num_classes: int = 47

    # ----- Data loading -----
    batch_size: int = 64
    num_workers: int = 0 if sys.platform == "darwin" else 4
    pin_memory: bool = False if sys.platform == "darwin" else True

    # ----- Optimizer -----
    optimizer: str = "adamw"   # 'adam' | 'adamw' | 'sgd'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9       # only for SGD

    # ----- Scheduler -----
    scheduler: str = "cosine"   # 'cosine' | 'step' | 'plateau' | 'none'
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    lr_min: float = 1e-6

    # ----- Training loop -----
    max_epochs: int = 50
    early_stopping_patience: int = 7
    early_stopping_metric: str = "val_accuracy"
    early_stopping_mode: str = "max"  # 'max' for accuracy, 'min' for loss

    # ----- Mixed precision -----
    use_amp: bool = True        # Automatic Mixed Precision (requires CUDA)

    # ----- Checkpointing -----
    checkpoint_dir: Path = SAVED_MODELS_DIR
    save_best_only: bool = True
    checkpoint_metric: str = "val_accuracy"

    # ----- Logging -----
    tensorboard_dir: Path = TENSORBOARD_DIR
    log_interval: int = 50      # log every N batches

    # ----- Cross validation -----
    use_cross_validation: bool = False
    cv_folds: int = 5

    # ----- Device -----
    device: str = "auto"        # 'auto' | 'cpu' | 'cuda' | 'mps'


# ---------------------------------------------------------------------------
# Model architecture configurations
# ---------------------------------------------------------------------------
@dataclass
class CNNConfig:
    """Hyperparameters specific to CNN architectures."""
    in_channels: int = 1
    base_filters: int = 32
    num_conv_blocks: int = 4
    dropout_rate: float = 0.3
    fc_hidden_size: int = 256


@dataclass
class CRNNConfig:
    """Hyperparameters for CRNN (CNN + BiLSTM + CTC) model."""
    in_channels: int = 1
    cnn_out_channels: int = 512
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.3
    # CTC blank index (typically last class index)
    ctc_blank: int = 0


@dataclass
class ViTConfig:
    """Hyperparameters for Vision Transformer."""
    image_size: int = 32
    patch_size: int = 4         # 32 / 4 = 8x8 = 64 patches
    in_channels: int = 1
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1


# ---------------------------------------------------------------------------
# Inference / prediction configuration
# ---------------------------------------------------------------------------
def _default_easyocr_languages() -> List[str]:
    return ["en"]


@dataclass
class InferenceConfig:
    """Settings for the OCR prediction pipeline and inference engines."""

    # Which model to use by default (will auto-select if 'auto')
    default_model: str = "auto"

    # Top-k candidates to retain per character
    top_k: int = 3

    # Confidence threshold below which prediction is flagged as uncertain
    confidence_threshold: float = 0.70

    # OCR engine routing
    easyocr_enabled: bool = True
    tesseract_enabled: bool = True
    custom_model_enabled: bool = True

    # EasyOCR settings
    easyocr_languages: List[str] = field(default_factory=_default_easyocr_languages)
    easyocr_gpu: bool = False

    # Segmentation settings
    min_char_area: int = 50           # minimum contour area for a character
    char_aspect_ratio_range: Tuple[float, float] = (0.1, 3.0)
    line_detection_method: str = "projection"  # 'projection' | 'hough'


# ---------------------------------------------------------------------------
# Master config — single object imported throughout the project
# ---------------------------------------------------------------------------
@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    crnn: CRNNConfig = field(default_factory=CRNNConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self) -> None:
        """Create required directories and apply environment overrides."""
        self._create_dirs()
        self._apply_env_overrides()

    def _create_dirs(self) -> None:
        dirs = [
            self.dataset.root_dir,
            self.dataset.processed_dir,
            self.training.checkpoint_dir,
            self.training.tensorboard_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _apply_env_overrides(self) -> None:
        """Override config values from environment variables if present."""
        if batch_size := os.getenv("BATCH_SIZE"):
            self.training.batch_size = int(batch_size)
        if lr := os.getenv("LEARNING_RATE"):
            self.training.learning_rate = float(lr)
        if epochs := os.getenv("NUM_EPOCHS"):
            self.training.max_epochs = int(epochs)
        if device := os.getenv("DEVICE"):
            self.training.device = device
        if model := os.getenv("DEFAULT_MODEL"):
            self.training.model_name = model
        if workers := os.getenv("NUM_WORKERS"):
            self.training.num_workers = int(workers)


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
config = Config()


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
def get_emnist_balanced_labels() -> dict[int, str]:
    """
    Returns the 47-class label map for EMNIST Balanced.
    Classes: 0-9 (digits), 10-35 (uppercase A-Z), 36-46 (lowercase a-z, merged).
    Merged lowercase: a,b,d,e,f,g,h,n,q,r,t
    """
    labels: dict[int, str] = {}
    for i in range(10):
        labels[i] = str(i)
    for i in range(26):
        labels[10 + i] = chr(ord("A") + i)
    merged_lower = list("abdefghnqrt")
    for i, ch in enumerate(merged_lower):
        labels[36 + i] = ch
    return labels


def get_full_alphanumeric_labels() -> dict[int, str]:
    """Returns 62-class label map: 0-9, A-Z, a-z."""
    labels: dict[int, str] = {}
    for i in range(10):
        labels[i] = str(i)
    for i in range(26):
        labels[10 + i] = chr(ord("A") + i)
    for i in range(26):
        labels[36 + i] = chr(ord("a") + i)
    return labels


EMNIST_BALANCED_LABELS = get_emnist_balanced_labels()
FULL_ALPHANUMERIC_LABELS = get_full_alphanumeric_labels()
