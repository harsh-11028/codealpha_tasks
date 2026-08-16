"""
Image preprocessing pipeline for OCR.

Implements all operations needed to prepare a raw handwritten image
for character segmentation and model inference:

    1. Load & decode (supports bytes, file path, PIL Image, NumPy array)
    2. Grayscale conversion
    3. Noise removal (Gaussian blur)
    4. Contrast enhancement (CLAHE)
    5. Binarization (Otsu / Adaptive / Fixed threshold)
    6. Morphological operations (closing, dilation, erosion)
    7. Deskew  (see deskew.py for the heavy lifting)
    8. Resize
    9. Normalization
    10. Sharpening (optional)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from models.training.config import Config, PreprocessingConfig, config as default_config

logger = logging.getLogger(__name__)

# Type alias for anything that can be turned into a numpy array
ImageInput = Union[str, Path, bytes, np.ndarray, Image.Image]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_image(source: ImageInput) -> np.ndarray:
    """
    Load an image from various source types and return a BGR uint8 NumPy array.

    Args:
        source: File path, raw bytes, PIL Image, or NumPy array.

    Returns:
        BGR uint8 array of shape (H, W, 3) or (H, W).

    Raises:
        ValueError: If the source cannot be decoded.
    """
    if isinstance(source, np.ndarray):
        return source.copy()

    if isinstance(source, Image.Image):
        return cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)

    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image from path: {source}")
        return img

    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not decode image from bytes.")
        return img

    raise ValueError(f"Unsupported image source type: {type(source)}")


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA image to single-channel grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def remove_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur for noise removal.

    Args:
        image:       Grayscale uint8 array.
        kernel_size: Must be odd. Larger → more smoothing.

    Returns:
        Denoised grayscale array.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def enhance_contrast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Better than global histogram equalization for handwriting because
    it preserves local contrast without over-amplifying noise.

    Args:
        image:      Grayscale uint8 array.
        clip_limit: Threshold for contrast limiting.
        tile_grid:  Size of grid for histogram equalization.

    Returns:
        Contrast-enhanced grayscale array.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def binarize(
    image: np.ndarray,
    method: str = "otsu",
    threshold: int = 128,
) -> np.ndarray:
    """
    Binarize a grayscale image to black (text) on white (background).

    Args:
        image:     Grayscale uint8 array.
        method:    'otsu' | 'adaptive' | 'fixed'
        threshold: Used only when method == 'fixed'.

    Returns:
        Binary uint8 array (values 0 or 255).
    """
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )
    elif method == "fixed":
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError(f"Unknown binarization method: {method!r}")
    return binary


def morphological_close(image: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """
    Apply morphological closing to fill small gaps in stroke lines.

    Args:
        image:       Binary uint8 array (text=255, bg=0).
        kernel_size: Size of the rectangular structuring element.

    Returns:
        Morphologically closed binary array.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Sharpen image using an unsharp mask kernel.

    Args:
        image: Grayscale uint8 array.

    Returns:
        Sharpened array (same dtype and shape).
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect: bool = True,
    padding_value: int = 0,
) -> np.ndarray:
    """
    Resize image to target_size (H, W).

    If keep_aspect=True, letterbox-pads the shorter dimension with padding_value
    so no distortion occurs — important for character shapes.

    Args:
        image:         Grayscale or binary uint8 array.
        target_size:   (target_H, target_W).
        keep_aspect:   Preserve aspect ratio with padding.
        padding_value: Pixel value used for padding (0=black, 255=white).

    Returns:
        Resized uint8 array of shape (target_H, target_W).
    """
    target_h, target_w = target_size
    h, w = image.shape[:2]

    if keep_aspect:
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to exact target size
        canvas = np.full((target_h, target_w), padding_value, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas
    else:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def normalize_to_float(
    image: np.ndarray,
    mean: float = 0.5,
    std: float = 0.5,
) -> np.ndarray:
    """
    Normalize a uint8 grayscale image to float32 in range [-1, 1].

    Formula: (pixel / 255.0 - mean) / std

    Args:
        image: Grayscale uint8 array.
        mean:  Normalization mean.
        std:   Normalization std.

    Returns:
        Float32 array of same shape with values in [-1, 1].
    """
    img_float = image.astype(np.float32) / 255.0
    return (img_float - mean) / std


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Stateless preprocessing pipeline for OCR inference.

    Designed to be called on individual images during prediction —
    NOT as a PyTorch transform (use torchvision transforms for that).
    """

    def __init__(self, cfg: Config = default_config) -> None:
        self.cfg = cfg
        self.pc: PreprocessingConfig = cfg.preprocessing

    def preprocess(
        self,
        source: ImageInput,
        target_size: Optional[Tuple[int, int]] = None,
        word_mode: bool = False,
        return_stages: bool = False,
    ) -> Union[np.ndarray, dict]:
        """
        Run the full preprocessing pipeline on an input image.

        Args:
            source:        Raw image (path, bytes, PIL, numpy).
            target_size:   Override output size; defaults to config value.
            word_mode:     Use word_image_size instead of image_size.
            return_stages: If True, return a dict with all intermediate stages
                           (useful for debugging / visualization).

        Returns:
            Preprocessed float32 array of shape (H, W), values in [-1, 1].
            Or a dict of stages if return_stages=True.
        """
        pc = self.pc
        stages: dict = {}

        # 1. Load
        img = load_image(source)
        if return_stages:
            stages["raw"] = img.copy()

        # 2. Grayscale
        img = to_grayscale(img)
        if return_stages:
            stages["grayscale"] = img.copy()

        # 3. Noise removal
        if pc.denoise:
            img = remove_noise(img, pc.denoise_kernel_size)
            if return_stages:
                stages["denoised"] = img.copy()

        # 4. Contrast enhancement
        if pc.enhance_contrast:
            img = enhance_contrast_clahe(img, pc.clahe_clip_limit, pc.clahe_tile_grid)
            if return_stages:
                stages["contrast_enhanced"] = img.copy()

        # 5. Binarization
        if pc.binarize:
            img = binarize(img, pc.binarize_method, pc.binarize_threshold)
            if return_stages:
                stages["binarized"] = img.copy()

        # 6. Deskew
        if pc.deskew:
            from models.preprocessing.deskew import deskew_image
            img = deskew_image(img, max_angle=pc.deskew_angle_limit)
            if return_stages:
                stages["deskewed"] = img.copy()

        # 7. Morphological closing
        if pc.morphological_close and pc.binarize:
            img = morphological_close(img, pc.morph_kernel_size)
            if return_stages:
                stages["morphological"] = img.copy()

        # 8. Sharpening
        if pc.sharpen:
            img = sharpen(img)
            if return_stages:
                stages["sharpened"] = img.copy()

        # 9. Resize
        size = (
            pc.word_image_size if word_mode
            else (target_size or pc.image_size)
        )
        img = resize_image(img, size, keep_aspect=True, padding_value=0)
        if return_stages:
            stages["resized"] = img.copy()

        # 10. Normalize to float
        mean = pc.normalize_mean[0]
        std = pc.normalize_std[0]
        img_float = normalize_to_float(img, mean, std)
        if return_stages:
            stages["normalized"] = img_float.copy()
            return stages

        return img_float

    def preprocess_batch(
        self,
        sources: list,
        word_mode: bool = False,
    ) -> np.ndarray:
        """
        Preprocess a list of images and stack into a batch array.

        Args:
            sources:   List of image sources.
            word_mode: Use word-level dimensions.

        Returns:
            Float32 array of shape (N, H, W).
        """
        processed = [self.preprocess(src, word_mode=word_mode) for src in sources]
        return np.stack(processed, axis=0)

    def preprocess_to_tensor(
        self,
        source: ImageInput,
        word_mode: bool = False,
    ):
        """
        Preprocess an image and return a PyTorch tensor of shape (1, 1, H, W).

        Args:
            source:    Image source.
            word_mode: Use word-level dimensions.

        Returns:
            Float32 torch.Tensor of shape (1, 1, H, W) (batch=1, channels=1).
        """
        import torch
        img_float = self.preprocess(source, word_mode=word_mode)
        tensor = torch.from_numpy(img_float).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor
