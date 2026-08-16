"""
Data augmentation pipeline for training the OCR models.

Implements torchvision-compatible transforms AND standalone NumPy/OpenCV
augmentations for use during offline data generation.

Augmentation strategies:
  - Geometric: rotation, affine, perspective, elastic deformation
  - Photometric: brightness, contrast, noise injection
  - Handwriting-specific: stroke width variation (dilation/erosion)
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom torchvision-compatible transforms
# ---------------------------------------------------------------------------

class ElasticDeformation(torch.nn.Module):
    """
    Elastic deformation to simulate handwriting style variability.

    Based on the method from Simard et al. (2003):
    'Best Practices for Convolutional Neural Networks Applied to Visual
    Document Analysis'.

    The deformation field is computed by convolving a random displacement
    field with a Gaussian kernel and then scaling by an amplitude factor.
    """

    def __init__(
        self,
        alpha: float = 30.0,
        sigma: float = 4.0,
        p: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.rng = np.random.default_rng(seed)

    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Random displacement fields
        dx = self.rng.uniform(-1, 1, (h, w)).astype(np.float32)
        dy = self.rng.uniform(-1, 1, (h, w)).astype(np.float32)

        # Smooth with Gaussian
        kernel_size = int(self.sigma * 6) | 1  # force odd
        dx = cv2.GaussianBlur(dx, (kernel_size, kernel_size), self.sigma) * self.alpha
        dy = cv2.GaussianBlur(dy, (kernel_size, kernel_size), self.sigma) * self.alpha

        # Create sampling grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

        if arr.ndim == 2:
            deformed = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR)
        else:
            deformed = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR)

        return Image.fromarray(np.clip(deformed, 0, 255).astype(np.uint8))


class StrokeWidthVariation(torch.nn.Module):
    """
    Randomly thicken or thin character strokes via morphological operations.

    Thickening (dilation) → bold handwriting
    Thinning (erosion)    → light/faint handwriting
    """

    def __init__(self, p: float = 0.3) -> None:
        super().__init__()
        self.p = p

    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        arr = np.array(img)
        op_choice = random.choice(["dilate", "erode", "none"])

        if op_choice == "none":
            return img

        kernel_size = random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if op_choice == "dilate":
            result = cv2.dilate(arr, kernel, iterations=1)
        else:
            result = cv2.erode(arr, kernel, iterations=1)

        return Image.fromarray(result)


class GaussianNoise(torch.nn.Module):
    """Add random Gaussian noise to simulate scanner/camera noise."""

    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.3) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, -1.0, 1.0)


class SaltPepperNoise(torch.nn.Module):
    """Add salt-and-pepper noise to simulate degraded documents."""

    def __init__(self, density: float = 0.02, p: float = 0.2) -> None:
        super().__init__()
        self.density = density
        self.p = p

    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img)
        total = int(arr.size * self.density)

        # Salt (white pixels)
        coords_y = np.random.randint(0, arr.shape[0], total // 2)
        coords_x = np.random.randint(0, arr.shape[1], total // 2)
        arr[coords_y, coords_x] = 255

        # Pepper (black pixels)
        coords_y = np.random.randint(0, arr.shape[0], total // 2)
        coords_x = np.random.randint(0, arr.shape[1], total // 2)
        arr[coords_y, coords_x] = 0

        return Image.fromarray(arr)


class RandomCropPad(torch.nn.Module):
    """
    Randomly crop the image by 1–4 pixels on each side and re-pad.

    Simulates slight misalignment in character extraction.
    """

    def __init__(self, max_crop: int = 4, p: float = 0.3) -> None:
        super().__init__()
        self.max_crop = max_crop
        self.p = p

    def forward(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        top = random.randint(0, self.max_crop)
        bottom = random.randint(0, self.max_crop)
        left = random.randint(0, self.max_crop)
        right = random.randint(0, self.max_crop)
        cropped = img.crop((left, top, w - right or w, h - bottom or h))
        return cropped.resize((w, h), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Composed pipelines
# ---------------------------------------------------------------------------

def get_char_augmentation_pipeline(image_size: Tuple[int, int] = (32, 32)) -> transforms.Compose:
    """
    Augmentation pipeline for character-level training.

    Applied after initial preprocessing and before normalization.
    Produces diverse training samples to prevent overfitting.

    Args:
        image_size: Target (H, W) after augmentation.

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        # Geometric
        ElasticDeformation(alpha=25.0, sigma=4.0, p=0.3),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=(-5, 5),
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        RandomCropPad(max_crop=2, p=0.3),
        # Photometric
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        SaltPepperNoise(density=0.02, p=0.2),
        # Stroke variation
        StrokeWidthVariation(p=0.3),
        # Resize to exact dimensions
        transforms.Resize(image_size),
    ])


def get_word_augmentation_pipeline(
    image_size: Tuple[int, int] = (32, 128),
) -> transforms.Compose:
    """
    Augmentation pipeline for word-level training (CRNN input).

    More conservative than character augmentation because word images
    already contain natural variation.

    Args:
        image_size: Target (H, W) — wide for word strips.

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        SaltPepperNoise(density=0.01, p=0.15),
        transforms.Resize(image_size),
    ])


# ---------------------------------------------------------------------------
# Offline augmentation (OpenCV-based, for bulk data generation)
# ---------------------------------------------------------------------------

def augment_image_opencv(
    image: np.ndarray,
    rotation_range: Tuple[float, float] = (-12.0, 12.0),
    scale_range: Tuple[float, float] = (0.85, 1.15),
    add_noise: bool = True,
    noise_std: float = 8.0,
) -> np.ndarray:
    """
    Apply random augmentation to a grayscale NumPy image using OpenCV.

    Suitable for offline dataset expansion — call this in a preprocessing
    script to generate augmented PNG files.

    Args:
        image:           Grayscale uint8 array.
        rotation_range:  (min, max) rotation angle in degrees.
        scale_range:     (min, max) scaling factor.
        add_noise:       Whether to add Gaussian noise.
        noise_std:       Standard deviation of Gaussian noise.

    Returns:
        Augmented uint8 array of the same shape.
    """
    h, w = image.shape
    angle = random.uniform(*rotation_range)
    scale = random.uniform(*scale_range)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    result = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    if add_noise:
        noise = np.random.normal(0, noise_std, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return result
