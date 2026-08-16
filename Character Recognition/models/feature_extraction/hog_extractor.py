"""
Histogram of Oriented Gradients (HOG) feature extractor.

HOG captures the distribution of gradient orientations in localized
image regions — one of the most effective hand-crafted features for
character recognition.

This module provides:
  - Standalone HOG feature extraction (scikit-image based)
  - A PyTorch-compatible HOG feature module for hybrid CNN+HOG models
  - Batch extraction for datasets
  - Visualization utilities
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import hog as skimage_hog
from skimage.transform import resize as skimage_resize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HOG configuration
# ---------------------------------------------------------------------------

HOG_DEFAULT_CONFIG = {
    "orientations": 9,           # number of gradient orientation bins
    "pixels_per_cell": (4, 4),   # cell size in pixels
    "cells_per_block": (2, 2),   # number of cells per block
    "block_norm": "L2-Hys",      # block normalization method
    "transform_sqrt": True,      # gamma correction before HOG
    "feature_vector": True,      # flatten to 1-D
}


# ---------------------------------------------------------------------------
# Core HOG extraction
# ---------------------------------------------------------------------------

def extract_hog(
    image: np.ndarray,
    image_size: Tuple[int, int] = (32, 32),
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (4, 4),
    cells_per_block: Tuple[int, int] = (2, 2),
    block_norm: str = "L2-Hys",
    visualize: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract HOG feature vector from a grayscale image.

    Args:
        image:           Grayscale uint8 or float image.
        image_size:      Resize to this (H, W) before computing HOG.
        orientations:    Number of gradient orientation bins.
        pixels_per_cell: HOG cell size.
        cells_per_block: HOG block size in cells.
        block_norm:      Block normalization method.
        visualize:       If True, return HOG visualization image too.

    Returns:
        (feature_vector, hog_image_or_None)
        feature_vector: 1-D float64 array.
        hog_image:      2-D float array for visualization (or None).
    """
    # Ensure float [0, 1]
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    # Resize
    if img.shape[:2] != image_size:
        img = skimage_resize(img, image_size, anti_aliasing=True)

    features, hog_img = skimage_hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        transform_sqrt=True,
        feature_vector=True,
        visualize=True,
    )

    return features, (hog_img if visualize else None)


def get_hog_feature_size(
    image_size: Tuple[int, int] = (32, 32),
    pixels_per_cell: Tuple[int, int] = (4, 4),
    cells_per_block: Tuple[int, int] = (2, 2),
    orientations: int = 9,
) -> int:
    """
    Compute the expected length of the HOG feature vector.

    Useful for initializing classifier input dimensions.

    Args:
        image_size:      (H, W) input image size.
        pixels_per_cell: HOG cell size in pixels.
        cells_per_block: HOG block size in cells.
        orientations:    Number of gradient bins.

    Returns:
        Integer length of the resulting HOG feature vector.
    """
    h, w = image_size
    n_cells_y = h // pixels_per_cell[0]
    n_cells_x = w // pixels_per_cell[1]
    n_blocks_y = n_cells_y - cells_per_block[0] + 1
    n_blocks_x = n_cells_x - cells_per_block[1] + 1
    return n_blocks_y * n_blocks_x * cells_per_block[0] * cells_per_block[1] * orientations


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

class HOGExtractor:
    """
    Stateful HOG extractor with configurable parameters.

    Usage:
        extractor = HOGExtractor(image_size=(32, 32))
        features = extractor.extract(image)
        batch_features = extractor.extract_batch(images)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (32, 32),
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (4, 4),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
    ) -> None:
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.feature_size = get_hog_feature_size(
            image_size, pixels_per_cell, cells_per_block, orientations
        )
        logger.info("HOGExtractor initialized: feature_size=%d", self.feature_size)

    def extract(
        self,
        image: np.ndarray,
        visualize: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract HOG features from a single image.

        Args:
            image:     Grayscale uint8 or float image.
            visualize: Return HOG visualization image.

        Returns:
            (feature_vector, hog_image_or_None)
        """
        return extract_hog(
            image,
            image_size=self.image_size,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=visualize,
        )

    def extract_batch(
        self,
        images: List[np.ndarray],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Extract HOG features from a list of images.

        Args:
            images:        List of grayscale uint8 arrays.
            show_progress: Show tqdm progress bar.

        Returns:
            Float64 array of shape (N, feature_size).
        """
        iterator = images
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(images, desc="HOG extraction")

        features = np.zeros((len(images), self.feature_size), dtype=np.float64)
        for i, img in enumerate(iterator):
            feat, _ = self.extract(img)
            features[i] = feat
        return features

    def extract_from_tensor(self, tensor) -> np.ndarray:
        """
        Extract HOG features from a PyTorch tensor batch.

        Args:
            tensor: Float tensor of shape (N, 1, H, W) in [-1, 1] range.

        Returns:
            Float64 array of shape (N, feature_size).
        """
        import torch
        if isinstance(tensor, torch.Tensor):
            # Denormalize [-1, 1] → [0, 255] uint8
            arr = ((tensor.squeeze(1).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        else:
            arr = tensor
        return self.extract_batch(list(arr))

    def get_hog_image(self, image: np.ndarray) -> np.ndarray:
        """
        Return HOG visualization for a single image.

        The returned image can be displayed alongside the original to
        show which gradient directions are dominant.

        Args:
            image: Grayscale uint8 image.

        Returns:
            Float array suitable for matplotlib imshow.
        """
        _, hog_img = self.extract(image, visualize=True)
        from skimage import exposure
        return exposure.rescale_intensity(hog_img, in_range=(0, 10))
