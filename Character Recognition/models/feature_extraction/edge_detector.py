"""
Edge detection module for OCR feature extraction.

Extracts structural edge information from character images using:
  - Canny edge detection (multi-scale)
  - Sobel gradient maps (horizontal, vertical, magnitude)
  - Laplacian of Gaussian (LoG)

Edge features capture stroke boundaries independent of ink thickness,
making them robust across different handwriting styles and pen pressures.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canny edge detection
# ---------------------------------------------------------------------------

def detect_canny_edges(
    image: np.ndarray,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    aperture_size: int = 3,
    l2_gradient: bool = True,
    auto_threshold: bool = True,
) -> np.ndarray:
    """
    Detect edges using Canny edge detector.

    When auto_threshold=True, thresholds are computed automatically
    using Otsu's method on the gradient magnitude — no manual tuning needed.

    Args:
        image:          Grayscale uint8 image.
        low_threshold:  Lower hysteresis threshold (overrides auto).
        high_threshold: Upper hysteresis threshold (overrides auto).
        aperture_size:  Sobel kernel size (3, 5, or 7).
        l2_gradient:    Use L2 norm for gradient (more accurate).
        auto_threshold: Automatically compute thresholds from image.

    Returns:
        Binary uint8 edge map (edges=255, background=0).
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if auto_threshold and (low_threshold is None or high_threshold is None):
        # Otsu's threshold on gradient magnitude
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        otsu_val, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        high_threshold = int(otsu_val)
        low_threshold = int(otsu_val * 0.4)  # standard 2.5:1 ratio

    return cv2.Canny(
        image,
        low_threshold or 50,
        high_threshold or 150,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )


def detect_multiscale_edges(
    image: np.ndarray,
    scales: Tuple[float, ...] = (1.0, 0.75, 0.5),
) -> np.ndarray:
    """
    Detect edges at multiple scales and combine via OR.

    Multi-scale detection captures both fine strokes and broader
    structural edges that single-scale Canny might miss.

    Args:
        image:  Grayscale uint8 image.
        scales: Relative scales to analyze.

    Returns:
        Combined binary uint8 edge map.
    """
    h, w = image.shape
    combined = np.zeros((h, w), dtype=np.uint8)

    for scale in scales:
        if scale == 1.0:
            scaled = image
        else:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        edges = detect_canny_edges(scaled, auto_threshold=True)

        # Resize edges back to original size
        if scale != 1.0:
            edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_NEAREST)

        combined = cv2.bitwise_or(combined, edges)

    return combined


# ---------------------------------------------------------------------------
# Sobel gradient
# ---------------------------------------------------------------------------

def compute_sobel_gradients(
    image: np.ndarray,
    kernel_size: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Compute Sobel gradient maps.

    Args:
        image:       Grayscale uint8 image.
        kernel_size: Sobel kernel size (1, 3, 5, or 7).

    Returns:
        Dict with keys:
            'x':         Horizontal gradient (float32)
            'y':         Vertical gradient (float32)
            'magnitude': Gradient magnitude (uint8)
            'direction': Gradient direction in degrees (float32)
    """
    img = image.astype(np.float32)

    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=kernel_size)

    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude_uint8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    direction = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

    return {
        "x": sobel_x,
        "y": sobel_y,
        "magnitude": magnitude_uint8,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# Laplacian of Gaussian
# ---------------------------------------------------------------------------

def detect_log_edges(
    image: np.ndarray,
    sigma: float = 1.0,
    threshold: int = 10,
) -> np.ndarray:
    """
    Laplacian of Gaussian edge detector.

    LoG is effective at detecting thin, fine strokes in handwriting
    because it combines smoothing and second-derivative edge detection.

    Args:
        image:     Grayscale uint8 image.
        sigma:     Gaussian smoothing sigma.
        threshold: Absolute LoG value threshold for edge classification.

    Returns:
        Binary uint8 edge map.
    """
    # Gaussian blur
    kernel_size = max(3, int(sigma * 6) | 1)
    blurred = cv2.GaussianBlur(image.astype(np.float32), (kernel_size, kernel_size), sigma)

    # Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)

    # Zero-crossing with threshold
    edges = (np.abs(laplacian) > threshold).astype(np.uint8) * 255
    return edges


# ---------------------------------------------------------------------------
# Feature vector from edge map
# ---------------------------------------------------------------------------

class EdgeDetector:
    """
    Computes edge-based feature vectors from character images.

    Supports Canny, Sobel, and LoG — can be combined for richer features.

    Usage:
        detector = EdgeDetector(method='canny')
        features = detector.extract_features(image)
    """

    def __init__(
        self,
        method: str = "canny",
        image_size: Tuple[int, int] = (32, 32),
        use_multiscale: bool = False,
    ) -> None:
        """
        Args:
            method:         'canny' | 'sobel' | 'log' | 'combined'
            image_size:     Resize images to this before extraction.
            use_multiscale: Use multi-scale Canny edge detection.
        """
        self.method = method
        self.image_size = image_size
        self.use_multiscale = use_multiscale

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Ensure grayscale uint8 of correct size."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        h, w = self.image_size
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Run edge detection and return the edge map.

        Args:
            image: Grayscale uint8 image.

        Returns:
            Binary uint8 edge map of shape self.image_size.
        """
        img = self._preprocess(image)

        if self.method == "canny":
            if self.use_multiscale:
                return detect_multiscale_edges(img)
            return detect_canny_edges(img, auto_threshold=True)

        elif self.method == "sobel":
            grads = compute_sobel_gradients(img)
            _, binary = cv2.threshold(
                grads["magnitude"], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary

        elif self.method == "log":
            return detect_log_edges(img)

        elif self.method == "combined":
            canny = detect_canny_edges(img, auto_threshold=True)
            log = detect_log_edges(img)
            return cv2.bitwise_or(canny, log)

        else:
            raise ValueError(f"Unknown edge detection method: {self.method!r}")

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a flat edge feature vector.

        Args:
            image: Grayscale uint8 image.

        Returns:
            Float64 1-D array (flattened normalized edge map).
        """
        edge_map = self.detect(image)
        return edge_map.astype(np.float64).flatten() / 255.0

    def get_gradient_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract combined Sobel gradient magnitude + direction features.

        Returns a 2× feature vector (magnitude + direction concatenated).

        Args:
            image: Grayscale uint8 image.

        Returns:
            Float64 1-D array.
        """
        img = self._preprocess(image)
        grads = compute_sobel_gradients(img)
        mag = grads["magnitude"].astype(np.float64).flatten() / 255.0
        direction = grads["direction"].astype(np.float64).flatten() / 360.0
        return np.concatenate([mag, direction])
