"""
Image deskewing module.

Handwritten documents are often tilted or rotated. This module detects
the dominant text orientation angle and corrects it.

Two strategies:
  1. Hough Line Transform  — fast, works well on printed/semi-printed text.
  2. Projection Profile    — robust for handwriting; finds angle by maximizing
                             the variance of horizontal projection sums.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hough-based skew detection
# ---------------------------------------------------------------------------

def detect_skew_hough(
    binary_image: np.ndarray,
    angle_limit: float = 15.0,
) -> float:
    """
    Estimate document skew angle using probabilistic Hough line transform.

    Args:
        binary_image: Binary uint8 array (text = 255, background = 0).
        angle_limit:  Maximum expected skew in degrees. Lines outside this
                      range are ignored.

    Returns:
        Estimated skew angle in degrees (positive = counter-clockwise tilt).
    """
    # Run Hough on thin edges for speed
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=binary_image.shape[1] // 5,
        maxLineGap=20,
    )

    if lines is None or len(lines) == 0:
        logger.debug("Hough skew detection: no lines found, returning 0°.")
        return 0.0

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Filter to near-horizontal lines only
        if abs(angle) <= angle_limit:
            angles.append(angle)

    if not angles:
        return 0.0

    # Use median to be robust against outliers
    skew = float(np.median(angles))
    logger.debug("Hough skew detection: %d lines → skew = %.2f°", len(angles), skew)
    return skew


# ---------------------------------------------------------------------------
# Projection profile-based skew detection
# ---------------------------------------------------------------------------

def detect_skew_projection(
    binary_image: np.ndarray,
    angle_limit: float = 15.0,
    angle_step: float = 0.5,
) -> float:
    """
    Estimate skew angle by maximizing variance of horizontal projection profiles.

    Rotates the image through a range of candidate angles and scores each
    rotation by the variance of the row-sum histogram. The angle that
    produces the highest variance corresponds to the best horizontal alignment.

    Args:
        binary_image: Binary uint8 array (text = 255, background = 0).
        angle_limit:  Search ± this many degrees.
        angle_step:   Angular resolution of the search.

    Returns:
        Estimated skew angle in degrees.
    """
    h, w = binary_image.shape
    cx, cy = w / 2, h / 2

    angles = np.arange(-angle_limit, angle_limit + angle_step, angle_step)
    best_angle = 0.0
    best_score = -1.0

    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            binary_image, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        row_sums = rotated.sum(axis=1).astype(np.float32)
        score = float(row_sums.var())
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    logger.debug("Projection skew detection: best angle = %.2f°", best_angle)
    return best_angle


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------

def rotate_image(
    image: np.ndarray,
    angle: float,
    background_value: int = 0,
) -> np.ndarray:
    """
    Rotate an image by the given angle around its center.

    Args:
        image:            uint8 array.
        angle:            Rotation angle in degrees (positive = CCW).
        background_value: Fill value for areas outside original bounds.

    Returns:
        Rotated uint8 array of same shape.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_value,
    )
    return rotated


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def deskew_image(
    image: np.ndarray,
    method: str = "projection",
    max_angle: float = 15.0,
    angle_step: float = 0.5,
) -> np.ndarray:
    """
    Detect and correct skew in a binary or grayscale image.

    Args:
        image:      Binary or grayscale uint8 array.
        method:     'projection' (default, more robust) or 'hough'.
        max_angle:  Maximum skew angle to correct (degrees).
        angle_step: Angular resolution for projection search.

    Returns:
        Deskewed uint8 array of the same shape.
    """
    # Ensure binary for detection
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    is_binary = len(np.unique(image)) <= 2

    if not is_binary:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = image

    if method == "projection":
        angle = detect_skew_projection(binary, max_angle, angle_step)
    elif method == "hough":
        angle = detect_skew_hough(binary, max_angle)
    else:
        raise ValueError(f"Unknown deskew method: {method!r}")

    if abs(angle) < 0.1:
        return image  # no correction needed

    logger.info("Deskewing by %.2f° (method=%s)", angle, method)
    bg = 255 if is_binary else 0
    return rotate_image(image, angle, background_value=bg)


def get_skew_angle(
    image: np.ndarray,
    method: str = "projection",
    max_angle: float = 15.0,
) -> float:
    """
    Return the detected skew angle without applying correction.

    Useful for visualization or batch reporting.

    Args:
        image:     Binary or grayscale uint8 array.
        method:    'projection' or 'hough'.
        max_angle: Maximum skew angle to consider.

    Returns:
        Skew angle in degrees.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if method == "projection":
        return detect_skew_projection(binary, max_angle)
    return detect_skew_hough(binary, max_angle)
