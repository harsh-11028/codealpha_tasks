"""
Text line detection for handwritten document images.

Detects horizontal text lines in a preprocessed binary image using
horizontal projection profiles (row-sum histograms).

Pipeline:
    binary image
        → horizontal projection (sum each row)
        → valley finding (between lines)
        → line bounding boxes
        → optional refinement with morphological dilation

Returns a list of LineRegion objects, each containing the bounding box
of a single text line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LineRegion:
    """
    Bounding box for a single detected text line.

    Attributes:
        y_min:   Top row of the line (inclusive).
        y_max:   Bottom row of the line (inclusive).
        x_min:   Left column (may be tighter than full image width).
        x_max:   Right column.
        words:   Will be populated by WordDetector downstream.
    """
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    words: List = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    def as_bbox(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) bounding box for OpenCV drawing."""
        return self.x_min, self.y_min, self.width, self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the line region from the full document image."""
        return image[self.y_min:self.y_max, self.x_min:self.x_max]


# ---------------------------------------------------------------------------
# Horizontal projection
# ---------------------------------------------------------------------------

def compute_horizontal_projection(binary_image: np.ndarray) -> np.ndarray:
    """
    Compute horizontal projection profile (row-sum histogram).

    Each value in the output represents the count of foreground (255)
    pixels in the corresponding row.

    Args:
        binary_image: Binary uint8 array where text = 255, background = 0.

    Returns:
        1-D float array of shape (H,) with row pixel counts.
    """
    return binary_image.sum(axis=1).astype(np.float64) / 255.0


def smooth_projection(projection: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Smooth the projection profile with a Gaussian kernel.

    Smoothing reduces noise spikes that would create false line boundaries
    in images with slight variations in ink density.

    Args:
        projection: Raw row-sum histogram.
        sigma:      Gaussian smoothing standard deviation.

    Returns:
        Smoothed projection array.
    """
    kernel_size = max(3, int(sigma * 4) | 1)
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    return cv2.filter2D(projection.reshape(-1, 1), -1, kernel).flatten()


# ---------------------------------------------------------------------------
# Valley / boundary detection
# ---------------------------------------------------------------------------

def find_line_boundaries(
    projection: np.ndarray,
    min_gap: int = 5,
    threshold_ratio: float = 0.05,
) -> List[Tuple[int, int]]:
    """
    Identify row ranges corresponding to text lines.

    Strategy:
        1. Threshold the projection to separate 'text rows' from 'gap rows'.
        2. Find contiguous runs of text rows.
        3. Merge runs that are separated by fewer than min_gap gap rows.

    Args:
        projection:       Smoothed horizontal projection.
        min_gap:          Minimum vertical gap (rows) between two lines.
        threshold_ratio:  Fraction of the max projection used as threshold.
                          Rows below this value are considered gaps.

    Returns:
        List of (y_start, y_end) tuples for each detected line.
    """
    threshold = projection.max() * threshold_ratio
    in_line = (projection > threshold).astype(int)

    # Detect transitions
    diff = np.diff(in_line, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return []

    # Merge close segments
    merged_starts: List[int] = [starts[0]]
    merged_ends: List[int] = []

    for i in range(1, len(starts)):
        if starts[i] - ends[i - 1] <= min_gap:
            # Merge with previous segment
            pass
        else:
            merged_ends.append(ends[i - 1])
            merged_starts.append(starts[i])
    merged_ends.append(ends[-1])

    return list(zip(merged_starts, merged_ends))


# ---------------------------------------------------------------------------
# Morphological refinement
# ---------------------------------------------------------------------------

def refine_line_boundaries_morph(
    binary_image: np.ndarray,
    boundaries: List[Tuple[int, int]],
    padding: int = 3,
) -> List[Tuple[int, int]]:
    """
    Refine line boundaries using morphological dilation to connect
    broken strokes near line boundaries.

    Also applies vertical padding so characters at line edges are not clipped.

    Args:
        binary_image: Full document binary image.
        boundaries:   Initial (y_start, y_end) boundaries.
        padding:      Extra pixels added above and below each boundary.

    Returns:
        Padded and refined boundaries.
    """
    h, w = binary_image.shape
    refined: List[Tuple[int, int]] = []
    for y_start, y_end in boundaries:
        y_start_padded = max(0, y_start - padding)
        y_end_padded = min(h - 1, y_end + padding)
        refined.append((y_start_padded, y_end_padded))
    return refined


# ---------------------------------------------------------------------------
# Main line detector class
# ---------------------------------------------------------------------------

class LineDetector:
    """
    Detects text lines in a binary handwritten document image.

    Usage:
        detector = LineDetector()
        lines = detector.detect(binary_image)
        for line in lines:
            line_crop = line.crop(binary_image)
    """

    def __init__(
        self,
        min_line_height: int = 10,
        min_gap_rows: int = 5,
        threshold_ratio: float = 0.05,
        projection_sigma: float = 2.0,
        padding: int = 3,
    ) -> None:
        """
        Args:
            min_line_height:  Discard lines shorter than this in pixels.
            min_gap_rows:     Merge lines closer than this gap.
            threshold_ratio:  Fraction of max projection to threshold.
            projection_sigma: Gaussian smoothing sigma for projection.
            padding:          Extra rows added to each line boundary.
        """
        self.min_line_height = min_line_height
        self.min_gap_rows = min_gap_rows
        self.threshold_ratio = threshold_ratio
        self.projection_sigma = projection_sigma
        self.padding = padding

    def detect(self, binary_image: np.ndarray) -> List[LineRegion]:
        """
        Detect text lines in the given binary image.

        Args:
            binary_image: uint8 array, text=255 on black background.

        Returns:
            List of LineRegion objects sorted top-to-bottom.
        """
        if binary_image.ndim != 2:
            raise ValueError("Expected a 2-D (grayscale/binary) image.")

        h, w = binary_image.shape

        # 1. Horizontal projection
        projection = compute_horizontal_projection(binary_image)

        # 2. Smooth
        smoothed = smooth_projection(projection, self.projection_sigma)

        # 3. Find boundaries
        boundaries = find_line_boundaries(
            smoothed,
            min_gap=self.min_gap_rows,
            threshold_ratio=self.threshold_ratio,
        )

        if not boundaries:
            logger.warning("LineDetector: no lines detected.")
            return []

        # 4. Refine with padding
        boundaries = refine_line_boundaries_morph(binary_image, boundaries, self.padding)

        # 5. Build LineRegion objects
        lines: List[LineRegion] = []
        for y_start, y_end in boundaries:
            if (y_end - y_start) < self.min_line_height:
                logger.debug("Skipping short line segment h=%d", y_end - y_start)
                continue

            # Compute horizontal extent (tight bounding box)
            line_strip = binary_image[y_start:y_end, :]
            col_proj = line_strip.sum(axis=0)
            nonzero_cols = np.where(col_proj > 0)[0]
            if len(nonzero_cols) == 0:
                continue
            x_min = max(0, nonzero_cols[0] - self.padding)
            x_max = min(w - 1, nonzero_cols[-1] + self.padding)

            lines.append(LineRegion(
                y_min=y_start,
                y_max=y_end,
                x_min=x_min,
                x_max=x_max,
            ))

        logger.info("LineDetector: detected %d lines.", len(lines))
        return lines

    def visualize(
        self,
        image: np.ndarray,
        lines: List[LineRegion],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detected line bounding boxes on the image.

        Args:
            image:     BGR or grayscale image (will be converted to BGR).
            lines:     Detected LineRegion objects.
            color:     BGR color for the bounding boxes.
            thickness: Line thickness in pixels.

        Returns:
            BGR image with drawn line bounding boxes.
        """
        if image.ndim == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for i, line in enumerate(lines):
            x, y, w, h = line.as_bbox()
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                vis, f"L{i}", (x + 2, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )
        return vis

    def get_projection_plot_data(
        self,
        binary_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return projection profile data for plotting / visualization in frontend.

        Returns:
            (projection, smoothed_projection) as 1-D arrays.
        """
        raw = compute_horizontal_projection(binary_image)
        smoothed = smooth_projection(raw, self.projection_sigma)
        return raw, smoothed
