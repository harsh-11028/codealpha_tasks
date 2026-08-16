"""
Word detection within text lines.

Given a single text line image (a strip), this module detects individual
word bounding boxes using vertical projection profiles and morphological
dilation to bridge the gaps within a word while separating words from
each other.

Pipeline:
    line strip (binary)
        → horizontal dilation (connect chars within a word)
        → vertical projection (column-sum histogram)
        → valley finding → word bounding boxes
        → optional contour-based refinement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WordRegion:
    """
    Bounding box for a single detected word within a line.

    All coordinates are relative to the FULL document image (not the line).

    Attributes:
        x_min, y_min:  Top-left corner.
        x_max, y_max:  Bottom-right corner.
        chars:         Will be populated by CharSegmentor downstream.
        text:          Recognized text (filled in after OCR).
        confidence:    Recognition confidence [0, 1].
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    chars: List = field(default_factory=list)
    text: str = ""
    confidence: float = 0.0

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def as_bbox(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) OpenCV-style bounding box."""
        return self.x_min, self.y_min, self.width, self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop word region from full document image."""
        return image[self.y_min:self.y_max, self.x_min:self.x_max]

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for API responses."""
        return {
            "x": self.x_min,
            "y": self.y_min,
            "w": self.width,
            "h": self.height,
            "text": self.text,
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Vertical projection
# ---------------------------------------------------------------------------

def compute_vertical_projection(binary_strip: np.ndarray) -> np.ndarray:
    """
    Compute vertical projection profile (column-sum histogram).

    Args:
        binary_strip: Binary uint8 line strip, text=255.

    Returns:
        1-D float array of shape (W,) with column pixel counts.
    """
    return binary_strip.sum(axis=0).astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# Morphological word grouping
# ---------------------------------------------------------------------------

def group_chars_into_words(
    binary_strip: np.ndarray,
    dilation_width: Optional[int] = None,
    dilation_height: Optional[int] = None,
) -> np.ndarray:
    """
    Dilate the binary strip horizontally to bridge character gaps within words.

    The dilation width is adaptive: it's estimated from the median character
    width in the line, scaled by a factor to be smaller than the inter-word gap.

    Args:
        binary_strip:    Binary uint8 line image (text=255).
        dilation_width:  Override dilation kernel width.
        dilation_height: Override dilation kernel height (usually 1).

    Returns:
        Dilated binary image where each word forms a connected blob.
    """
    h, w = binary_strip.shape

    # Estimate adaptive dilation width
    if dilation_width is None:
        # Rough estimate: dilation = ~10% of image width, min 5 pixels
        dilation_width = max(5, int(w * 0.10))

    if dilation_height is None:
        dilation_height = max(3, h // 4)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (dilation_width, dilation_height),
    )
    return cv2.dilate(binary_strip, kernel, iterations=1)


# ---------------------------------------------------------------------------
# Word bounding box extraction
# ---------------------------------------------------------------------------

def extract_word_boxes_from_contours(
    dilated: np.ndarray,
    original_strip: np.ndarray,
    line_y_offset: int,
    line_x_offset: int,
    min_word_width: int = 5,
    min_word_height: int = 5,
    padding: int = 2,
) -> List[WordRegion]:
    """
    Find contours in the dilated image and map them back to word regions
    in the original (undilated) strip, translated to full-image coordinates.

    Args:
        dilated:         Horizontally dilated binary strip.
        original_strip:  Original (undilated) line strip for tight bounds.
        line_y_offset:   Y coordinate of the line's top in the full image.
        line_x_offset:   X coordinate of the line's left in the full image.
        min_word_width:  Minimum width to keep a detected word box.
        min_word_height: Minimum height to keep a detected word box.
        padding:         Extra pixels added on each side.

    Returns:
        List of WordRegion sorted left-to-right.
    """
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    words: List[WordRegion] = []
    h_strip, w_strip = original_strip.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_word_width or h < min_word_height:
            continue

        # Tighten bounding box using original (undilated) strip
        x_start = max(0, x - padding)
        x_end = min(w_strip, x + w + padding)
        y_start = max(0, y - padding)
        y_end = min(h_strip, y + h + padding)

        # Translate to full-image coordinates
        words.append(WordRegion(
            x_min=line_x_offset + x_start,
            y_min=line_y_offset + y_start,
            x_max=line_x_offset + x_end,
            y_max=line_y_offset + y_end,
        ))

    # Sort left to right
    words.sort(key=lambda r: r.x_min)
    return words


# ---------------------------------------------------------------------------
# Main word detector class
# ---------------------------------------------------------------------------

class WordDetector:
    """
    Detects word bounding boxes within individual text line strips.

    Usage:
        from models.segmentation.line_detector import LineDetector

        line_detector = LineDetector()
        word_detector = WordDetector()

        lines = line_detector.detect(binary_image)
        for line in lines:
            line_strip = line.crop(binary_image)
            words = word_detector.detect(line_strip, line.y_min, line.x_min)
            line.words = words
    """

    def __init__(
        self,
        dilation_width: Optional[int] = None,
        dilation_height: Optional[int] = None,
        min_word_width: int = 5,
        min_word_height: int = 5,
        padding: int = 2,
    ) -> None:
        """
        Args:
            dilation_width:  Horizontal dilation kernel width (adaptive if None).
            dilation_height: Vertical dilation kernel height (adaptive if None).
            min_word_width:  Minimum word width in pixels.
            min_word_height: Minimum word height in pixels.
            padding:         Extra margin around each word box.
        """
        self.dilation_width = dilation_width
        self.dilation_height = dilation_height
        self.min_word_width = min_word_width
        self.min_word_height = min_word_height
        self.padding = padding

    def detect(
        self,
        line_strip: np.ndarray,
        line_y_offset: int = 0,
        line_x_offset: int = 0,
    ) -> List[WordRegion]:
        """
        Detect word bounding boxes in a single line strip.

        Args:
            line_strip:     Binary uint8 strip for one text line.
            line_y_offset:  Y offset of this line in the full document.
            line_x_offset:  X offset of this line in the full document.

        Returns:
            List of WordRegion objects sorted left-to-right.
        """
        if line_strip.size == 0:
            return []

        # Ensure binary
        if line_strip.max() <= 1:
            line_strip = (line_strip * 255).astype(np.uint8)

        dilated = group_chars_into_words(
            line_strip,
            dilation_width=self.dilation_width,
            dilation_height=self.dilation_height,
        )

        words = extract_word_boxes_from_contours(
            dilated,
            line_strip,
            line_y_offset=line_y_offset,
            line_x_offset=line_x_offset,
            min_word_width=self.min_word_width,
            min_word_height=self.min_word_height,
            padding=self.padding,
        )

        logger.debug(
            "WordDetector: %d words detected in line at y=%d.",
            len(words), line_y_offset,
        )
        return words

    def detect_all_lines(
        self,
        binary_image: np.ndarray,
        lines,  # List[LineRegion]
    ) -> None:
        """
        Detect words for all lines and populate line.words in-place.

        Args:
            binary_image: Full binary document image.
            lines:        List of LineRegion objects from LineDetector.
        """
        for line in lines:
            strip = line.crop(binary_image)
            line.words = self.detect(strip, line.y_min, line.x_min)

        total_words = sum(len(l.words) for l in lines)
        logger.info(
            "WordDetector: %d words detected across %d lines.",
            total_words, len(lines),
        )

    def visualize(
        self,
        image: np.ndarray,
        words: List[WordRegion],
        color: Tuple[int, int, int] = (255, 165, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw word bounding boxes on the image.

        Args:
            image:     BGR or grayscale image.
            words:     List of detected WordRegion objects.
            color:     BGR bounding box color.
            thickness: Line thickness.

        Returns:
            BGR image with drawn word bounding boxes.
        """
        if image.ndim == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for i, word in enumerate(words):
            x, y, w, h = word.as_bbox()
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                vis, f"W{i}", (x + 2, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )
        return vis
