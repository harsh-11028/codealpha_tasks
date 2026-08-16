"""
Character segmentation within word images.

Given a word strip (binary image), this module:
  1. Applies connected component analysis to find character blobs.
  2. Handles touching / overlapping characters via vertical projection valleys.
  3. Handles broken characters by merging nearby components.
  4. Returns sorted, normalized character ROIs ready for model inference.

Two complementary strategies are used:
  - Primary: Connected Component Analysis (CCA)
  - Fallback: Vertical Projection Profile segmentation (for touching chars)
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
class CharRegion:
    """
    Bounding box for a single character within a word.

    All coordinates are relative to the FULL document image.

    Attributes:
        x_min, y_min:  Top-left corner.
        x_max, y_max:  Bottom-right corner.
        char:          Recognized character (filled after inference).
        confidence:    Recognition confidence [0, 1].
        component_id:  Connected component label (for debugging).
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    char: str = ""
    confidence: float = 0.0
    component_id: int = -1

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    def as_bbox(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) for OpenCV drawing."""
        return self.x_min, self.y_min, self.width, self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop this character from the full document image."""
        return image[self.y_min:self.y_max, self.x_min:self.x_max]

    def to_dict(self) -> dict:
        return {
            "x": self.x_min,
            "y": self.y_min,
            "w": self.width,
            "h": self.height,
            "char": self.char,
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Connected component analysis
# ---------------------------------------------------------------------------

def extract_chars_via_cca(
    binary_word: np.ndarray,
    word_x_offset: int,
    word_y_offset: int,
    min_area: int = 20,
    max_area_ratio: float = 0.95,
    aspect_ratio_range: Tuple[float, float] = (0.05, 5.0),
    padding: int = 1,
) -> List[CharRegion]:
    """
    Extract character bounding boxes using connected component analysis.

    Filters out noise (too small), full-word blobs (too large),
    and obviously non-character shapes (extreme aspect ratios).

    Args:
        binary_word:      Binary uint8 word strip (text=255).
        word_x_offset:    X offset of the word in the full document.
        word_y_offset:    Y offset of the word in the full document.
        min_area:         Minimum contour area in pixels².
        max_area_ratio:   Components larger than this fraction of total
                          word area are considered full blobs (skip).
        aspect_ratio_range: (min, max) accepted aspect ratios.
        padding:          Extra pixels around each bounding box.

    Returns:
        List of CharRegion sorted left-to-right.
    """
    h, w = binary_word.shape
    total_area = h * w

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_word, connectivity=8
    )

    chars: List[CharRegion] = []
    for label_id in range(1, num_labels):  # 0 is background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if area > total_area * max_area_ratio:
            logger.debug("CCA: skipping oversized component area=%d", area)
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        cw = stats[label_id, cv2.CC_STAT_WIDTH]
        ch = stats[label_id, cv2.CC_STAT_HEIGHT]

        ratio = cw / max(ch, 1)
        if not (aspect_ratio_range[0] <= ratio <= aspect_ratio_range[1]):
            logger.debug("CCA: skipping component with aspect_ratio=%.2f", ratio)
            continue

        chars.append(CharRegion(
            x_min=word_x_offset + max(0, x - padding),
            y_min=word_y_offset + max(0, y - padding),
            x_max=word_x_offset + min(w, x + cw + padding),
            y_max=word_y_offset + min(h, y + ch + padding),
            component_id=label_id,
        ))

    chars.sort(key=lambda c: c.x_min)
    return chars


# ---------------------------------------------------------------------------
# Touching character splitting via vertical projection valleys
# ---------------------------------------------------------------------------

def split_touching_chars(
    char_region: CharRegion,
    binary_word: np.ndarray,
    word_x_offset: int,
    word_y_offset: int,
    min_valley_depth_ratio: float = 0.2,
    min_char_width: int = 5,
) -> List[CharRegion]:
    """
    Attempt to split a wide component that may contain touching characters.

    Uses vertical projection profile: finds valleys (local minima) that
    indicate potential split points between touching characters.

    Args:
        char_region:             The potentially-touching CharRegion.
        binary_word:             Full binary word strip.
        word_x_offset:           Word X offset in document.
        word_y_offset:           Word Y offset in document.
        min_valley_depth_ratio:  Valley must be this fraction below the
                                 adjacent peaks to be considered a split.
        min_char_width:          Minimum width of a sub-character.

    Returns:
        List of split CharRegions, or the original list if no split was made.
    """
    # Extract the region from binary word (local coordinates)
    lx = char_region.x_min - word_x_offset
    ly = char_region.y_min - word_y_offset
    rx = char_region.x_max - word_x_offset
    ry = char_region.y_max - word_y_offset

    strip = binary_word[
        max(0, ly):min(binary_word.shape[0], ry),
        max(0, lx):min(binary_word.shape[1], rx),
    ]
    if strip.size == 0:
        return [char_region]

    # Vertical projection of this component
    v_proj = strip.sum(axis=0).astype(np.float64)
    if v_proj.max() == 0:
        return [char_region]

    v_proj_norm = v_proj / v_proj.max()

    # Find valleys: points where projection drops significantly
    valleys: List[int] = []
    half_w = len(v_proj_norm) // 2
    for x_local in range(1, len(v_proj_norm) - 1):
        left_max = v_proj_norm[:x_local].max() if x_local > 0 else 0
        right_max = v_proj_norm[x_local + 1:].max() if x_local < len(v_proj_norm) - 1 else 0
        local_val = v_proj_norm[x_local]
        if (
            local_val < left_max * (1 - min_valley_depth_ratio)
            and local_val < right_max * (1 - min_valley_depth_ratio)
        ):
            valleys.append(x_local)

    # Cluster valleys (keep one valley per gap region)
    if not valleys:
        return [char_region]

    clustered_valleys: List[int] = [valleys[0]]
    for v in valleys[1:]:
        if v - clustered_valleys[-1] > 3:
            clustered_valleys.append(v)

    # Build split boundaries
    boundaries = [0] + clustered_valleys + [len(v_proj_norm)]
    result: List[CharRegion] = []
    for i in range(len(boundaries) - 1):
        x_start = boundaries[i]
        x_end = boundaries[i + 1]
        if (x_end - x_start) < min_char_width:
            continue
        result.append(CharRegion(
            x_min=char_region.x_min + x_start,
            y_min=char_region.y_min,
            x_max=char_region.x_min + x_end,
            y_max=char_region.y_max,
            component_id=char_region.component_id,
        ))

    return result if len(result) > 1 else [char_region]


# ---------------------------------------------------------------------------
# Broken character merging
# ---------------------------------------------------------------------------

def merge_broken_chars(
    chars: List[CharRegion],
    gap_threshold_ratio: float = 0.3,
) -> List[CharRegion]:
    """
    Merge adjacent CharRegions that are suspiciously close together.

    Broken strokes (e.g., dotted 'i', crossed 't') may produce multiple
    components for a single character. This heuristic merges them
    when the horizontal gap between adjacent components is very small
    relative to the smaller component's width.

    Args:
        chars:               Left-to-right sorted list of CharRegions.
        gap_threshold_ratio: Gap / smaller_width must be below this to merge.

    Returns:
        Merged list of CharRegions.
    """
    if len(chars) <= 1:
        return chars

    merged: List[CharRegion] = [chars[0]]
    for current in chars[1:]:
        prev = merged[-1]
        gap = current.x_min - prev.x_max
        smaller_width = min(prev.width, current.width)

        if smaller_width > 0 and (gap <= 0 or (gap <= 3 and gap / smaller_width < gap_threshold_ratio)):
            # Merge
            merged[-1] = CharRegion(
                x_min=prev.x_min,
                y_min=min(prev.y_min, current.y_min),
                x_max=current.x_max,
                y_max=max(prev.y_max, current.y_max),
                component_id=prev.component_id,
            )
        else:
            merged.append(current)

    return merged


# ---------------------------------------------------------------------------
# Main character segmentor class
# ---------------------------------------------------------------------------

class CharSegmentor:
    """
    Segments individual characters from word images.

    Combines CCA, touching-character splitting, and broken-character merging
    for robust segmentation of both printed and cursive handwriting.

    Usage:
        segmentor = CharSegmentor()
        chars = segmentor.segment(word_image, word.x_min, word.y_min)
        for ch in chars:
            char_crop = ch.crop(full_binary_image)
    """

    def __init__(
        self,
        min_char_area: int = 20,
        aspect_ratio_range: Tuple[float, float] = (0.05, 5.0),
        padding: int = 1,
        split_touching: bool = True,
        merge_broken: bool = True,
        max_touching_width_ratio: float = 1.8,
        gap_merge_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            min_char_area:          Minimum blob area for a character.
            aspect_ratio_range:     (min, max) accepted aspect ratio.
            padding:                Extra pixels around each character box.
            split_touching:         Attempt to split wide components.
            merge_broken:           Attempt to merge nearby tiny components.
            max_touching_width_ratio: Components wider than this factor of
                                    the median char width trigger splitting.
            gap_merge_threshold:    Merge threshold ratio for broken chars.
        """
        self.min_char_area = min_char_area
        self.aspect_ratio_range = aspect_ratio_range
        self.padding = padding
        self.split_touching = split_touching
        self.merge_broken = merge_broken
        self.max_touching_width_ratio = max_touching_width_ratio
        self.gap_merge_threshold = gap_merge_threshold

    def segment(
        self,
        binary_word: np.ndarray,
        word_x_offset: int = 0,
        word_y_offset: int = 0,
    ) -> List[CharRegion]:
        """
        Segment characters from a binary word strip.

        Args:
            binary_word:    Binary uint8 word strip (text=255).
            word_x_offset:  X coordinate of this word in the full image.
            word_y_offset:  Y coordinate of this word in the full image.

        Returns:
            List of CharRegion sorted left-to-right.
        """
        if binary_word.size == 0:
            return []

        # Normalize to binary 0/255
        if binary_word.max() <= 1:
            binary_word = (binary_word * 255).astype(np.uint8)

        # Step 1: CCA
        chars = extract_chars_via_cca(
            binary_word,
            word_x_offset,
            word_y_offset,
            min_area=self.min_char_area,
            aspect_ratio_range=self.aspect_ratio_range,
            padding=self.padding,
        )

        if not chars:
            return []

        # Step 2: Merge broken characters (e.g., dotted 'i', dashed 't')
        if self.merge_broken:
            chars = merge_broken_chars(chars, self.gap_merge_threshold)

        # Step 3: Split touching characters
        if self.split_touching:
            median_width = float(np.median([c.width for c in chars]))
            split_chars: List[CharRegion] = []
            for ch in chars:
                if median_width > 0 and ch.width > median_width * self.max_touching_width_ratio:
                    split_result = split_touching_chars(
                        ch, binary_word, word_x_offset, word_y_offset
                    )
                    split_chars.extend(split_result)
                else:
                    split_chars.append(ch)
            chars = split_chars

        # Final sort
        chars.sort(key=lambda c: c.x_min)

        logger.debug(
            "CharSegmentor: %d characters at word offset (%d, %d).",
            len(chars), word_x_offset, word_y_offset,
        )
        return chars

    def segment_all_words(
        self,
        binary_image: np.ndarray,
        lines,  # List[LineRegion]
    ) -> None:
        """
        Segment characters for all words in all lines in-place.

        Args:
            binary_image: Full binary document image.
            lines:        LineRegion list (words already populated).
        """
        total_chars = 0
        for line in lines:
            for word in line.words:
                word_strip = word.crop(binary_image)
                word.chars = self.segment(word_strip, word.x_min, word.y_min)
                total_chars += len(word.chars)

        logger.info(
            "CharSegmentor: %d characters across all words.", total_chars
        )

    def visualize(
        self,
        image: np.ndarray,
        chars: List[CharRegion],
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 1,
    ) -> np.ndarray:
        """
        Draw character bounding boxes on the image.

        Args:
            image:     BGR or grayscale image.
            chars:     List of CharRegion objects.
            color:     BGR box color.
            thickness: Line thickness.

        Returns:
            BGR image with drawn character bounding boxes.
        """
        if image.ndim == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        for i, ch in enumerate(chars):
            x, y, w, h = ch.as_bbox()
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            if ch.char:
                cv2.putText(
                    vis, ch.char, (x + 1, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                )
        return vis

    def normalize_char_image(
        self,
        char_crop: np.ndarray,
        target_size: Tuple[int, int] = (32, 32),
        padding_value: int = 0,
    ) -> np.ndarray:
        """
        Resize and center a character crop for model input.

        Preserves aspect ratio with padding to avoid distorting character shapes.

        Args:
            char_crop:     Cropped character image (uint8).
            target_size:   (H, W) output dimensions.
            padding_value: Background fill value.

        Returns:
            Normalized uint8 array of shape target_size.
        """
        from models.preprocessing.image_processor import resize_image
        return resize_image(char_crop, target_size, keep_aspect=True, padding_value=padding_value)
