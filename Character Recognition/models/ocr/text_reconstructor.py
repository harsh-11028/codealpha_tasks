"""
Text reconstructor — assembles characters into words and sentences.

Takes the output of segmentation + character recognition and rebuilds
the original text by:
  1. Sorting characters spatially (left-to-right, top-to-bottom)
  2. Grouping characters into words based on inter-character gap analysis
  3. Grouping words into lines
  4. Joining lines into the final sentence output

Also applies light post-processing:
  - Remove non-printable characters
  - Collapse repeated spaces
  - Apply basic spell-correction hints (optional)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial sorting helpers
# ---------------------------------------------------------------------------

def sort_chars_reading_order(
    chars: list,
    line_height_tolerance: float = 0.5,
) -> list:
    """
    Sort character regions into reading order (left-to-right, top-to-bottom).

    Characters are first grouped into lines using vertical overlap,
    then sorted left-to-right within each line.

    Args:
        chars:                 List of objects with x_min, y_min, y_max attributes.
        line_height_tolerance: Fraction of char height to use as line grouping tolerance.

    Returns:
        Characters sorted in reading order.
    """
    if not chars:
        return []

    avg_height = np.mean([getattr(c, "height", 20) for c in chars])
    tolerance = avg_height * line_height_tolerance

    # Sort by y_min first
    sorted_y = sorted(chars, key=lambda c: getattr(c, "y_min", 0))

    # Group into lines
    lines: List[List] = []
    current_line = [sorted_y[0]]
    current_y = getattr(sorted_y[0], "y_min", 0)

    for char in sorted_y[1:]:
        char_y = getattr(char, "y_min", 0)
        if abs(char_y - current_y) <= tolerance:
            current_line.append(char)
        else:
            lines.append(sorted(current_line, key=lambda c: getattr(c, "x_min", 0)))
            current_line = [char]
            current_y = char_y

    if current_line:
        lines.append(sorted(current_line, key=lambda c: getattr(c, "x_min", 0)))

    return [char for line in lines for char in line]


def compute_word_gaps(chars: list) -> List[float]:
    """
    Compute horizontal gaps between consecutive characters.

    Args:
        chars: Left-to-right sorted character list (with x_min, x_max).

    Returns:
        List of gap values (len = len(chars) - 1).
    """
    gaps: List[float] = []
    for i in range(1, len(chars)):
        prev_x_max = getattr(chars[i - 1], "x_max", 0)
        curr_x_min = getattr(chars[i], "x_min", 0)
        gaps.append(max(0.0, float(curr_x_min - prev_x_max)))
    return gaps


def detect_word_boundaries(
    chars: list,
    multiplier: float = 2.5,
) -> List[int]:
    """
    Detect indices where a word boundary exists.

    A word boundary is detected when the inter-character gap exceeds
    `multiplier` × the median gap.

    Args:
        chars:      Sorted character list.
        multiplier: Gap threshold multiplier (tune based on handwriting style).

    Returns:
        List of indices (in `chars`) where a new word starts.
    """
    if len(chars) <= 1:
        return []

    gaps = compute_word_gaps(chars)
    valid_gaps = [g for g in gaps if g > 0]
    if not valid_gaps:
        return []

    # For short sequences (< 4 gaps), use minimum gap as baseline character spacing so a large word space does not skew the median
    baseline_gap = float(np.min(valid_gaps) if len(valid_gaps) < 4 else np.median(valid_gaps))
    threshold = baseline_gap * multiplier

    boundaries: List[int] = []
    for i, gap in enumerate(gaps):
        if gap > threshold:
            boundaries.append(i + 1)  # i+1 = start of new word

    return boundaries


# ---------------------------------------------------------------------------
# Text reconstruction
# ---------------------------------------------------------------------------

class TextReconstructor:
    """
    Assembles OCR character predictions into coherent text output.

    Handles:
      - Spatial sorting into reading order
      - Word gap detection
      - Line grouping
      - Post-processing and cleanup
    """

    def __init__(
        self,
        word_gap_multiplier: float = 2.5,
        line_height_tolerance: float = 0.5,
        min_confidence_to_include: float = 0.0,
        apply_cleanup: bool = True,
    ) -> None:
        """
        Args:
            word_gap_multiplier:      Gap multiplier for word boundary detection.
            line_height_tolerance:    Line grouping vertical tolerance fraction.
            min_confidence_to_include: Skip characters below this confidence.
            apply_cleanup:            Apply post-processing cleanup.
        """
        self.word_gap_multiplier = word_gap_multiplier
        self.line_height_tolerance = line_height_tolerance
        self.min_confidence_to_include = min_confidence_to_include
        self.apply_cleanup = apply_cleanup

    def reconstruct_from_chars(
        self,
        char_regions: list,
    ) -> Tuple[str, float]:
        """
        Build text from a flat list of recognized character regions.

        Each item in `char_regions` must have:
          - .char: str (recognized character)
          - .confidence: float
          - .x_min, .x_max, .y_min, .y_max: int

        Args:
            char_regions: List of CharRegion objects from char_segmentor.

        Returns:
            (reconstructed_text, mean_confidence)
        """
        # Filter low-confidence characters
        chars = [
            c for c in char_regions
            if getattr(c, "confidence", 1.0) >= self.min_confidence_to_include
            and getattr(c, "char", "").strip()
        ]

        if not chars:
            return "", 0.0

        # Sort into reading order
        chars = sort_chars_reading_order(chars, self.line_height_tolerance)

        # Detect word boundaries
        boundaries = detect_word_boundaries(chars, self.word_gap_multiplier)
        boundary_set = set(boundaries)

        # Build text
        text_parts: List[str] = []
        for i, ch in enumerate(chars):
            if i in boundary_set:
                text_parts.append(" ")
            char_str = getattr(ch, "char", "?")
            text_parts.append(char_str)

        raw_text = "".join(text_parts)
        mean_conf = float(np.mean([
            getattr(c, "confidence", 1.0) for c in chars
        ]))

        if self.apply_cleanup:
            raw_text = self._cleanup(raw_text)

        return raw_text, mean_conf

    def reconstruct_from_words(
        self,
        word_regions: list,
        line_height_tolerance: float = 0.5,
    ) -> Tuple[str, float]:
        """
        Build text from a list of recognized word regions.

        Each word must have:
          - .text: str
          - .confidence: float
          - .x_min, .y_min: int

        Lines are reconstructed by grouping words by vertical position.

        Args:
            word_regions: List of WordRegion objects with .text filled.
            line_height_tolerance: Fraction of avg word height to group into lines.

        Returns:
            (reconstructed_text, mean_confidence)
        """
        if not word_regions:
            return "", 0.0

        words = [w for w in word_regions if getattr(w, "text", "").strip()]
        if not words:
            return "", 0.0

        # Sort by position
        words.sort(key=lambda w: (getattr(w, "y_min", 0), getattr(w, "x_min", 0)))

        # Group into lines
        avg_height = np.mean([
            getattr(w, "height", 20) for w in words if hasattr(w, "height")
        ]) or 20
        tolerance = avg_height * line_height_tolerance

        lines: List[List] = []
        current_line = [words[0]]
        current_y = getattr(words[0], "y_min", 0)

        for word in words[1:]:
            word_y = getattr(word, "y_min", 0)
            if abs(word_y - current_y) <= tolerance:
                current_line.append(word)
            else:
                lines.append(sorted(current_line, key=lambda w: getattr(w, "x_min", 0)))
                current_line = [word]
                current_y = word_y
        if current_line:
            lines.append(sorted(current_line, key=lambda w: getattr(w, "x_min", 0)))

        # Build text
        line_texts = [
            " ".join(getattr(w, "text", "") for w in line)
            for line in lines
        ]
        full_text = "\n".join(line_texts)

        mean_conf = float(np.mean([
            getattr(w, "confidence", 1.0) for w in words
        ]))

        if self.apply_cleanup:
            full_text = self._cleanup(full_text)

        return full_text, mean_conf

    def merge_engine_outputs(
        self,
        custom_text: str,
        custom_conf: float,
        easyocr_text: str,
        easyocr_conf: float,
        tesseract_text: str,
        tesseract_conf: float,
        strategy: str = "best_confidence",
    ) -> Tuple[str, float, str]:
        """
        Merge outputs from multiple OCR engines into a single result.

        Strategies:
          - 'best_confidence': Pick the engine with highest confidence.
          - 'custom_first':    Use custom model; fall back to EasyOCR or Tesseract.
          - 'voting':          Majority vote on word level (experimental).

        Args:
            custom_text, custom_conf:       Custom model output.
            easyocr_text, easyocr_conf:     EasyOCR output.
            tesseract_text, tesseract_conf: Tesseract output.
            strategy:                       Merge strategy name.

        Returns:
            (merged_text, confidence, engine_name)
        """
        candidates = [
            (custom_text,    custom_conf,    "custom"),
            (easyocr_text,   easyocr_conf,   "easyocr"),
            (tesseract_text, tesseract_conf, "tesseract"),
        ]
        # Filter empty results
        candidates = [(t, c, e) for t, c, e in candidates if t.strip()]

        if not candidates:
            return "", 0.0, "none"

        if strategy == "best_confidence":
            best = max(candidates, key=lambda x: x[1])
            return best

        elif strategy == "custom_first":
            for text, conf, engine in candidates:
                if engine == "custom" and conf >= 0.5:
                    return text, conf, engine
            return candidates[0]

        elif strategy == "voting":
            # Word-level majority vote
            all_words: List[List[str]] = [t.split() for t, c, e in candidates]
            max_len = max(len(w) for w in all_words) if all_words else 0
            voted_words: List[str] = []
            for i in range(max_len):
                word_votes: Dict[str, int] = {}
                for words in all_words:
                    if i < len(words):
                        word = words[i].strip()
                        word_votes[word] = word_votes.get(word, 0) + 1
                if word_votes:
                    voted_words.append(max(word_votes, key=word_votes.get))
            merged = " ".join(voted_words)
            mean_conf = float(np.mean([c for _, c, _ in candidates]))
            return merged, mean_conf, "voting"

        return candidates[0]

    @staticmethod
    def _cleanup(text: str) -> str:
        """
        Apply light post-processing to OCR output.

        Operations:
          - Remove non-printable control characters
          - Normalize Unicode to NFC
          - Collapse multiple spaces
          - Strip leading/trailing whitespace per line
        """
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)
        # Remove control characters (except newline and tab)
        text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
        # Strip each line
        lines = [line.strip() for line in text.split("\n")]
        # Remove empty consecutive lines
        cleaned_lines: List[str] = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    cleaned_lines.append("")
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        return "\n".join(cleaned_lines).strip()

    def compute_confidence_summary(
        self,
        char_regions: list,
    ) -> Dict[str, float]:
        """
        Compute confidence statistics for the frontend display.

        Returns:
            Dict with overall, min, max, low_conf_fraction.
        """
        confs = [
            float(getattr(c, "confidence", 1.0))
            for c in char_regions
        ]
        if not confs:
            return {"overall": 0.0, "min": 0.0, "max": 0.0, "low_conf_fraction": 0.0}

        low_threshold = 0.60
        return {
            "overall": round(float(np.mean(confs)), 4),
            "min": round(float(np.min(confs)), 4),
            "max": round(float(np.max(confs)), 4),
            "low_conf_fraction": round(
                sum(c < low_threshold for c in confs) / len(confs), 4
            ),
        }
