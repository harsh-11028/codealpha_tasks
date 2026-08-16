"""
EasyOCR engine wrapper.

Provides a clean interface around the EasyOCR library for recognizing
text in natural scenes and semi-printed handwriting.

EasyOCR is best used when:
  - Input contains mixed printed + handwritten text
  - Character segmentation is unreliable (connected cursive writing)
  - Fast, high-accuracy word-level recognition is needed
"""

from __future__ import annotations

import logging
import ssl
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# Fix macOS Python framework SSL verification issues when downloading EasyOCR models
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

logger = logging.getLogger(__name__)

# Lazy import & caching — defer EasyOCR reader initialization until first use
_readers: Dict[Tuple[str, bool], Any] = {}


def _get_reader(languages: List[str] = None, gpu: bool = False):
    """Lazy-initialize and cache EasyOCR reader per language combination."""
    langs = languages or ["en"]
    key = (",".join(sorted(langs)), gpu)
    if key not in _readers:
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader (languages=%s, gpu=%s) ...", langs, gpu)
            models_dir = Path(__file__).resolve().parent.parent / "saved_models" / "easyocr"
            user_net_dir = models_dir / "user_network"
            models_dir.mkdir(parents=True, exist_ok=True)
            user_net_dir.mkdir(parents=True, exist_ok=True)
            _readers[key] = easyocr.Reader(
                langs,
                gpu=gpu,
                verbose=False,
                model_storage_directory=str(models_dir),
                user_network_directory=str(user_net_dir),
            )
            logger.info("EasyOCR ready.")
        except ImportError:
            logger.error("easyocr not installed. Run: pip install easyocr")
            raise
    return _readers[key]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class EasyOCRResult:
    """
    Structured result from EasyOCR detection.

    Attributes:
        bbox:       List of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        text:       Recognized text string.
        confidence: Recognition confidence [0, 1].
    """

    __slots__ = ("bbox", "text", "confidence")

    def __init__(self, bbox, text: str, confidence: float) -> None:
        self.bbox = bbox
        self.text = text
        self.confidence = confidence

    @property
    def x_min(self) -> int:
        return int(min(pt[0] for pt in self.bbox))

    @property
    def y_min(self) -> int:
        return int(min(pt[1] for pt in self.bbox))

    @property
    def x_max(self) -> int:
        return int(max(pt[0] for pt in self.bbox))

    @property
    def y_max(self) -> int:
        return int(max(pt[1] for pt in self.bbox))

    def to_dict(self) -> dict:
        return {
            "bbox": {"x": self.x_min, "y": self.y_min,
                     "w": self.x_max - self.x_min, "h": self.y_max - self.y_min},
            "text": self.text,
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class EasyOCREngine:
    """
    Wrapper around EasyOCR for handwritten and printed text recognition.

    Usage:
        engine = EasyOCREngine(languages=["en"], gpu=False)
        results = engine.read_image(image_array)
        full_text = engine.extract_text(image_array)
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        confidence_threshold: float = 0.10,
        paragraph: bool = False,
    ) -> None:
        """
        Args:
            languages:            List of language codes (e.g., ["en"]).
            gpu:                  Use GPU for EasyOCR (requires CUDA).
            confidence_threshold: Discard detections below this confidence.
            paragraph:            Group text into paragraphs.
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.paragraph = paragraph
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            _get_reader(self.languages, self.gpu)
            self._initialized = True

    def read_image(
        self,
        image: np.ndarray,
        detail: int = 1,
    ) -> List[EasyOCRResult]:
        """
        Run EasyOCR on an image array.

        Args:
            image:  BGR or grayscale uint8 numpy array.
            detail: 0 = text only, 1 = text + bbox + confidence.

        Returns:
            List of EasyOCRResult sorted top-to-bottom, left-to-right.
        """
        self._ensure_initialized()
        reader = _get_reader(self.languages, self.gpu)

        raw = reader.readtext(
            image,
            detail=detail,
            paragraph=self.paragraph,
            mag_ratio=1.5,
            width_ths=0.7,
            slope_ths=0.2,
        )

        results: List[EasyOCRResult] = []
        for item in raw:
            if detail == 0:
                results.append(EasyOCRResult(bbox=[[0,0],[0,0],[0,0],[0,0]],
                                             text=str(item), confidence=1.0))
            else:
                bbox, text, conf = item
                if conf >= self.confidence_threshold:
                    results.append(EasyOCRResult(bbox=bbox, text=str(text), confidence=float(conf)))

        # Sort top-to-bottom, then left-to-right
        results.sort(key=lambda r: (r.y_min, r.x_min))
        return results

    def extract_text(
        self,
        image: np.ndarray,
        join_char: str = " ",
    ) -> Tuple[str, float]:
        """
        Extract all text from an image as a single string.

        Args:
            image:     Input image array.
            join_char: Character used to join detected text segments.

        Returns:
            (full_text, mean_confidence)
        """
        results = self.read_image(image)
        if not results:
            return "", 0.0

        texts = [r.text for r in results]
        confs = [r.confidence for r in results]
        full_text = join_char.join(texts)
        mean_conf = sum(confs) / len(confs)
        return full_text, mean_conf

    def extract_words(
        self,
        image: np.ndarray,
    ) -> List[dict]:
        """
        Extract individual words with bounding boxes.

        Returns:
            List of dicts: {text, confidence, bbox: {x, y, w, h}}
        """
        results = self.read_image(image)
        output = []
        for r in results:
            for word in r.text.split():
                output.append({
                    "text": word,
                    "confidence": r.confidence,
                    "bbox": {"x": r.x_min, "y": r.y_min,
                             "w": r.x_max - r.x_min, "h": r.y_max - r.y_min},
                })
        return output

    def is_available(self) -> bool:
        """Check if EasyOCR is importable."""
        try:
            import easyocr
            return True
        except ImportError:
            return False
