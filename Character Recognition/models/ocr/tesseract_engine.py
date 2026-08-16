"""
Tesseract OCR engine wrapper.

Wraps pytesseract to provide document-level OCR with bounding box data.
Best used for structured documents, printed text, or when EasyOCR
confidence is low.

System requirement:
    macOS:   brew install tesseract
    Ubuntu:  sudo apt install tesseract-ocr
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _find_tesseract() -> Optional[str]:
    """Return path to tesseract binary or None if not found."""
    # Check environment variable override first
    env_path = os.getenv("TESSERACT_CMD")
    if env_path and Path(env_path).exists():
        return env_path

    # Search PATH
    which = shutil.which("tesseract")
    if which:
        return which

    # Common install locations
    common_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",            # macOS Apple Silicon
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_paths:
        if Path(path).exists():
            return path

    return None


def _setup_tesseract() -> bool:
    """Configure pytesseract and return True if Tesseract is available."""
    try:
        import pytesseract
        tess_cmd = _find_tesseract()
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
            logger.info("Tesseract found at: %s", tess_cmd)
            return True
        else:
            logger.warning(
                "Tesseract binary not found. "
                "Install with: brew install tesseract (macOS) or "
                "apt install tesseract-ocr (Ubuntu)"
            )
            return False
    except ImportError:
        logger.warning("pytesseract not installed. Run: pip install pytesseract")
        return False


# ---------------------------------------------------------------------------
# PSM / OEM configuration
# ---------------------------------------------------------------------------

# Page Segmentation Modes (PSM)
PSM_SINGLE_CHAR = 10     # Single character
PSM_SINGLE_WORD = 8      # Single word
PSM_SINGLE_LINE = 7      # Single line of text
PSM_SINGLE_BLOCK = 6     # Assume a single uniform block
PSM_AUTO = 3             # Auto page segmentation (default)

# OCR Engine Modes (OEM)
OEM_TESSERACT_ONLY = 0
OEM_LSTM_ONLY = 1
OEM_COMBINED = 2
OEM_DEFAULT = 3


# ---------------------------------------------------------------------------
# Result class
# ---------------------------------------------------------------------------

class TesseractWordResult:
    """Single word detected by Tesseract with position and confidence."""

    __slots__ = ("text", "confidence", "x", "y", "w", "h", "level")

    def __init__(
        self,
        text: str,
        confidence: float,
        x: int, y: int, w: int, h: int,
        level: int = 5,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.x, self.y, self.w, self.h = x, y, w, h
        self.level = level   # Tesseract hierarchy: 1=page,2=block,3=para,4=line,5=word

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence / 100.0, 4),
            "bbox": {"x": self.x, "y": self.y, "w": self.w, "h": self.h},
        }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class TesseractEngine:
    """
    Tesseract OCR engine for structured document text recognition.

    Usage:
        engine = TesseractEngine()
        if engine.available:
            text, conf = engine.extract_text(image)
            words = engine.extract_words(image)
    """

    def __init__(
        self,
        lang: str = "eng",
        psm: int = PSM_AUTO,
        oem: int = OEM_DEFAULT,
        confidence_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            lang:                 Tesseract language code (e.g., 'eng').
            psm:                  Page segmentation mode.
            oem:                  OCR engine mode.
            confidence_threshold: Filter words below this confidence (0–1).
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.confidence_threshold = confidence_threshold
        self.available = _setup_tesseract()

    @property
    def _config_string(self) -> str:
        return f"--psm {self.psm} --oem {self.oem}"

    def _image_to_pil(self, image: np.ndarray):
        """Convert numpy array to PIL Image."""
        from PIL import Image
        if image.ndim == 2:
            return Image.fromarray(image, mode="L")
        return Image.fromarray(image)

    def extract_text(
        self,
        image: np.ndarray,
        psm_override: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Run Tesseract on an image and return the recognized text.

        Args:
            image:        Grayscale or BGR uint8 numpy array.
            psm_override: Override the default PSM for this call.

        Returns:
            (text, confidence)  — confidence is mean word confidence [0, 1].
        """
        if not self.available:
            return "", 0.0

        import pytesseract
        pil_img = self._image_to_pil(image)
        config = self._config_string
        if psm_override is not None:
            config = f"--psm {psm_override} --oem {self.oem}"

        try:
            text = pytesseract.image_to_string(pil_img, lang=self.lang, config=config)
            data = pytesseract.image_to_data(
                pil_img, lang=self.lang, config=config,
                output_type=pytesseract.Output.DICT,
            )
            confs = [
                c / 100.0 for c in data["conf"]
                if isinstance(c, (int, float)) and int(c) > 0
            ]
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            return text.strip(), mean_conf
        except Exception as exc:
            logger.error("Tesseract extraction failed: %s", exc)
            return "", 0.0

    def extract_words(
        self,
        image: np.ndarray,
    ) -> List[TesseractWordResult]:
        """
        Extract individual words with bounding boxes and confidences.

        Args:
            image: Grayscale uint8 numpy array.

        Returns:
            List of TesseractWordResult sorted top-to-bottom, left-to-right.
        """
        if not self.available:
            return []

        import pytesseract
        pil_img = self._image_to_pil(image)
        try:
            data = pytesseract.image_to_data(
                pil_img, lang=self.lang, config=self._config_string,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.error("Tesseract word extraction failed: %s", exc)
            return []

        results: List[TesseractWordResult] = []
        n = len(data["text"])
        for i in range(n):
            text = str(data["text"][i]).strip()
            conf = data["conf"][i]
            if not text or not isinstance(conf, (int, float)) or int(conf) < 0:
                continue
            conf_norm = int(conf) / 100.0
            if conf_norm < self.confidence_threshold:
                continue
            results.append(TesseractWordResult(
                text=text,
                confidence=conf_norm,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                level=int(data["level"][i]),
            ))

        results.sort(key=lambda r: (r.y, r.x))
        return results

    def predict_char(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict a single character from an isolated character image.

        Uses PSM 10 (single character mode).

        Args:
            image: Grayscale character image.

        Returns:
            (character, confidence)
        """
        text, conf = self.extract_text(image, psm_override=PSM_SINGLE_CHAR)
        char = text.strip()[:1] if text.strip() else "?"
        return char, conf

    def predict_word(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict a single word from a word-strip image (PSM 8)."""
        text, conf = self.extract_text(image, psm_override=PSM_SINGLE_WORD)
        return text.strip(), conf

    def get_version(self) -> str:
        """Return Tesseract version string."""
        if not self.available:
            return "not available"
        try:
            import pytesseract
            return pytesseract.get_tesseract_version().vstring
        except Exception:
            return "unknown"
