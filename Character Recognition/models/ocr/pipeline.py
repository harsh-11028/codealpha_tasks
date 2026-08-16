"""
Full OCR pipeline orchestrator.

Connects all pipeline stages into a single callable:

    Raw Image
        → Preprocessing (image_processor)
        → Line Detection  (line_detector)
        → Word Detection  (word_detector)
        → Char Segmentation (char_segmentor)
        → Model Inference  (OCRPredictor / EasyOCR / Tesseract)
        → Text Reconstruction (text_reconstructor)
        → Final Output

The pipeline automatically selects the best engine combination
based on image characteristics (image size, confidence scores, etc.)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.training.config import config as default_config, Config
from models.ocr.text_reconstructor import TextReconstructor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

def _sanitize_python_types(val: Any) -> Any:
    """Recursively convert NumPy scalars and arrays to native Python types for Pydantic serialization."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return [_sanitize_python_types(v) for v in val.tolist()]
    elif isinstance(val, dict):
        return {str(k): _sanitize_python_types(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_sanitize_python_types(item) for item in val]
    return val


@dataclass
class OCRPipelineResult:
    """
    Full output from the OCR pipeline.

    Attributes:
        text:             Final recognized text.
        confidence:       Overall confidence [0, 1].
        words:            List of word-level dicts {text, confidence, bbox}.
        chars:            List of char-level dicts {char, confidence, bbox}.
        line_boxes:       List of line bounding boxes {x, y, w, h}.
        word_boxes:       List of word bounding boxes {x, y, w, h, text}.
        char_boxes:       List of char bounding boxes {x, y, w, h, char}.
        processing_ms:    Total pipeline time in milliseconds.
        engine_used:      Engine(s) used: 'custom', 'easyocr', 'tesseract', 'hybrid'.
        model_used:       Model name used.
        confidence_stats: Dict with overall/min/max/low_conf_fraction.
        annotated_image:  BGR image with bounding boxes drawn (base64 PNG).
    """
    text: str = ""
    confidence: float = 0.0
    words: List[Dict] = field(default_factory=list)
    chars: List[Dict] = field(default_factory=list)
    line_boxes: List[Dict] = field(default_factory=list)
    word_boxes: List[Dict] = field(default_factory=list)
    char_boxes: List[Dict] = field(default_factory=list)
    processing_ms: float = 0.0
    engine_used: str = "custom"
    model_used: str = ""
    confidence_stats: Dict = field(default_factory=dict)
    annotated_image: Optional[str] = None

    def __post_init__(self) -> None:
        self.confidence = float(self.confidence) if self.confidence is not None else 0.0
        self.processing_ms = float(self.processing_ms) if self.processing_ms is not None else 0.0
        self.words = _sanitize_python_types(self.words)
        self.chars = _sanitize_python_types(self.chars)
        self.line_boxes = _sanitize_python_types(self.line_boxes)
        self.word_boxes = _sanitize_python_types(self.word_boxes)
        self.char_boxes = _sanitize_python_types(self.char_boxes)
        self.confidence_stats = _sanitize_python_types(self.confidence_stats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "words": self.words,
            "chars": self.chars,
            "line_boxes": self.line_boxes,
            "word_boxes": self.word_boxes,
            "char_boxes": self.char_boxes,
            "processing_ms": round(self.processing_ms, 2),
            "engine_used": self.engine_used,
            "model_used": self.model_used,
            "confidence_stats": self.confidence_stats,
            "annotated_image": self.annotated_image,
        }


# ---------------------------------------------------------------------------
# Image quality assessment (for engine routing)
# ---------------------------------------------------------------------------

def _assess_image_quality(image: np.ndarray) -> Dict[str, float]:
    """
    Compute quick image quality metrics to inform engine selection.

    Returns:
        Dict with: sharpness, contrast, noise_level, text_density
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Sharpness via Laplacian variance
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Contrast via standard deviation
    contrast = float(gray.std())

    # Noise level estimate (high-freq component)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.abs(gray.astype(np.float32) - blurred.astype(np.float32)).mean())

    # Text density (fraction of dark pixels after binarization)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_density = float(binary.sum() / (255 * binary.size))

    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "noise_level": noise,
        "text_density": text_density,
    }


def _select_engine(quality: Dict[str, float], task: str) -> str:
    """
    Select OCR engine based on image quality and task type.

    Args:
        quality: Quality metrics from _assess_image_quality.
        task:    'character' | 'word' | 'sentence'

    Returns:
        Engine name: 'custom' | 'easyocr' | 'tesseract' | 'hybrid'
    """
    if task == "character":
        return "custom"  # Always use custom model for isolated chars

    # For word/sentence: route based on image characteristics and handwriting specialization
    if quality["sharpness"] > 500 and quality["text_density"] < 0.15:
        # Clean, sparse handwritten document — EasyOCR performs best with Latin handwriting models
        return "easyocr"
    elif quality["contrast"] > 60:
        # Good contrast — EasyOCR for natural scene and handwritten text
        return "easyocr"
    else:
        # Difficult handwriting — use hybrid
        return "hybrid"


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class OCRPipeline:
    """
    End-to-end OCR pipeline for handwritten text recognition.

    Orchestrates preprocessing → segmentation → inference → reconstruction
    and supports routing between custom models, EasyOCR, and Tesseract.

    Usage:
        pipeline = OCRPipeline(cfg)
        pipeline.initialize()
        result = pipeline.run(image_bytes, task='sentence')
    """

    def __init__(self, cfg: Config = default_config) -> None:
        self.cfg = cfg
        self._initialized = False

        # Lazy-loaded components
        self._predictor = None
        self._easyocr = None
        self._tesseract = None
        self._reconstructor = TextReconstructor(
            word_gap_multiplier=2.5,
            apply_cleanup=True,
        )

    def initialize(self) -> None:
        """Load all OCR engines and models."""
        from models.training.predict import OCRPredictor
        from models.ocr.easyocr_engine import EasyOCREngine
        from models.ocr.tesseract_engine import TesseractEngine
        from models.training.config import config as cfg

        logger.info("Initializing OCR pipeline ...")

        self._predictor = OCRPredictor(self.cfg)
        self._predictor.load()

        if self.cfg.inference.easyocr_enabled:
            self._easyocr = EasyOCREngine(
                languages=self.cfg.inference.easyocr_languages,
                gpu=self.cfg.inference.easyocr_gpu,
            )

        if self.cfg.inference.tesseract_enabled:
            self._tesseract = TesseractEngine()

        self._initialized = True
        logger.info("OCR pipeline ready.")

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, source) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (color_image, binary_image) from any image source.
        """
        from models.preprocessing.image_processor import (
            load_image, to_grayscale, remove_noise,
            enhance_contrast_clahe, binarize, morphological_close
        )
        from models.preprocessing.deskew import deskew_image

        img = load_image(source)
        if img.ndim == 2:
            color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            color = img.copy()

        gray = to_grayscale(img)
        gray = remove_noise(gray, self.cfg.preprocessing.denoise_kernel_size)
        gray = enhance_contrast_clahe(
            gray,
            self.cfg.preprocessing.clahe_clip_limit,
            self.cfg.preprocessing.clahe_tile_grid,
        )
        if self.cfg.preprocessing.deskew:
            gray = deskew_image(gray, max_angle=self.cfg.preprocessing.deskew_angle_limit)
        binary = binarize(gray, self.cfg.preprocessing.binarize_method)
        binary = morphological_close(binary, self.cfg.preprocessing.morph_kernel_size)

        return color, binary

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate_image(
        self,
        color_image: np.ndarray,
        line_boxes: List[Dict],
        word_boxes: List[Dict],
        char_boxes: List[Dict],
    ) -> str:
        """Draw bounding boxes on image and return as base64 PNG."""
        import base64
        vis = color_image.copy()

        # Lines (green)
        for b in line_boxes:
            cv2.rectangle(vis, (b["x"], b["y"]),
                         (b["x"] + b["w"], b["y"] + b["h"]), (0, 200, 0), 2)

        # Words (orange)
        for b in word_boxes:
            cv2.rectangle(vis, (b["x"], b["y"]),
                         (b["x"] + b["w"], b["y"] + b["h"]), (0, 165, 255), 2)
            if b.get("text"):
                cv2.putText(vis, b["text"][:8], (b["x"], b["y"] - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # Chars (red)
        for b in char_boxes:
            cv2.rectangle(vis, (b["x"], b["y"]),
                         (b["x"] + b["w"], b["y"] + b["h"]), (0, 0, 220), 1)

        _, buf = cv2.imencode(".png", vis)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------
    # Task-specific runners
    # ------------------------------------------------------------------

    def run_character(self, source, model_name: str = "auto") -> OCRPipelineResult:
        """Run single-character OCR."""
        self._ensure_initialized()
        t_start = time.perf_counter()

        result = self._predictor.predict_character(source, model_name, top_k=5)

        return OCRPipelineResult(
            text=result.text,
            confidence=result.confidence,
            chars=[{"char": result.text, "confidence": result.confidence}],
            processing_ms=(time.perf_counter() - t_start) * 1000,
            engine_used=result.engine_used,
            model_used=result.model_used,
            confidence_stats={"overall": result.confidence, "min": result.confidence,
                              "max": result.confidence, "low_conf_fraction": 0.0},
        )

    def run_word(self, source, model_name: str = "auto", engine: str = "auto") -> OCRPipelineResult:
        """Run word-level OCR."""
        self._ensure_initialized()
        t_start = time.perf_counter()

        color, binary = self._preprocess(source)
        quality = _assess_image_quality(color)
        selected_engine = engine if engine != "auto" else _select_engine(quality, "word")

        text, confidence, engine_used = "", 0.0, selected_engine

        if selected_engine == "custom":
            result = self._predictor.predict_word(source, model_name)
            text, confidence = result.text, result.confidence
        elif selected_engine == "easyocr" and self._easyocr:
            try:
                text, confidence = self._easyocr.extract_text(color)
                engine_used = "easyocr"
            except Exception as exc:
                logger.warning("EasyOCR word extraction failed (%s); falling back to custom engine.", exc)
                result = self._predictor.predict_word(source, model_name)
                text, confidence = result.text, result.confidence
                engine_used = "custom (fallback)"
        elif selected_engine == "tesseract" and self._tesseract:
            try:
                text, confidence = self._tesseract.predict_word(binary)
                engine_used = "tesseract"
            except Exception as exc:
                logger.warning("Tesseract word extraction failed (%s); falling back to custom engine.", exc)
                result = self._predictor.predict_word(source, model_name)
                text, confidence = result.text, result.confidence
                engine_used = "custom (fallback)"
        elif selected_engine == "hybrid":
            text, confidence, engine_used = self._run_hybrid_word(source, color, binary, model_name)
        else:
            result = self._predictor.predict_word(source, model_name)
            text, confidence = result.text, result.confidence
            engine_used = "custom"

        return OCRPipelineResult(
            text=text,
            confidence=confidence,
            words=[{"text": text, "confidence": confidence}],
            processing_ms=(time.perf_counter() - t_start) * 1000,
            engine_used=engine_used,
            model_used=model_name,
            confidence_stats={"overall": confidence, "min": confidence,
                              "max": confidence, "low_conf_fraction": 0.0},
        )

    def _run_hybrid_word(self, source, color, binary, model_name):
        """Run all engines and merge via best-confidence strategy."""
        custom_text, custom_conf = "", 0.0
        easy_text, easy_conf = "", 0.0
        tess_text, tess_conf = "", 0.0

        try:
            r = self._predictor.predict_word(source, model_name)
            custom_text, custom_conf = r.text, r.confidence
        except Exception:
            pass

        if self._easyocr:
            try:
                easy_text, easy_conf = self._easyocr.extract_text(color)
            except Exception:
                pass

        if self._tesseract:
            try:
                tess_text, tess_conf = self._tesseract.predict_word(binary)
            except Exception:
                pass

        text, conf, engine = self._reconstructor.merge_engine_outputs(
            custom_text, custom_conf, easy_text, easy_conf, tess_text, tess_conf,
            strategy="best_confidence",
        )
        return text, conf, f"hybrid({engine})"

    def run_sentence(
        self,
        source,
        engine: str = "auto",
        annotate: bool = True,
    ) -> OCRPipelineResult:
        """
        Run full sentence-level OCR with line + word + char detection.
        """
        self._ensure_initialized()
        t_start = time.perf_counter()

        from models.segmentation.line_detector import LineDetector
        from models.segmentation.word_detector import WordDetector
        from models.segmentation.char_segmentor import CharSegmentor

        color, binary = self._preprocess(source)
        quality = _assess_image_quality(color)
        selected_engine = engine if engine != "auto" else _select_engine(quality, "sentence")

        # --- Segmentation ---
        line_detector = LineDetector(
            min_line_height=self.cfg.inference.min_char_area,
        )
        word_detector = WordDetector()
        char_segmentor = CharSegmentor()

        lines = line_detector.detect(binary)
        word_detector.detect_all_lines(binary, lines)
        char_segmentor.segment_all_words(binary, lines)

        # --- Collect bounding boxes ---
        line_boxes = [
            {"x": int(l.x_min), "y": int(l.y_min), "w": int(l.width), "h": int(l.height)}
            for l in lines
        ]
        word_boxes_list = []
        char_boxes_list = []
        all_word_regions = []
        all_char_regions = []

        for line in lines:
            for word in line.words:
                word_boxes_list.append({
                    "x": int(word.x_min), "y": int(word.y_min),
                    "w": int(word.width), "h": int(word.height),
                    "text": "",
                })
                all_word_regions.append(word)
                for char in word.chars:
                    char_boxes_list.append({
                        "x": int(char.x_min), "y": int(char.y_min),
                        "w": int(char.width), "h": int(char.height),
                        "char": "",
                    })
                    all_char_regions.append(char)

        # --- Recognition ---
        engine_used = "custom"
        text, confidence = "", 0.0
        success = False

        if selected_engine == "tesseract" and self._tesseract and self._tesseract.available:
            try:
                text, confidence = self._tesseract.extract_text(binary)
                engine_used = "tesseract"
                success = True
            except Exception as exc:
                logger.warning("Tesseract sentence extraction failed (%s); trying fallback engines.", exc)

        has_custom_models = bool(self._predictor and getattr(self._predictor, "loaded_models", []))
        if not success and self._easyocr and (selected_engine in ("easyocr", "hybrid", "tesseract", "auto") or (selected_engine == "custom" and not has_custom_models)):
            try:
                text, confidence = self._easyocr.extract_text(color, join_char="\n")
                engine_used = "easyocr" if selected_engine == "easyocr" else f"easyocr ({selected_engine} routing)"
                success = True
            except Exception as exc:
                logger.warning("EasyOCR sentence extraction failed (%s); trying custom engine.", exc)
                engine_used = "custom (fallback)"

        if not success:
            # Custom: predict each word
            for word_region, wb in zip(all_word_regions, word_boxes_list):
                word_strip = word_region.crop(binary)
                try:
                    wr = self._predictor.predict_word(word_strip)
                    word_region.text = wr.text
                    word_region.confidence = wr.confidence
                    wb["text"] = wr.text
                except Exception:
                    word_region.text = ""
                    word_region.confidence = 0.0

            text, confidence = self._reconstructor.reconstruct_from_words(all_word_regions)
            if "fallback" not in engine_used:
                engine_used = "custom"

        # --- Annotate ---
        annotated_b64 = None
        if annotate:
            annotated_b64 = self._annotate_image(
                color, line_boxes, word_boxes_list, char_boxes_list
            )

        confidence_stats = {"overall": confidence, "min": confidence,
                            "max": confidence, "low_conf_fraction": 0.0}

        return OCRPipelineResult(
            text=text,
            confidence=confidence,
            line_boxes=line_boxes,
            word_boxes=word_boxes_list,
            char_boxes=char_boxes_list,
            processing_ms=(time.perf_counter() - t_start) * 1000,
            engine_used=engine_used,
            model_used="auto",
            confidence_stats=confidence_stats,
            annotated_image=annotated_b64,
        )

    def run(
        self,
        source,
        task: str = "sentence",
        model_name: str = "auto",
        engine: str = "auto",
        annotate: bool = True,
    ) -> OCRPipelineResult:
        """
        Unified entry point for the OCR pipeline.

        Args:
            source:     Image source (bytes, path, PIL, numpy).
            task:       'character' | 'word' | 'sentence'
            model_name: Custom model name or 'auto'.
            engine:     Force engine or 'auto' for smart routing.
            annotate:   Attach annotated image to result.

        Returns:
            OCRPipelineResult with full output.
        """
        self._ensure_initialized()
        if task == "character":
            return self.run_character(source, model_name)
        elif task == "word":
            return self.run_word(source, model_name, engine)
        else:
            return self.run_sentence(source, engine, annotate)


_default_pipeline: Optional[OCRPipeline] = None


def get_pipeline(cfg: Config = default_config) -> OCRPipeline:
    """Return a singleton instance of OCRPipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = OCRPipeline(cfg)
        _default_pipeline.initialize()
    return _default_pipeline

