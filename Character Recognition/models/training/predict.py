"""
Inference / prediction script.

Provides a clean API for running OCR on single images or batches.
Used by the FastAPI backend and can also be run standalone from CLI.

Usage:
    # Single image
    python -m models.training.predict --image path/to/img.png --task character

    # Directory of images
    python -m models.training.predict --dir path/to/images/ --task word

    # Webcam frame (base64 encoded)
    python -m models.training.predict --base64 <encoded_string>
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.training.config import config, Config, EMNIST_BALANCED_LABELS
from models.preprocessing.image_processor import ImagePreprocessor
from models.utils.model_selector import ModelSelector, resolve_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction result dataclass
# ---------------------------------------------------------------------------

class PredictionResult:
    """
    Container for OCR prediction output.

    Attributes:
        text:           Recognized text (char / word / sentence).
        confidence:     Top-1 confidence [0, 1].
        char_confidences: Per-character confidence (for word/sentence).
        top_k:          List of (label, confidence) for top-k predictions.
        processing_ms:  Total inference time in milliseconds.
        model_used:     Name of the model that produced the result.
        engine_used:    OCR engine used ('custom', 'easyocr', 'tesseract').
    """

    def __init__(
        self,
        text: str,
        confidence: float,
        char_confidences: Optional[List[float]] = None,
        top_k: Optional[List[Tuple[str, float]]] = None,
        processing_ms: float = 0.0,
        model_used: str = "",
        engine_used: str = "custom",
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.char_confidences = char_confidences or []
        self.top_k = top_k or []
        self.processing_ms = processing_ms
        self.model_used = model_used
        self.engine_used = engine_used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "char_confidences": [round(c, 4) for c in self.char_confidences],
            "top_k": [(label, round(conf, 4)) for label, conf in self.top_k],
            "processing_ms": round(self.processing_ms, 2),
            "model_used": self.model_used,
            "engine_used": self.engine_used,
        }

    def __repr__(self) -> str:
        return (
            f"PredictionResult(text={self.text!r}, "
            f"confidence={self.confidence:.2%}, "
            f"model={self.model_used})"
        )


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class OCRPredictor:
    """
    High-level OCR predictor used by the FastAPI backend.

    Wraps:
      - Preprocessing pipeline
      - Model selector
      - Label decoding

    Usage:
        predictor = OCRPredictor()
        predictor.load()
        result = predictor.predict_character(image_bytes)
        result = predictor.predict_word(image_bytes)
        result = predictor.predict_sentence(image_bytes)
    """

    def __init__(self, cfg: Config = config) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.training.device)
        self.preprocessor = ImagePreprocessor(cfg)
        self.selector = ModelSelector(cfg)
        self.label_map = EMNIST_BALANCED_LABELS
        self._loaded = False

    def load(self) -> None:
        """Load all available model checkpoints."""
        self.selector.load_all()
        self._loaded = True
        loaded = self.selector.get_loaded_models()
        logger.info("OCRPredictor loaded %d models: %s", len(loaded), loaded)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def predict_character(
        self,
        source,
        model_name: str = "auto",
        top_k: int = 5,
    ) -> PredictionResult:
        """
        Predict a single character from an image.

        Args:
            source:     Image source (bytes, path, PIL, numpy).
            model_name: Specific model or 'auto'.
            top_k:      Return top-k predictions.

        Returns:
            PredictionResult with text (single char), confidence, top_k.
        """
        self._ensure_loaded()
        t_start = time.perf_counter()

        # Preprocess
        tensor = self.preprocessor.preprocess_to_tensor(source, word_mode=False)

        # Inference
        logits, confidence = self.selector.predict_single(tensor, model_name, task="character")
        probs = F.softmax(logits.squeeze(0), dim=0)

        # Top prediction
        top_val, top_idx = probs.max(dim=0)
        predicted_char = self.label_map.get(int(top_idx.item()), "?")

        # Top-k predictions
        top_k_vals, top_k_idxs = probs.topk(min(top_k, len(probs)))
        top_k_list = [
            (self.label_map.get(int(idx.item()), "?"), float(val.item()))
            for idx, val in zip(top_k_idxs, top_k_vals)
        ]

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return PredictionResult(
            text=predicted_char,
            confidence=float(top_val.item()),
            top_k=top_k_list,
            processing_ms=elapsed_ms,
            model_used=model_name if model_name != "auto" else "auto",
            engine_used="custom",
        )

    def predict_word(
        self,
        source,
        model_name: str = "auto",
    ) -> PredictionResult:
        """
        Predict a word from an image using the CRNN model.

        Falls back to char-by-char prediction if CRNN is unavailable.

        Args:
            source:     Image source.
            model_name: Specific model or 'auto'.

        Returns:
            PredictionResult with word text and per-char confidences.
        """
        self._ensure_loaded()
        t_start = time.perf_counter()

        # Try CRNN (word-mode)
        if "crnn" in self.selector.get_loaded_models() and model_name in ("auto", "crnn"):
            return self._predict_crnn(source, t_start)

        # Fall back to segmentation + char prediction
        return self._predict_word_via_segmentation(source, t_start)

    def _predict_crnn(self, source, t_start: float) -> PredictionResult:
        """Run CRNN model on a word strip image."""
        tensor = self.preprocessor.preprocess_to_tensor(source, word_mode=True)
        tensor = tensor.to(self.device)

        from models.architectures.crnn import CRNN
        model = self.selector._models.get("crnn")
        if model is None:
            return self._predict_word_via_segmentation(source, t_start)

        model.eval()
        with torch.no_grad():
            log_probs = model(tensor)          # (T, 1, num_classes+1)
            sequences = model.decode_greedy(log_probs)  # List[List[int]]

        decoded_indices = sequences[0] if sequences else []
        # Build reverse char map (index → char)
        idx_to_char = {v: k for k, v in {
            **{str(i): i + 1 for i in range(10)},
            **{chr(ord("A") + i): i + 11 for i in range(26)},
        }.items()}
        # Use EMNIST label map for decoding
        word_text = "".join(
            self.label_map.get(idx - 1, "") for idx in decoded_indices if idx > 0
        )

        # Confidence: mean max prob across time steps
        probs = log_probs.exp()
        conf = float(probs.max(dim=2)[0].mean().item())

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return PredictionResult(
            text=word_text,
            confidence=conf,
            processing_ms=elapsed_ms,
            model_used="crnn",
            engine_used="custom",
        )

    def _predict_word_via_segmentation(self, source, t_start: float) -> PredictionResult:
        """Segment word into chars and predict each individually."""
        from models.preprocessing.image_processor import load_image, to_grayscale, binarize
        from models.segmentation.char_segmentor import CharSegmentor

        img = load_image(source)
        gray = to_grayscale(img)
        binary = binarize(gray, method="otsu")

        segmentor = CharSegmentor()
        chars = segmentor.segment(binary)

        word_text = ""
        char_confs: List[float] = []

        for char_region in chars:
            crop = char_region.crop(binary)
            try:
                result = self.predict_character(crop)
                word_text += result.text
                char_confs.append(result.confidence)
            except Exception:
                word_text += "?"
                char_confs.append(0.0)

        confidence = float(np.mean(char_confs)) if char_confs else 0.0
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return PredictionResult(
            text=word_text,
            confidence=confidence,
            char_confidences=char_confs,
            processing_ms=elapsed_ms,
            model_used="segmentation+char_model",
            engine_used="custom",
        )

    def predict_sentence(
        self,
        source,
    ) -> PredictionResult:
        """
        Predict full sentence from a document image.

        Runs the complete OCR pipeline:
          preprocess → line detect → word detect → char segment → predict

        Args:
            source: Image source.

        Returns:
            PredictionResult with full sentence text.
        """
        self._ensure_loaded()
        t_start = time.perf_counter()

        from models.preprocessing.image_processor import (
            load_image, to_grayscale, remove_noise,
            enhance_contrast_clahe, binarize, morphological_close,
        )
        from models.segmentation.line_detector import LineDetector
        from models.segmentation.word_detector import WordDetector

        # Preprocessing
        img = load_image(source)
        gray = to_grayscale(img)
        gray = remove_noise(gray, 3)
        gray = enhance_contrast_clahe(gray)
        binary = binarize(gray, "otsu")
        binary = morphological_close(binary, 2)

        # Line detection
        line_detector = LineDetector()
        word_detector = WordDetector()
        lines = line_detector.detect(binary)
        word_detector.detect_all_lines(binary, lines)

        # Predict each word
        all_lines_text: List[str] = []
        all_confs: List[float] = []

        for line in lines:
            line_words: List[str] = []
            for word_region in line.words:
                word_strip = word_region.crop(binary)
                try:
                    word_result = self.predict_word(word_strip)
                    line_words.append(word_result.text)
                    all_confs.append(word_result.confidence)
                except Exception as e:
                    logger.warning("Word prediction failed: %s", e)
                    line_words.append("")
            all_lines_text.append(" ".join(line_words))

        sentence = "\n".join(all_lines_text).strip()
        confidence = float(np.mean(all_confs)) if all_confs else 0.0
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return PredictionResult(
            text=sentence,
            confidence=confidence,
            processing_ms=elapsed_ms,
            model_used="pipeline",
            engine_used="custom",
        )

    def get_model_info(self) -> List[Dict]:
        """Return info for all loaded models (for /model-info endpoint)."""
        self._ensure_loaded()
        return self.selector.get_model_info()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR inference on images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--dir", help="Directory of images")
    group.add_argument("--base64", help="Base64-encoded image string")

    parser.add_argument(
        "--task",
        default="character",
        choices=["character", "word", "sentence"],
        help="OCR task to perform",
    )
    parser.add_argument("--model", default="auto", help="Model to use")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", help="Save results to JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    cfg = config
    cfg.training.device = args.device

    predictor = OCRPredictor(cfg)
    predictor.load()

    def predict_one(source) -> Dict:
        if args.task == "character":
            result = predictor.predict_character(source, args.model, args.top_k)
        elif args.task == "word":
            result = predictor.predict_word(source, args.model)
        else:
            result = predictor.predict_sentence(source)
        return result.to_dict()

    if args.base64:
        img_bytes = base64.b64decode(args.base64)
        result = predict_one(img_bytes)
        print(json.dumps(result, indent=2))

    elif args.image:
        result = predict_one(Path(args.image))
        print(json.dumps(result, indent=2))

    elif args.dir:
        img_dir = Path(args.dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]
        results = []
        for img_path in sorted(images):
            r = predict_one(img_path)
            r["file"] = str(img_path)
            results.append(r)
            print(f"{img_path.name}: {r['text']!r} ({r['confidence']:.1%})")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output_json}")
