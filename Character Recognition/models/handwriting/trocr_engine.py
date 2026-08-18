"""
models/handwriting/trocr_engine.py

TrOCR inference engine for handwritten text line recognition.

Microsoft TrOCR is a vision encoder-decoder model pretrained on large-scale
handwritten and printed text data. The handwritten variant achieves state-of-the-art
CER on IAM (~2.89% at line level).

Architecture: ViT image encoder + RoBERTa text decoder (encoder-decoder Transformer)
Pretrained on: IAM, GNHK, CVL, RIMES, READ-2016 (combined ~large-scale)

Usage (inference):
    engine = TrOCREngine()
    text, confidence = engine.recognize(image_array)  # BGR numpy array

Usage (integrating with pipeline):
    engine = TrOCREngine(checkpoint_dir="models/saved_models/best_trocr")
    text, confidence = engine.recognize(cropped_line_image)

References:
    Li et al. (2021) — "TrOCR: Transformer-based Optical Character Recognition
    with Pre-trained Models" — https://arxiv.org/abs/2109.10282
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Base pretrained model name — used when no fine-tuned checkpoint is available
TROCR_BASE_MODEL = "microsoft/trocr-base-handwritten"

# Lazy-cached model + processor
_engine_cache: dict = {}


def _load_trocr(checkpoint_dir: Optional[str] = None):
    """
    Lazy-initialize TrOCR processor and model.

    Args:
        checkpoint_dir: If provided, loads fine-tuned weights from this directory.
                        Falls back to pretrained base model if not found.

    Returns:
        (processor, model) tuple cached in _engine_cache.
    """
    cache_key = str(checkpoint_dir) if checkpoint_dir else "base"
    if cache_key in _engine_cache:
        return _engine_cache[cache_key]

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizerFast, ViTImageProcessor
    except ImportError:
        raise ImportError(
            "transformers package required. Install: pip install transformers sentencepiece"
        )

    # Try to load fine-tuned checkpoint first
    model_source = TROCR_BASE_MODEL
    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir)
        if (ckpt_path / "config.json").exists():
            model_source = str(ckpt_path)
            logger.info("Loading fine-tuned TrOCR from: %s", model_source)
        else:
            logger.warning(
                "No fine-tuned checkpoint found at %s — using pretrained base model.",
                checkpoint_dir,
            )

    logger.info("Initializing TrOCR: %s (this may take a moment on first run)...", model_source)

    # Build processor manually — TrOCRProcessor.from_pretrained is broken in
    # newer transformers versions due to fast tokenizer vocab file resolution.
    img_proc  = ViTImageProcessor.from_pretrained(TROCR_BASE_MODEL)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(model_source)

    device = _get_device()
    model = model.to(device)
    model.eval()

    _engine_cache[cache_key] = (processor, model, device)
    logger.info("TrOCR ready on device: %s", device)
    return _engine_cache[cache_key]


def _get_device():
    """Auto-select best available device: CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _preprocess_for_trocr(image: np.ndarray):
    """
    Convert a raw numpy image (BGR or grayscale) to a PIL RGB image
    suitable for TrOCR's ViT encoder.

    TrOCR expects: PIL Image, RGB, any size (processor will resize to 384×384).
    """
    from PIL import Image

    if image is None or image.size == 0:
        raise ValueError("Empty image passed to TrOCR preprocessor.")

    # Convert BGR → RGB
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 2:
        # Grayscale → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image

    return Image.fromarray(rgb)


class TrOCREngine:
    """
    Handwriting recognition engine backed by Microsoft TrOCR.

    Attributes:
        checkpoint_dir: Path to fine-tuned model directory. If None, uses
                        the pretrained microsoft/trocr-base-handwritten.
        confidence_from_scores: Whether to extract token probabilities as confidence.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        confidence_from_scores: bool = True,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.confidence_from_scores = confidence_from_scores
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            _load_trocr(self.checkpoint_dir)
            self._initialized = True

    def recognize(
        self,
        image: np.ndarray,
        max_new_tokens: int = 128,
    ) -> Tuple[str, float]:
        """
        Recognize text in a single handwritten line image.

        Args:
            image:          BGR or grayscale numpy array (single text line crop).
            max_new_tokens: Maximum decoder tokens to generate.

        Returns:
            (recognized_text, confidence_score)
            confidence_score is the mean softmax probability of generated tokens.
        """
        self._ensure_initialized()
        import torch

        processor, model, device = _load_trocr(self.checkpoint_dir)

        pil_img = _preprocess_for_trocr(image)

        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            if self.confidence_from_scores:
                outputs = model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                generated_ids = outputs.sequences
                # Compute mean max-probability per token (geometric mean of probs)
                if outputs.scores:
                    token_probs = [
                        torch.softmax(s, dim=-1).max(dim=-1).values
                        for s in outputs.scores
                    ]
                    confidence = float(torch.stack(token_probs).mean().item())
                else:
                    confidence = 0.0
            else:
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                )
                confidence = 0.0

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip(), confidence

    def recognize_batch(
        self,
        images: List[np.ndarray],
        max_new_tokens: int = 128,
    ) -> List[Tuple[str, float]]:
        """
        Recognize text in a batch of handwritten line images.

        Args:
            images: List of BGR/grayscale numpy arrays.

        Returns:
            List of (text, confidence) tuples.
        """
        self._ensure_initialized()
        import torch

        processor, model, device = _load_trocr(self.checkpoint_dir)

        pil_imgs = [_preprocess_for_trocr(img) for img in images]

        pixel_values = processor(images=pil_imgs, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
            generated_ids = outputs.sequences
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            if outputs.scores:
                token_probs = [
                    torch.softmax(s, dim=-1).max(dim=-1).values
                    for s in outputs.scores
                ]
                mean_conf = float(torch.stack(token_probs).mean().item())
            else:
                mean_conf = 0.0

        return [(t.strip(), mean_conf) for t in texts]

    def is_available(self) -> bool:
        """Return True if transformers is installed."""
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False
