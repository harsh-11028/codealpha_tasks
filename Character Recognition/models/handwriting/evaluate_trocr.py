"""
models/handwriting/evaluate_trocr.py

Evaluation script for TrOCR model vs EasyOCR baseline.

Computes and reports:
  - CER (Character Error Rate)
  - WER (Word Error Rate)
  - Loss on IAM test split
  - Prediction samples (ground truth vs output)

Usage:
    cd "/Users/harshmac/Desktop/Character Recognition"
    source .venv/bin/activate

    # Evaluate best fine-tuned TrOCR checkpoint:
    python -m models.handwriting.evaluate_trocr

    # Evaluate TrOCR + compare with EasyOCR:
    python -m models.handwriting.evaluate_trocr --compare-easyocr

    # Evaluate on single image file:
    python -m models.handwriting.evaluate_trocr --image path/to/image.png --ground-truth "actual text"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ─── PYTHONPATH bootstrap ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_trocr")

DEFAULT_CHECKPOINT = "models/saved_models/best_trocr"
MAX_SAMPLES        = 500   # Limit test set evaluation to save time on CPU


# ─── CER / WER ────────────────────────────────────────────────────────────────

def compute_cer_wer(predictions: List[str], references: List[str]) -> dict:
    try:
        from jiwer import cer, wer
    except ImportError:
        raise ImportError("jiwer required: pip install jiwer")

    preds_clean = [p if p.strip() else " " for p in predictions]
    refs_clean  = [r if r.strip() else " " for r in references]
    return {
        "cer": cer(refs_clean, preds_clean),
        "wer": wer(refs_clean, preds_clean),
    }


# ─── TrOCR evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_trocr_on_dataset(
    checkpoint_dir: str,
    max_samples: int = MAX_SAMPLES,
) -> dict:
    """Run TrOCR on IAM test split and compute CER/WER."""
    from models.handwriting.iam_hf_loader import load_iam_hf
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists() or not (ckpt_path / "config.json").exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_dir}.\n"
            "Run training first: python -m models.handwriting.train_trocr"
        )

    device_str = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)

    logger.info("Loading TrOCR from: %s on %s", checkpoint_dir, device)
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizerFast, ViTImageProcessor
    img_proc  = ViTImageProcessor.from_pretrained(str(ckpt_path))
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tokenizer)
    model     = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    _, _, test_ds = load_iam_hf()

    n = min(max_samples, len(test_ds))
    logger.info("Evaluating TrOCR on %d test samples...", n)

    preds, refs = [], []
    for i in range(n):
        sample = test_ds[i]
        img    = sample["image"]
        text   = sample.get("text", sample.get("label", ""))

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        preds.append(pred)
        refs.append(text)

        if i % 50 == 0 and i > 0:
            interim = compute_cer_wer(preds, refs)
            logger.info(
                "  Progress %d/%d | CER=%.2f%% | WER=%.2f%%",
                i, n, interim["cer"] * 100, interim["wer"] * 100,
            )

    metrics = compute_cer_wer(preds, refs)
    metrics["n_samples"] = n
    metrics["predictions"] = preds
    metrics["references"]  = refs
    return metrics


# ─── EasyOCR baseline ─────────────────────────────────────────────────────────

def evaluate_easyocr_baseline(max_samples: int = MAX_SAMPLES) -> dict:
    """Run EasyOCR on IAM test split and compute CER/WER."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    import easyocr
    import numpy as np
    from models.handwriting.iam_hf_loader import load_iam_hf
    from PIL import Image

    reader = easyocr.Reader(
        ["en"], gpu=False, verbose=False,
        model_storage_directory="models/saved_models/easyocr",
    )

    _, _, test_ds = load_iam_hf()
    n = min(max_samples, len(test_ds))
    logger.info("Evaluating EasyOCR baseline on %d test samples...", n)

    preds, refs = [], []
    for i in range(n):
        sample = test_ds[i]
        img    = sample["image"]
        text   = sample.get("text", sample.get("label", ""))

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img_arr = np.array(img.convert("RGB"))

        try:
            result = reader.readtext(img_arr, detail=0)
            pred   = " ".join(result).strip()
        except Exception:
            pred = ""

        preds.append(pred)
        refs.append(text)

        if i % 50 == 0 and i > 0:
            interim = compute_cer_wer(preds, refs)
            logger.info(
                "  EasyOCR progress %d/%d | CER=%.2f%% | WER=%.2f%%",
                i, n, interim["cer"] * 100, interim["wer"] * 100,
            )

    metrics = compute_cer_wer(preds, refs)
    metrics["n_samples"] = n
    return metrics


# ─── Single image evaluation ──────────────────────────────────────────────────

def evaluate_single_image(
    image_path: str,
    ground_truth: Optional[str],
    checkpoint_dir: str,
) -> None:
    """Recognize text from a single image file."""
    from models.handwriting.trocr_engine import TrOCREngine

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    engine = TrOCREngine(checkpoint_dir=checkpoint_dir)
    text, conf = engine.recognize(img)

    print("\n" + "─" * 60)
    print(f"Image       : {image_path}")
    print(f"Prediction  : {text}")
    print(f"Confidence  : {conf:.2%}")
    if ground_truth:
        metrics = compute_cer_wer([text], [ground_truth])
        print(f"Ground truth: {ground_truth}")
        print(f"CER         : {metrics['cer']:.2%}")
        print(f"WER         : {metrics['wer']:.2%}")
    print("─" * 60)


# ─── Report printing ──────────────────────────────────────────────────────────

def print_report(
    trocr_metrics: dict,
    easyocr_metrics: Optional[dict] = None,
    n_sample_display: int = 10,
) -> None:
    print("\n" + "=" * 70)
    print("HANDWRITING OCR EVALUATION REPORT")
    print("=" * 70)

    if easyocr_metrics:
        print(f"\n{'Metric':<25} {'EasyOCR (baseline)':>20} {'TrOCR (fine-tuned)':>20}")
        print("-" * 65)
        print(f"{'CER':<25} {easyocr_metrics['cer']*100:>19.2f}% {trocr_metrics['cer']*100:>19.2f}%")
        print(f"{'WER':<25} {easyocr_metrics['wer']*100:>19.2f}% {trocr_metrics['wer']*100:>19.2f}%")
        print(f"{'Samples evaluated':<25} {easyocr_metrics['n_samples']:>20} {trocr_metrics['n_samples']:>20}")
    else:
        print(f"\nTrOCR CER : {trocr_metrics['cer']*100:.2f}%")
        print(f"TrOCR WER : {trocr_metrics['wer']*100:.2f}%")
        print(f"Samples   : {trocr_metrics['n_samples']}")

    # Sample predictions
    preds = trocr_metrics.get("predictions", [])
    refs  = trocr_metrics.get("references",  [])
    if preds and refs:
        print(f"\n{'─'*70}")
        print("SAMPLE PREDICTIONS (TrOCR)")
        print(f"{'─'*70}")
        for i in range(min(n_sample_display, len(preds))):
            print(f"[{i+1:02d}] GT  : {refs[i]}")
            print(f"      PRED: {preds[i]}")
            print()

    print("=" * 70)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TrOCR on IAM test set")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                   help="Path to fine-tuned TrOCR checkpoint directory")
    p.add_argument("--compare-easyocr", action="store_true",
                   help="Also evaluate EasyOCR baseline for comparison")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                   help="Maximum number of test samples to evaluate")
    p.add_argument("--image", type=str, default=None,
                   help="Evaluate on a single image file path")
    p.add_argument("--ground-truth", type=str, default=None,
                   help="Ground truth text for single image evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.image:
        evaluate_single_image(args.image, args.ground_truth, args.checkpoint)
    else:
        trocr_metrics = evaluate_trocr_on_dataset(args.checkpoint, args.max_samples)
        easyocr_metrics = None
        if args.compare_easyocr:
            easyocr_metrics = evaluate_easyocr_baseline(args.max_samples)

        print_report(trocr_metrics, easyocr_metrics)
