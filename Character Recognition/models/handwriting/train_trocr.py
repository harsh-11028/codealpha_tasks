"""
models/handwriting/train_trocr.py

Fine-tune Microsoft TrOCR (trocr-base-handwritten) on IAM Lines dataset.

Usage:
    cd "/Users/harshmac/Desktop/Character Recognition"
    source .venv/bin/activate
    python -m models.handwriting.train_trocr

Advanced options:
    python -m models.handwriting.train_trocr \\
        --epochs 15 \\
        --batch-size 8 \\
        --lr 5e-5 \\
        --output-dir models/saved_models/best_trocr \\
        --device auto

This script:
    1. Downloads IAM Lines from Hugging Face (Teklia/IAM-line) — no credentials needed
    2. Fine-tunes microsoft/trocr-base-handwritten
    3. Tracks CER + WER per epoch via jiwer
    4. Saves best checkpoint by validation CER
    5. Writes TensorBoard logs
    6. Prints a final evaluation table
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ─── PYTHONPATH bootstrap (allow running directly) ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_trocr")

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL      = "microsoft/trocr-base-handwritten"
DEFAULT_OUTPUT  = "models/saved_models/best_trocr"
DEFAULT_EPOCHS  = 15
DEFAULT_BATCH   = 8
DEFAULT_LR      = 5e-5
SEED            = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def compute_cer_wer(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute Character Error Rate and Word Error Rate using jiwer."""
    try:
        from jiwer import cer, wer
    except ImportError:
        raise ImportError("jiwer required: pip install jiwer")

    # jiwer requires non-empty strings
    preds_clean = [p if p.strip() else " " for p in predictions]
    refs_clean  = [r if r.strip() else " " for r in references]

    cer_val = cer(refs_clean, preds_clean)
    wer_val = wer(refs_clean, preds_clean)
    return {"cer": cer_val, "wer": wer_val}


# ─── Dataset wrapper ──────────────────────────────────────────────────────────

class IAMTrOCRDataset(torch.utils.data.Dataset):
    """
    Wrap a Hugging Face IAM dataset split for TrOCR fine-tuning.

    Each sample → (pixel_values tensor, labels tensor).
    """

    def __init__(self, hf_dataset, processor, max_target_length: int = 128):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        # Image
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        pixel_values = self.processor(
            images=img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Label
        text = sample.get("text", sample.get("label", ""))
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so loss ignores it
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels, "text": text}


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels       = torch.stack([b["labels"]       for b in batch])
    texts        = [b["text"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels, "texts": texts}


# ─── Training epoch ───────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches  += 1

        if batch_idx % 20 == 0:
            logger.info(
                "Epoch %d | batch %d/%d | loss=%.4f",
                epoch, batch_idx, len(loader), loss.item(),
            )
            if writer:
                step = epoch * len(loader) + batch_idx
                writer.add_scalar("Train/Loss", loss.item(), step)

    return total_loss / max(n_batches, 1)


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    processor,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
    split: str = "Val",
    max_new_tokens: int = 128,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds: List[str] = []
    all_refs:  List[str] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)
        refs         = batch["texts"]

        # Compute loss
        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += float(outputs.loss.item())
        n_batches  += 1

        # Generate predictions
        generated_ids = model.generate(
            pixel_values, max_new_tokens=max_new_tokens,
        )
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_preds.extend([p.strip() for p in preds])
        all_refs.extend(refs)

    avg_loss = total_loss / max(n_batches, 1)
    metrics  = compute_cer_wer(all_preds, all_refs)
    metrics["loss"] = avg_loss

    logger.info(
        "%s Epoch %d | loss=%.4f | CER=%.2f%% | WER=%.2f%%",
        split, epoch, avg_loss,
        metrics["cer"] * 100, metrics["wer"] * 100,
    )

    if writer:
        writer.add_scalar(f"{split}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{split}/CER",  metrics["cer"],  epoch)
        writer.add_scalar(f"{split}/WER",  metrics["wer"],  epoch)

    return metrics


# ─── Main training loop ───────────────────────────────────────────────────────

def train(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    output_dir: str = DEFAULT_OUTPUT,
    device_str: str = "auto",
    resume: bool = False,
) -> None:
    """
    Full fine-tuning pipeline for TrOCR on IAM Lines.
    """
    set_seed(SEED)
    device = resolve_device(device_str)
    logger.info("Device: %s", device)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    from models.handwriting.iam_hf_loader import load_iam_hf
    train_hf, val_hf, test_hf = load_iam_hf()

    # ── 2. Load processor + model ────────────────────────────────────────────
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizerFast, ViTImageProcessor

    source = output_dir if (resume and (Path(output_dir) / "config.json").exists()) else BASE_MODEL
    logger.info("Loading model from: %s", source)

    # Build processor manually — TrOCRProcessor.from_pretrained is broken in
    # newer transformers versions due to fast tokenizer vocab file resolution.
    img_proc = ViTImageProcessor.from_pretrained(BASE_MODEL)
    tok      = RobertaTokenizerFast.from_pretrained("roberta-base")
    processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)
    model     = VisionEncoderDecoderModel.from_pretrained(source)
    model     = model.to(device)

    # Required for seq2seq generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id

    # ── 3. Build DataLoaders ─────────────────────────────────────────────────
    train_ds = IAMTrOCRDataset(train_hf, processor)
    val_ds   = IAMTrOCRDataset(val_hf,   processor)
    test_ds  = IAMTrOCRDataset(test_hf,  processor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )

    logger.info(
        "Dataset sizes — train: %d | val: %d | test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # ── 4. Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # ── 5. TensorBoard ───────────────────────────────────────────────────────
    tb_dir = Path("tensorboard_logs") / "trocr"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    # ── 6. Training loop ─────────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_cer       = float("inf")
    patience_count = 0
    patience       = 5
    best_epoch     = 0

    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        val_metrics = evaluate(model, processor, val_loader, device, epoch, writer, "Val")
        scheduler.step()

        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "elapsed_s": elapsed,
        }
        history.append(row)

        # Save best checkpoint
        val_cer = val_metrics["cer"]
        if val_cer < best_cer:
            best_cer   = val_cer
            best_epoch = epoch
            patience_count = 0
            logger.info(
                "✅ New best CER=%.2f%% at epoch %d — saving to %s",
                best_cer * 100, epoch, output_dir,
            )
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            # Save vocab/config alongside
            import json
            meta = {
                "base_model": BASE_MODEL,
                "best_epoch": epoch,
                "best_val_cer": best_cer,
                "epochs_trained": epoch,
                "batch_size": batch_size,
                "learning_rate": lr,
                "seed": SEED,
            }
            (output_path / "training_meta.json").write_text(
                json.dumps(meta, indent=2)
            )
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    patience,
                )
                break

    writer.close()

    # ── 7. Test evaluation ───────────────────────────────────────────────────
    logger.info("Loading best checkpoint for final test evaluation...")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizerFast, ViTImageProcessor
    img_proc2    = ViTImageProcessor.from_pretrained(output_dir)
    tok2         = RobertaTokenizerFast.from_pretrained("roberta-base")
    best_processor = TrOCRProcessor(image_processor=img_proc2, tokenizer=tok2)
    best_model     = VisionEncoderDecoderModel.from_pretrained(output_dir).to(device)
    best_model.eval()

    test_metrics = evaluate(
        best_model, best_processor, test_loader, device,
        best_epoch, writer=None, split="Test",
    )

    # ── 8. Final report ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TrOCR FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Best epoch    : {best_epoch}")
    print(f"Best val CER  : {best_cer * 100:.2f}%")
    print(f"Test CER      : {test_metrics['cer'] * 100:.2f}%")
    print(f"Test WER      : {test_metrics['wer'] * 100:.2f}%")
    print(f"Test loss     : {test_metrics['loss']:.4f}")
    print(f"Checkpoint    : {output_dir}")
    print("=" * 60)

    # Save history
    import json
    (output_path / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )
    logger.info("Training history saved to %s/training_history.json", output_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune TrOCR on IAM Lines dataset")
    p.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS, help="Max training epochs")
    p.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH,  help="Batch size")
    p.add_argument("--lr",         type=float, default=DEFAULT_LR,     help="Peak learning rate")
    p.add_argument("--output-dir", type=str,   default=DEFAULT_OUTPUT, help="Checkpoint output directory")
    p.add_argument("--device",     type=str,   default="auto",         help="Device: auto|cpu|cuda|mps")
    p.add_argument("--resume",     action="store_true",                 help="Resume from existing checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        device_str=args.device,
        resume=args.resume,
    )
