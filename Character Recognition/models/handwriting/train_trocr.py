"""
TrOCR Fine-tuning Script (Redesigned for Apple Silicon)

Features:
- Sanity Mode (--sanity): Subsets dataset to 50 samples for rapid testing
- Robust Progress Reporting: ETA, epoch/batch counts, percentages
- Continuous Checkpointing: Saves every epoch
- Resumable: --resume flag loads latest weights
- Evaluation: Side-by-side comparison with EasyOCR
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    RobertaTokenizerFast,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_trocr")

# ─── Constants ─────────────────────────────────────────────────────────────
BASE_MODEL     = "microsoft/trocr-base-handwritten"
DEFAULT_EPOCHS = 15
DEFAULT_BATCH  = 4
DEFAULT_LR     = 5e-5
DEFAULT_OUTPUT = "models/saved_models/best_trocr"
SEED           = 42

def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"

def compute_cer_wer(preds: List[str], refs: List[str]) -> Dict[str, float]:
    import evaluate as hf_evaluate
    cer_metric = hf_evaluate.load("cer")
    wer_metric = hf_evaluate.load("wer")
    
    # Filter out completely empty references
    valid_preds, valid_refs = [], []
    for p, r in zip(preds, refs):
        if r.strip():
            valid_preds.append(p.strip() if p.strip() else " ")
            valid_refs.append(r.strip())
            
    if not valid_refs:
        return {"cer": 1.0, "wer": 1.0}
        
    try:
        c = cer_metric.compute(predictions=valid_preds, references=valid_refs)
        w = wer_metric.compute(predictions=valid_preds, references=valid_refs)
        return {"cer": float(c), "wer": float(w)}
    except Exception:
        return {"cer": 1.0, "wer": 1.0}


# ─── Training loop ───────────────────────────────────────────────────────────
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
    n_batches = 0
    total_batches = len(loader)
    
    epoch_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        batch_start = time.perf_counter()
        
        pixel_values = batch["pixel_values"].to(device)
        labels       = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        loss_val = float(loss.item())
        if math.isnan(loss_val) or math.isinf(loss_val):
            logger.warning(f"Epoch {epoch} | batch {batch_idx} | loss is NaN/Inf! Skipping optimizer step.")
            optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss_val
            n_batches += 1

        batch_time = time.perf_counter() - batch_start
        
        # Logging
        if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
            percent = (batch_idx + 1) / total_batches * 100
            elapsed = time.perf_counter() - epoch_start
            avg_batch_time = elapsed / (batch_idx + 1)
            eta = (total_batches - (batch_idx + 1)) * avg_batch_time
            
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{total_batches}] ({percent:.1f}%) | "
                f"Loss: {loss_val:.4f} | "
                f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}"
            )
            
            if writer:
                step = (epoch - 1) * total_batches + batch_idx
                writer.add_scalar("Train/Loss", loss_val, step)

    return total_loss / max(n_batches, 1)


# ─── Validation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(
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

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += float(outputs.loss.item())
        n_batches  += 1

        generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_preds.extend([p.strip() for p in preds])
        all_refs.extend(refs)

    avg_loss = total_loss / max(n_batches, 1)
    metrics  = compute_cer_wer(all_preds, all_refs)
    metrics["loss"] = avg_loss

    logger.info(
        f"{split} Epoch {epoch} | Loss: {avg_loss:.4f} | "
        f"CER: {metrics['cer']*100:.2f}% | WER: {metrics['wer']*100:.2f}%"
    )

    if writer:
        writer.add_scalar(f"{split}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{split}/CER",  metrics["cer"],  epoch)
        writer.add_scalar(f"{split}/WER",  metrics["wer"],  epoch)

    return metrics, all_preds, all_refs


# ─── Evaluation Comparison ──────────────────────────────────────────────────
def run_comparison(preds: List[str], refs: List[str], images, easyocr_engine):
    """Compare TrOCR vs EasyOCR on the test set."""
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION COMPARISON")
    logger.info("=" * 60)
    
    easy_preds = []
    import numpy as np
    
    for idx, (img, ref, trocr_pred) in enumerate(zip(images, refs, preds)):
        if easyocr_engine:
            try:
                # EasyOCR expects BGR numpy array
                img_np = np.array(img.convert('RGB'))
                img_bgr = img_np[:, :, ::-1].copy()
                results = easyocr_engine.read_image(img_bgr)
                easy_pred = " ".join(r.text for r in results)
            except Exception:
                easy_pred = ""
        else:
            easy_pred = "[EasyOCR Engine Not Loaded]"
            
        easy_preds.append(easy_pred)
        
        # Print first 5 samples
        if idx < 5:
            logger.info(f"Sample {idx+1}:")
            logger.info(f"  Ground Truth : {ref}")
            logger.info(f"  TrOCR (New)  : {trocr_pred}")
            logger.info(f"  EasyOCR (Old): {easy_pred}")
            logger.info("-" * 40)
            
    trocr_metrics = compute_cer_wer(preds, refs)
    easy_metrics = compute_cer_wer(easy_preds, refs)
    
    logger.info("OVERALL METRICS:")
    logger.info(f"  TrOCR   CER: {trocr_metrics['cer']*100:.2f}% | WER: {trocr_metrics['wer']*100:.2f}%")
    logger.info(f"  EasyOCR CER: {easy_metrics['cer']*100:.2f}% | WER: {easy_metrics['wer']*100:.2f}%")
    logger.info("=" * 60)


# ─── Main Pipeline ───────────────────────────────────────────────────────────
def train(
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: str,
    device_str: str,
    resume: bool,
    sanity: bool,
):
    torch.manual_seed(SEED)
    
    # ── 1. Device ────────────────────────────────────────────────────────────
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    logger.info(f"Device: {device}")
    if sanity:
        logger.info("SANITY MODE ENABLED: Using tiny dataset subset.")

    # ── 2. Load Model & Processor ────────────────────────────────────────────
    output_path = Path(output_dir)
    
    if resume and (output_path / "config.json").exists():
        logger.info(f"Resuming from checkpoint: {output_dir}")
        model_path = output_dir
    else:
        logger.info(f"Loading pretrained model: {BASE_MODEL}")
        model_path = BASE_MODEL
        
    img_proc  = ViTImageProcessor.from_pretrained(model_path)
    tok       = RobertaTokenizerFast.from_pretrained("roberta-base")
    processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)
    
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.to(device)

    # ── 3. Dataset ───────────────────────────────────────────────────────────
    from models.handwriting.iam_hf_loader import load_iam_hf
    from torch.utils.data import Dataset
    
    class IAMTrOCRDataset(Dataset):
        def __init__(self, hf_dataset, processor):
            self.dataset = hf_dataset
            self.processor = processor

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx: int):
            item = self.dataset[idx]
            image = item["image"].convert("RGB")
            text = item["text"]
            
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
            labels = self.processor.tokenizer(text, padding="max_length", max_length=128).input_ids
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            
            return {"pixel_values": pixel_values, "labels": torch.tensor(labels), "text": text}

    train_hf, val_hf, test_hf = load_iam_hf()
    
    if sanity:
        # Subset dataset for quick sanity test
        train_hf = train_hf.select(range(min(50, len(train_hf))))
        val_hf   = val_hf.select(range(min(10, len(val_hf))))
        test_hf  = test_hf.select(range(min(10, len(test_hf))))
        
    train_ds = IAMTrOCRDataset(train_hf, processor)
    val_ds   = IAMTrOCRDataset(val_hf,   processor)
    test_ds  = IAMTrOCRDataset(test_hf,  processor)

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        labels       = torch.stack([b["labels"] for b in batch])
        texts        = [b["text"] for b in batch]
        return {"pixel_values": pixel_values, "labels": labels, "texts": texts}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f"Dataset sizes — train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    # ── 4. Optimizer ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # ── 5. TensorBoard ───────────────────────────────────────────────────────
    tb_dir = Path("tensorboard_logs") / "trocr"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 6. Training Loop ─────────────────────────────────────────────────────
    best_cer = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        val_metrics, _, _ = evaluate_model(model, processor, val_loader, device, epoch, writer, "Val")
        scheduler.step()

        elapsed = time.perf_counter() - t0
        
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "elapsed_s": elapsed,
        }
        history.append(row)

        # Save checkpoint every epoch for resumption
        logger.info(f"Saving checkpoint to {output_dir}")
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        
        meta = {
            "base_model": BASE_MODEL,
            "last_epoch": epoch,
            "epochs_trained": epoch,
            "batch_size": batch_size,
            "learning_rate": lr,
            "seed": SEED,
        }
        (output_path / "training_meta.json").write_text(json.dumps(meta, indent=2))
        
        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            logger.info(f"🌟 New best CER: {best_cer*100:.2f}%")

    writer.close()
    
    (output_path / "training_history.json").write_text(json.dumps(history, indent=2))

    # ── 7. Final Test Evaluation ─────────────────────────────────────────────
    logger.info("Running final test evaluation on held-out samples...")
    test_metrics, test_preds, test_refs = evaluate_model(model, processor, test_loader, device, epochs, None, "Test")
    
    # Extract original PIL images for EasyOCR
    test_images = [test_hf[i]["image"] for i in range(len(test_hf))]
    
    # Load EasyOCR for comparison
    try:
        from models.ocr.easyocr_engine import EasyOCREngine
        easyocr_engine = EasyOCREngine(languages=["en"], gpu=True)
    except Exception as e:
        logger.warning(f"Could not load EasyOCR for comparison: {e}")
        easyocr_engine = None
        
    run_comparison(test_preds, test_refs, test_images, easyocr_engine)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",         type=float, default=DEFAULT_LR)
    p.add_argument("--output-dir", type=str,   default=DEFAULT_OUTPUT)
    p.add_argument("--device",     type=str,   default="auto")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--sanity",     action="store_true", help="Run quick 50-sample sanity test")
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
        sanity=args.sanity,
    )
