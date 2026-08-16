"""
Main training script for the OCR models.

Supports all 5 architectures with:
  - Mixed precision training (AMP)
  - Early stopping
  - Learning rate scheduling (cosine / step / plateau)
  - Model checkpointing (best + last)
  - TensorBoard logging
  - K-fold cross validation
  - CLI interface

Usage:
    python -m models.training.train --model cnn_batchnorm --epochs 50
    python -m models.training.train --model crnn --word-mode --epochs 30
    python -m models.training.train --model vit --device cuda --amp
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.tensorboard import SummaryWriter

from models.training.config import config, Config
from models.utils.metrics import AverageMeter, accuracy, MetricsTracker
from models.utils.model_selector import build_model, get_checkpoint_path, resolve_device
from models.utils.visualization import plot_training_curves, save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    model_name: str,
    cfg: Config,
    is_best: bool = False,
) -> None:
    """Save model checkpoint (always saves 'last', optionally saves 'best')."""
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    last_path = ckpt_dir / f"last_{model_name}.pt"
    torch.save(payload, last_path)

    if is_best:
        best_path = ckpt_dir / f"best_{model_name}.pt"
        torch.save(payload, best_path)
        logger.info("✅ Saved best checkpoint → %s", best_path)


def load_last_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    model_name: str,
    cfg: Config,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Resume from last checkpoint if it exists.

    Returns:
        (start_epoch, best_metric_value)
    """
    last_path = Path(cfg.training.checkpoint_dir) / f"last_{model_name}.pt"
    if not last_path.exists():
        return 0, 0.0

    ckpt = torch.load(last_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    best_metric = ckpt["metrics"].get("val_accuracy", 0.0)
    logger.info("Resumed from epoch %d (best val_acc=%.2f%%)", start_epoch, best_metric)
    return start_epoch, best_metric


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: Config):
    tc = cfg.training
    params = model.parameters()
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
    elif tc.optimizer == "adam":
        return torch.optim.Adam(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(params, lr=tc.learning_rate,
                               momentum=tc.momentum, weight_decay=tc.weight_decay,
                               nesterov=True)
    raise ValueError(f"Unknown optimizer: {tc.optimizer!r}")


def build_scheduler(optimizer, cfg: Config, num_epochs: int):
    tc = cfg.training
    if tc.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=tc.lr_min)
    elif tc.scheduler == "step":
        return StepLR(optimizer, step_size=tc.lr_step_size, gamma=tc.lr_gamma)
    elif tc.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=tc.lr_gamma,
                                 patience=3, verbose=True)
    return None  # no scheduler


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler: Optional[GradScaler],
    cfg: Config,
    epoch: int,
    writer: Optional[SummaryWriter],
) -> Dict[str, float]:
    """Run one full training epoch."""
    model.train()
    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_acc")

    tc = cfg.training

    for batch_idx, batch in enumerate(loader):
        # Unpack batch (support both (img, label) and (img, label, length))
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        top1, = accuracy(logits.detach(), targets, topk=(1,))
        loss_meter.update(float(loss.item()), n=images.size(0))
        acc_meter.update(top1, n=images.size(0))

        if batch_idx % tc.log_interval == 0:
            logger.info(
                "Epoch %d [%d/%d] loss=%.4f acc=%.2f%%",
                epoch, batch_idx, len(loader), loss_meter.avg, acc_meter.avg,
            )
            if writer:
                global_step = epoch * len(loader) + batch_idx
                writer.add_scalar("Train/Loss", loss_meter.val, global_step)
                writer.add_scalar("Train/Accuracy", acc_meter.val, global_step)

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    cfg: Config,
) -> Dict[str, float]:
    """Run one full validation epoch."""
    model.eval()
    tracker = MetricsTracker(num_classes=cfg.training.num_classes)

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        tracker.update(logits, targets, float(loss.item()))

    return tracker.compute()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    cfg: Config,
    word_mode: bool = False,
    resume: bool = False,
) -> Dict[str, List[float]]:
    """
    Full training loop for a single model.

    Args:
        model_name: Architecture name from MODEL_REGISTRY.
        cfg:        Project configuration.
        word_mode:  Use IAM word-level loaders (for CRNN).
        resume:     Resume from last checkpoint.

    Returns:
        History dict with train/val loss and accuracy lists.
    """
    device = resolve_device(cfg.training.device)
    logger.info("=" * 60)
    logger.info("Training model: %s | device: %s | epochs: %d",
                model_name, device, cfg.training.max_epochs)
    logger.info("=" * 60)

    # --- Data ---
    if word_mode:
        from models.datasets.iam_loader import load_iam
        train_loader, val_loader, _ = load_iam(cfg)
    else:
        from models.datasets.combined_loader import load_combined
        train_loader, val_loader, _, dataset_info = load_combined(cfg)
        cfg.training.num_classes = dataset_info.num_classes

    # --- Model ---
    model = build_model(model_name, cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %s", f"{n_params:,}")

    # --- Loss ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Optimizer + Scheduler ---
    optimizer = build_optimizer(model, cfg)
    num_epochs = cfg.training.max_epochs
    scheduler = build_scheduler(optimizer, cfg, num_epochs)

    # --- AMP scaler ---
    use_amp = cfg.training.use_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed precision training enabled.")

    # --- TensorBoard ---
    tb_dir = Path(cfg.training.tensorboard_dir) / model_name
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info("TensorBoard logs → %s", tb_dir)

    # --- Resume ---
    start_epoch = 0
    best_val_acc = 0.0
    if resume:
        start_epoch, best_val_acc = load_last_checkpoint(
            model, optimizer, scheduler, model_name, cfg, device
        )

    # --- Early stopping state ---
    patience = cfg.training.early_stopping_patience
    patience_counter = 0

    # --- History ---
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    # =====================================================================
    # Training loop
    # =====================================================================
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.perf_counter()

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, cfg, epoch, writer,
        )
        val_metrics = val_epoch(model, val_loader, criterion, device, cfg)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["accuracy"])
            else:
                scheduler.step()

        # Log to TensorBoard
        writer.add_scalar("Epoch/Train_Loss", train_metrics["loss"], epoch)
        writer.add_scalar("Epoch/Val_Loss", val_metrics["loss"], epoch)
        writer.add_scalar("Epoch/Train_Acc", train_metrics["accuracy"], epoch)
        writer.add_scalar("Epoch/Val_Acc", val_metrics["accuracy"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # History
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
            model_name, cfg, is_best=is_best,
        )

        elapsed = time.perf_counter() - epoch_start
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f acc=%.2f%% | "
            "val_loss=%.4f acc=%.2f%% | best=%.2f%% | "
            "lr=%.2e | time=%.1fs%s",
            epoch + 1, num_epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            best_val_acc,
            optimizer.param_groups[0]["lr"],
            elapsed,
            " ⭐" if is_best else "",
        )

        # Early stopping
        if patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs).",
                epoch + 1, patience,
            )
            break

    # --- Save training curves ---
    writer.close()
    fig = plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"], history["val_acc"],
        model_name=model_name,
    )
    curves_path = Path(cfg.training.checkpoint_dir) / f"{model_name}_curves.png"
    save_figure(fig, curves_path)
    logger.info("Training complete. Best val accuracy: %.2f%%", best_val_acc)
    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OCR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="cnn_batchnorm",
        choices=["cnn_basic", "cnn_batchnorm", "residual_cnn", "crnn", "vit"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", default="auto", help="Compute device")
    parser.add_argument("--word-mode", action="store_true", help="Use IAM word-level dataset")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--no-emnist", action="store_true", help="Disable EMNIST")
    parser.add_argument("--no-mnist", action="store_true", help="Disable MNIST")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply CLI overrides to config
    cfg = config
    if args.epochs:
        cfg.training.max_epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.learning_rate = args.lr
    cfg.training.device = args.device
    cfg.training.use_amp = args.amp
    if args.no_emnist:
        cfg.dataset.use_emnist = False
    if args.no_mnist:
        cfg.dataset.use_mnist = False

    train(args.model, cfg, word_mode=args.word_mode, resume=args.resume)
