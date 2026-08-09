"""
training/train.py — Main training script.

Supports training all 5 models with:
- Early stopping
- Learning rate scheduling
- Model checkpointing
- TensorBoard logging
- Automatic mixed precision (AMP)
"""

import copy
import os
import time
from pathlib import Path

import click
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ml.models.cnn_model import CNNModel
from ml.models.cnn_lstm_model import CNNLSTMModel
from ml.models.bilstm_model import BiLSTMModel
from ml.models.cnn_attention import CNNAttentionModel
from ml.models.wav2vec_model import Wav2Vec2EmotionModel
from training.config import DEFAULT_CONFIG, Config
from training.dataset import SERDataset
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


def get_model(architecture: str, config: Config) -> nn.Module:
    """Instantiate the requested model architecture."""
    num_classes = config.num_classes
    # Estimate input size from config (e.g., number of features extracted)
    # This is a simplification; in reality, we'd inspect a feature tensor shape
    input_size = 111  # Example size based on MFCC + Chroma + ZCR + etc.
    
    if architecture == "cnn":
        return CNNModel(num_classes=num_classes, input_size=input_size)
    elif architecture == "cnn_lstm":
        return CNNLSTMModel(num_classes=num_classes, input_size=input_size)
    elif architecture == "bilstm":
        return BiLSTMModel(num_classes=num_classes, input_size=input_size)
    elif architecture == "cnn_attention":
        return CNNAttentionModel(num_classes=num_classes, input_size=input_size)
    elif architecture == "wav2vec2":
        return Wav2Vec2EmotionModel(
            num_classes=num_classes, 
            model_name=config.model.wav2vec2_model_name,
            freeze_layers=config.model.wav2vec2_freeze_layers
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_model(architecture: str, config: Config):
    """Full training loop for a single model architecture."""
    logger.info(f"--- Starting training for {architecture} ---")
    
    device = torch.device(config.training.device)
    logger.info(f"Using device: {device}")
    
    # ── 1. Setup Data ────────────────────────────────────────────────────────
    metadata_path = Path(config.dataset.processed_data_dir) / "metadata.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata not found at {metadata_path}. Run dataset pipeline first.")
        return
        
    df = pd.read_csv(metadata_path)
    is_wav2vec2 = (architecture == "wav2vec2")
    
    train_dataset = SERDataset(df, "train", config, is_wav2vec2=is_wav2vec2)
    val_dataset = SERDataset(df, "val", config, is_wav2vec2=is_wav2vec2)
    
    # Simple hack to get actual input_size dynamically from the first sample
    if not is_wav2vec2 and len(train_dataset) > 0:
        sample_feat, _ = train_dataset[0]
        actual_input_size = sample_feat.shape[0]
        logger.info(f"Dynamic input size detected: {actual_input_size}")
    
    # On CPU / macOS, multiprocessing DataLoader workers cause HDF5 file-corruption
    # and "too many open files" errors.  Safe default: num_workers=0 (single-process).
    cpu_only = str(device) == "cpu"
    n_workers = 0 if cpu_only else config.training.num_workers
    pin = False if cpu_only else config.training.pin_memory
    logger.info(f"DataLoader: num_workers={n_workers}, pin_memory={pin}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin,
    )
    
    # ── 2. Setup Model & Optimizers ──────────────────────────────────────────
    model = get_model(architecture, config)
    # Re-init model if we found the actual input size (for non-wav2vec2)
    if not is_wav2vec2 and len(train_dataset) > 0:
        if architecture == "cnn":
            model = CNNModel(num_classes=config.num_classes, input_size=actual_input_size)
        elif architecture == "cnn_lstm":
            model = CNNLSTMModel(num_classes=config.num_classes, input_size=actual_input_size)
        elif architecture == "bilstm":
            model = BiLSTMModel(num_classes=config.num_classes, input_size=actual_input_size)
        elif architecture == "cnn_attention":
            model = CNNAttentionModel(num_classes=config.num_classes, input_size=actual_input_size)
            
    model = model.to(device)
    
    # Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    
    # Optimizer
    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.scheduler_t_max
    )
    
    scaler = torch.amp.GradScaler(device.type) if config.training.use_amp else None
    writer = SummaryWriter(log_dir=str(Path(config.training.tensorboard_dir) / f"{architecture}_{int(time.time())}"))
    
    # ── 3. Training Loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(1, config.training.num_epochs + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.num_epochs} [Train]"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward
            with torch.amp.autocast(device.type, enabled=config.training.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        # ── Val ──
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{config.training.num_epochs} [Val]"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.amp.autocast(device.type, enabled=config.training.use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        
        scheduler.step()
        
        # ── Logging ──
        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}"
        )
        
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/val", val_epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        writer.add_scalar("Accuracy/val", val_epoch_acc, epoch)
        
        # ── Early Stopping & Checkpointing ──
        if val_epoch_acc > best_val_acc + config.training.early_stopping_min_delta:
            best_val_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            # Save checkpoint
            checkpoint_dir = Path(config.training.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_path = checkpoint_dir / f"{architecture}_best.pt"
            
            torch.save({
                "model_name": architecture,
                "architecture": architecture,
                "version": "1.0.0",
                "model_state_dict": model.state_dict(),
                "num_classes": config.num_classes,
                "input_size": actual_input_size if not is_wav2vec2 else 0,
                "metrics": {"val_accuracy": float(val_epoch_acc)}
            }, save_path)
            
            logger.info(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            if config.training.early_stopping and epochs_no_improve >= config.training.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break
                
    writer.close()
    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc


@click.command()
@click.option("--model", "-m", default="cnn", help="Model architecture (cnn, cnn_lstm, bilstm, cnn_attention, wav2vec2, all)")
@click.option("--epochs", "-e", default=None, type=int, help="Override number of epochs")
@click.option("--batch-size", "-b", default=None, type=int, help="Override batch size")
@click.option("--auto-select", is_flag=True, help="Automatically copy best model to best_model.pt")
def main(model, epochs, batch_size, auto_select):
    """Train SER models."""
    setup_logger(log_level="INFO", log_file="logs/training.log")
    
    config = DEFAULT_CONFIG
    if epochs:
        config.training.num_epochs = epochs
    if batch_size:
        config.training.batch_size = batch_size
        
    models_to_train = [model] if model != "all" else ["cnn", "cnn_lstm", "bilstm", "cnn_attention", "wav2vec2"]
    
    results = {}
    for arch in models_to_train:
        acc = train_model(arch, config)
        results[arch] = acc

    logger.info("--- Final Results ---")
    for arch, acc in results.items():
        if acc is not None:
            logger.info(f"{arch}: {acc:.4f} Val Acc")
        else:
            logger.info(f"{arch}: training failed (check logs above)")

    # Always copy the best available checkpoint to best_model.pt
    # so the inference engine loads it automatically on next server start.
    import shutil

    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_arch = max(valid_results, key=lambda k: valid_results[k])
    else:
        # Fall back: find any *_best.pt file
        pts = list(Path(config.training.checkpoint_dir).glob("*_best.pt"))
        best_arch = pts[0].stem.replace("_best", "") if pts else None

    if best_arch:
        src = Path(config.training.checkpoint_dir) / f"{best_arch}_best.pt"
        dst = Path(config.training.checkpoint_dir) / "best_model.pt"
        if src.exists():
            shutil.copy(src, dst)
            logger.info(f"✅ Saved best model ({best_arch}) → {dst}")
        else:
            logger.warning(f"Could not find {src} to copy.")
    else:
        logger.warning("No trained model found to copy to best_model.pt.")


if __name__ == "__main__":
    main()
