"""
training/evaluate.py — Model Evaluation Script.

Evaluates a trained model on the test set.
Generates:
- Classification report (Precision, Recall, F1-score)
- Confusion matrix (saved as a plot in the metrics directory)
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.config import DEFAULT_CONFIG, EMOTION_LABELS
from training.dataset import SERDataset
from training.train import get_model
from utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


def evaluate_model(model_path: Path, output_dir: Path):
    """Run evaluation on the test set and generate metrics."""
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return

    device = torch.device(DEFAULT_CONFIG.training.device)
    logger.info(f"Loading checkpoint from {model_path} onto {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "cnn")
    input_size = checkpoint.get("input_size", 40)
    
    # ── 1. Setup Model ───────────────────────────────────────────────────────
    # Temporarily inject input size into config just to instantiate the model
    DEFAULT_CONFIG.num_classes = checkpoint.get("num_classes", 8)
    
    # Custom instantiate to pass input_size correctly
    is_wav2vec2 = (architecture == "wav2vec2")
    
    if architecture == "cnn":
        from ml.models.cnn_model import CNNModel
        model = CNNModel(num_classes=DEFAULT_CONFIG.num_classes, input_size=input_size)
    elif architecture == "cnn_lstm":
        from ml.models.cnn_lstm_model import CNNLSTMModel
        model = CNNLSTMModel(num_classes=DEFAULT_CONFIG.num_classes, input_size=input_size)
    elif architecture == "bilstm":
        from ml.models.bilstm_model import BiLSTMModel
        model = BiLSTMModel(num_classes=DEFAULT_CONFIG.num_classes, input_size=input_size)
    elif architecture == "cnn_attention":
        from ml.models.cnn_attention import CNNAttentionModel
        model = CNNAttentionModel(num_classes=DEFAULT_CONFIG.num_classes, input_size=input_size)
    elif architecture == "wav2vec2":
        from ml.models.wav2vec_model import Wav2Vec2EmotionModel
        model = Wav2Vec2EmotionModel(num_classes=DEFAULT_CONFIG.num_classes, freeze_layers=0)
    else:
        logger.error(f"Unknown architecture in checkpoint: {architecture}")
        return
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # ── 2. Setup Data ────────────────────────────────────────────────────────
    metadata_path = Path(DEFAULT_CONFIG.dataset.processed_data_dir) / "metadata.csv"
    df = pd.read_csv(metadata_path)
    
    test_dataset = SERDataset(df, "test", DEFAULT_CONFIG, is_wav2vec2=is_wav2vec2)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=DEFAULT_CONFIG.training.batch_size, 
        shuffle=False, 
        num_workers=DEFAULT_CONFIG.training.num_workers
    )
    
    # ── 3. Evaluate ──────────────────────────────────────────────────────────
    logger.info("Running inference on test set...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast(device.type, enabled=DEFAULT_CONFIG.training.use_amp):
                outputs = model(inputs)
                
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # ── 4. Metrics & Reports ─────────────────────────────────────────────────
    target_names = EMOTION_LABELS[:DEFAULT_CONFIG.num_classes]
    
    logger.info("\n--- Classification Report ---")
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print(report)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {architecture}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = output_dir / f"confusion_matrix_{architecture}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix plot to {cm_path}")


@click.command()
@click.option("--model-path", "-m", default=None, help="Path to model checkpoint. Defaults to best_model.pt")
@click.option("--output-dir", "-o", default="metrics", help="Output directory for plots")
def main(model_path, output_dir):
    """Evaluate a trained SER model."""
    setup_logger(log_level="INFO", log_file="logs/evaluate.log")
    
    if model_path is None:
        model_path = Path(DEFAULT_CONFIG.training.checkpoint_dir) / "best_model.pt"
    else:
        model_path = Path(model_path)
        
    evaluate_model(model_path, Path(output_dir))


if __name__ == "__main__":
    main()
