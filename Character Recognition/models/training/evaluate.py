"""
Model evaluation script.

Evaluates a trained model on the test set and generates:
  - Accuracy, Top-5 Accuracy, CER, WER
  - Confusion matrix (saved as PNG)
  - Per-class accuracy report (saved as PNG + CSV)
  - Confidence calibration (ECE)
  - Sample prediction grid

Usage:
    python -m models.training.evaluate --model cnn_batchnorm
    python -m models.training.evaluate --model crnn --word-mode
    python -m models.training.evaluate --all  # evaluate all checkpointed models
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from models.training.config import config, Config, EMNIST_BALANCED_LABELS
from models.utils.metrics import (
    MetricsTracker,
    character_error_rate,
    word_error_rate,
    per_class_accuracy,
    compute_confusion_matrix,
    expected_calibration_error,
)
from models.utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_confidence_distribution,
    plot_sample_predictions,
    plot_model_comparison,
    save_figure,
)
from models.utils.model_selector import (
    build_model,
    get_checkpoint_path,
    resolve_device,
    load_checkpoint,
    MODEL_REGISTRY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model_name: str,
    cfg: Config,
    word_mode: bool = False,
    save_plots: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a single trained model on the test set.

    Args:
        model_name: Architecture name.
        cfg:        Project configuration.
        word_mode:  Evaluate CRNN on word-level IAM test set.
        save_plots: Save confusion matrix and curve plots.

    Returns:
        Dict of metric name → value.
    """
    device = resolve_device(cfg.training.device)

    ckpt_path = get_checkpoint_path(model_name, cfg)
    if not ckpt_path.exists():
        logger.warning("No checkpoint found for %s at %s", model_name, ckpt_path)
        return {}

    # --- Load model ---
    model = build_model(model_name, cfg)
    checkpoint = load_checkpoint(model, ckpt_path, device)
    logger.info("Evaluating model: %s", model_name)

    # --- Load test data ---
    if word_mode:
        from models.datasets.iam_loader import load_iam
        _, _, test_loader = load_iam(cfg)
    else:
        from models.datasets.combined_loader import load_combined
        _, _, test_loader, dataset_info = load_combined(cfg)
        cfg.training.num_classes = dataset_info.num_classes

    # --- Collect predictions ---
    tracker = MetricsTracker(num_classes=cfg.training.num_classes)
    all_images: List[torch.Tensor] = []
    all_preds_str: List[str] = []
    all_targets_str: List[str] = []
    all_confs: List[float] = []
    all_correct: List[bool] = []
    label_map = EMNIST_BALANCED_LABELS

    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(test_loader):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        tracker.update(logits, targets, float(loss.item()))

        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        if batch_idx < 4:  # collect a few batches for visualization
            all_images.extend(images.cpu())

        for pred, target, conf in zip(
            preds.cpu().tolist(),
            targets.cpu().tolist(),
            confs.cpu().tolist(),
        ):
            pred_str = label_map.get(pred, str(pred))
            target_str = label_map.get(target, str(target))
            all_preds_str.append(pred_str)
            all_targets_str.append(target_str)
            all_confs.append(conf)
            all_correct.append(pred == target)

    # --- Compute metrics ---
    base_metrics = tracker.compute()
    cer = character_error_rate(all_preds_str, all_targets_str)
    wer = word_error_rate(
        [" ".join(all_preds_str)],
        [" ".join(all_targets_str)],
    )

    metrics = {
        **base_metrics,
        "cer": cer,
        "wer": wer,
    }

    logger.info("=" * 50)
    logger.info("Model: %s", model_name)
    logger.info("  Accuracy:   %.2f%%", metrics["accuracy"])
    logger.info("  Top-5 Acc:  %.2f%%", metrics["top5_accuracy"])
    logger.info("  CER:        %.4f", cer)
    logger.info("  WER:        %.4f", wer)
    logger.info("  ECE:        %.4f", metrics["ece"])
    logger.info("  Samples:    %d", metrics["num_samples"])
    logger.info("=" * 50)

    # --- Save plots ---
    if save_plots:
        plots_dir = Path(cfg.training.checkpoint_dir) / "plots" / model_name
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm = tracker.get_confusion_matrix(normalize=True)
        fig_cm = plot_confusion_matrix(cm, label_map=label_map, title=f"Confusion Matrix — {model_name}")
        save_figure(fig_cm, plots_dir / "confusion_matrix.png")

        # Per-class accuracy
        per_cls = per_class_accuracy(
            tracker.all_preds, tracker.all_targets,
            cfg.training.num_classes, label_map
        )
        fig_cls = plot_per_class_accuracy(per_cls, title=f"Per-Class Accuracy — {model_name}")
        save_figure(fig_cls, plots_dir / "per_class_accuracy.png")

        # Confidence distribution
        fig_conf = plot_confidence_distribution(
            all_confs, all_correct,
            title=f"Confidence Distribution — {model_name}",
        )
        save_figure(fig_conf, plots_dir / "confidence_distribution.png")

        # Sample predictions
        if all_images:
            sample_imgs = torch.stack(all_images[:16]).numpy()
            fig_samples = plot_sample_predictions(
                sample_imgs,
                all_preds_str[:len(sample_imgs)],
                all_targets_str[:len(sample_imgs)],
                all_confs[:len(sample_imgs)],
            )
            save_figure(fig_samples, plots_dir / "sample_predictions.png")

        # Save metrics to JSON
        metrics_path = plots_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save per-class accuracy to CSV
        csv_path = plots_dir / "per_class_accuracy.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "accuracy"])
            for cls_name, acc in sorted(per_cls.items(), key=lambda x: x[1]):
                writer.writerow([cls_name, f"{acc:.2f}"])

        logger.info("Saved evaluation plots and reports to %s", plots_dir)

    return metrics


# ---------------------------------------------------------------------------
# Evaluate all models and compare
# ---------------------------------------------------------------------------

def evaluate_all(cfg: Config, word_mode: bool = False) -> Dict[str, Dict]:
    """Evaluate every model that has a checkpoint and generate comparison plot."""
    all_results: Dict[str, Dict] = {}

    for model_name in MODEL_REGISTRY:
        ckpt = get_checkpoint_path(model_name, cfg)
        if not ckpt.exists():
            logger.info("Skipping %s — no checkpoint found.", model_name)
            continue
        metrics = evaluate_model(model_name, cfg, word_mode=word_mode)
        if metrics:
            all_results[model_name] = metrics

    if len(all_results) > 1:
        plots_dir = Path(cfg.training.checkpoint_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for metric_key in ("accuracy", "cer", "ece"):
            fig = plot_model_comparison(all_results, metric=metric_key)
            save_figure(fig, plots_dir / f"comparison_{metric_key}.png")

        # Summary JSON
        summary_path = plots_dir / "all_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Comparison saved to %s", plots_dir)

        # Print ranking table
        logger.info("\n%s", "=" * 60)
        logger.info("%-20s %10s %8s %8s", "Model", "Accuracy", "CER", "ECE")
        logger.info("-" * 60)
        for name, m in sorted(all_results.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True):
            logger.info(
                "%-20s %9.2f%% %8.4f %8.4f",
                name,
                m.get("accuracy", 0),
                m.get("cer", 0),
                m.get("ece", 0),
            )
        logger.info("=" * 60)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained OCR models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="cnn_batchnorm",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to evaluate",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpointed models")
    parser.add_argument("--word-mode", action="store_true", help="Use word-level IAM test set")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = config
    cfg.training.device = args.device

    if args.all:
        evaluate_all(cfg, word_mode=args.word_mode)
    else:
        evaluate_model(args.model, cfg, word_mode=args.word_mode, save_plots=not args.no_plots)
