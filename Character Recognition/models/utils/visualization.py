"""
Visualization utilities for training and evaluation.

Generates:
  - Training/validation loss and accuracy curves
  - Confusion matrix heatmap
  - Per-class accuracy bar chart
  - Confidence distribution histogram
  - Sample prediction grid
  - HOG visualization
  - Attention map overlay (for ViT)

All functions return matplotlib figures that can be:
  - Saved to disk as PNG/PDF
  - Converted to base64 for the REST API
  - Logged to TensorBoard
"""

from __future__ import annotations

import os
import base64
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure matplotlib cache writes inside the workspace rather than system home directory
_mpl_dir = Path(__file__).resolve().parent.parent / "saved_models" / ".matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

STYLE = {
    "bg_color": "#0f0f23",
    "grid_color": "#1e1e3a",
    "text_color": "#e0e0f0",
    "accent1": "#7c3aed",   # purple
    "accent2": "#2563eb",   # blue
    "accent3": "#06b6d4",   # cyan
    "accent4": "#10b981",   # green
    "error_color": "#ef4444",
    "font_size": 11,
}


def _apply_dark_style(fig: plt.Figure, axes) -> None:
    """Apply dark theme to figure and axes."""
    fig.patch.set_facecolor(STYLE["bg_color"])
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(STYLE["grid_color"])
        ax.tick_params(colors=STYLE["text_color"])
        ax.xaxis.label.set_color(STYLE["text_color"])
        ax.yaxis.label.set_color(STYLE["text_color"])
        ax.title.set_color(STYLE["text_color"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid_color"])
        ax.grid(True, color=STYLE["grid_color"], linewidth=0.5, alpha=0.7)


def figure_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def save_figure(fig: plt.Figure, path: Path, dpi: int = 120) -> None:
    """Save figure to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved figure to %s", path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    model_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training and validation loss + accuracy over epochs.

    Args:
        train_losses: Training loss per epoch.
        val_losses:   Validation loss per epoch.
        train_accs:   Training accuracy per epoch.
        val_accs:     Validation accuracy per epoch.
        model_name:   Title label.
        save_path:    If provided, saves the figure.

    Returns:
        Matplotlib Figure.
    """
    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_style(fig, [ax1, ax2])

    # Loss
    ax1.plot(epochs, train_losses, color=STYLE["accent2"], linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_losses,   color=STYLE["error_color"], linewidth=2, label="Val Loss", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curves — {model_name}")
    ax1.legend(facecolor=STYLE["bg_color"], edgecolor=STYLE["grid_color"],
               labelcolor=STYLE["text_color"])
    if val_losses:
        best_epoch = int(np.argmin(val_losses)) + 1
        ax1.axvline(best_epoch, color=STYLE["accent4"], linewidth=1, linestyle=":", alpha=0.8,
                    label=f"Best (epoch {best_epoch})")

    # Accuracy
    ax2.plot(epochs, train_accs, color=STYLE["accent1"], linewidth=2, label="Train Acc")
    ax2.plot(epochs, val_accs,   color=STYLE["accent3"], linewidth=2, label="Val Acc", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Accuracy Curves — {model_name}")
    ax2.set_ylim(0, 100)
    ax2.legend(facecolor=STYLE["bg_color"], edgecolor=STYLE["grid_color"],
               labelcolor=STYLE["text_color"])
    if val_accs:
        best_epoch = int(np.argmax(val_accs)) + 1
        ax2.axvline(best_epoch, color=STYLE["accent4"], linewidth=1, linestyle=":", alpha=0.8)

    fig.suptitle(f"Training History — {model_name}", color=STYLE["text_color"], fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    label_map: Optional[Dict[int, str]] = None,
    title: str = "Confusion Matrix",
    max_classes: int = 47,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a normalized confusion matrix heatmap.

    Args:
        cm:          Normalized confusion matrix (num_classes × num_classes).
        label_map:   Class index → character label mapping.
        title:       Figure title.
        max_classes: Cap displayed classes (for readability).
        save_path:   Optional save path.

    Returns:
        Matplotlib Figure.
    """
    n = min(cm.shape[0], max_classes)
    cm_display = cm[:n, :n]

    labels = [label_map[i] if (label_map and i in label_map) else str(i)
              for i in range(n)]

    fig_size = max(8, n // 3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    _apply_dark_style(fig, ax)

    im = ax.imshow(cm_display, interpolation="nearest", cmap="plasma", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=STYLE["text_color"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=STYLE["text_color"])

    if n <= 50:
        tick_marks = np.arange(n)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8,
                           color=STYLE["text_color"])
        ax.set_yticklabels(labels, fontsize=8, color=STYLE["text_color"])

    ax.set_xlabel("Predicted", color=STYLE["text_color"])
    ax.set_ylabel("True", color=STYLE["text_color"])
    ax.set_title(title, color=STYLE["text_color"])
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Per-class accuracy bar chart
# ---------------------------------------------------------------------------

def plot_per_class_accuracy(
    per_class_acc: Dict[str, float],
    title: str = "Per-Class Accuracy",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of per-class accuracy.

    Classes sorted from lowest to highest accuracy to highlight
    the weakest recognition classes.

    Args:
        per_class_acc: Dict[class_name → accuracy%].
        title:         Figure title.
        save_path:     Optional save path.

    Returns:
        Matplotlib Figure.
    """
    sorted_items = sorted(per_class_acc.items(), key=lambda kv: kv[1])
    classes = [item[0] for item in sorted_items]
    accs = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.2)))
    _apply_dark_style(fig, ax)

    colors = [STYLE["error_color"] if a < 80 else STYLE["accent4"] for a in accs]
    bars = ax.barh(classes, accs, color=colors, alpha=0.85, height=0.7)

    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title, color=STYLE["text_color"])
    ax.set_xlim(0, 100)
    ax.axvline(np.mean(accs), color=STYLE["accent3"], linewidth=1.5, linestyle="--",
               label=f"Mean: {np.mean(accs):.1f}%")
    ax.legend(facecolor=STYLE["bg_color"], labelcolor=STYLE["text_color"])

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Confidence distribution
# ---------------------------------------------------------------------------

def plot_confidence_distribution(
    confidences: List[float],
    correct_mask: List[bool],
    title: str = "Confidence Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Histogram of prediction confidences, split by correct vs incorrect.

    Args:
        confidences:  List of confidence scores [0, 1].
        correct_mask: Corresponding boolean list (True = correct prediction).
        title:        Figure title.
        save_path:    Optional save path.

    Returns:
        Matplotlib Figure.
    """
    confs = np.array(confidences)
    mask = np.array(correct_mask, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, ax)

    bins = np.linspace(0, 1, 25)
    ax.hist(confs[mask],  bins=bins, alpha=0.75, color=STYLE["accent4"],
            label="Correct", density=True)
    ax.hist(confs[~mask], bins=bins, alpha=0.75, color=STYLE["error_color"],
            label="Incorrect", density=True)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(title, color=STYLE["text_color"])
    ax.legend(facecolor=STYLE["bg_color"], labelcolor=STYLE["text_color"])

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Sample prediction grid
# ---------------------------------------------------------------------------

def plot_sample_predictions(
    images: np.ndarray,
    predictions: List[str],
    targets: List[str],
    confidences: List[float],
    num_samples: int = 16,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Grid of sample images with predicted and true labels.

    Incorrect predictions are highlighted with a red border.

    Args:
        images:       Array of grayscale images (N, H, W) or (N, 1, H, W).
        predictions:  Predicted character/word strings.
        targets:      Ground truth strings.
        confidences:  Confidence scores.
        num_samples:  Number of images to display.
        save_path:    Optional save path.

    Returns:
        Matplotlib Figure.
    """
    n = min(num_samples, len(images))
    cols = 8
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.0))
    fig.patch.set_facecolor(STYLE["bg_color"])
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        ax.axis("off")
        if i >= n:
            continue

        img = images[i]
        if img.ndim == 3:
            img = img.squeeze()  # (1, H, W) → (H, W)

        # Denormalize if needed
        if img.min() < 0:
            img = (img + 1) / 2

        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        is_correct = predictions[i] == targets[i]
        border_color = STYLE["accent4"] if is_correct else STYLE["error_color"]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        ax.set_title(
            f"P:{predictions[i]} T:{targets[i]}\n{confidences[i]:.0%}",
            fontsize=7,
            color=STYLE["accent4"] if is_correct else STYLE["error_color"],
            pad=2,
        )

    fig.suptitle("Sample Predictions", color=STYLE["text_color"], fontsize=12)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing multiple models on a given metric.

    Args:
        results:   Dict[model_name → {metric_name → value}]
        metric:    Metric key to compare (e.g., 'accuracy', 'cer', 'wer').
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    model_names = list(results.keys())
    values = [results[m].get(metric, 0.0) for m in model_names]
    colors = [STYLE["accent1"], STYLE["accent2"], STYLE["accent3"],
              STYLE["accent4"], "#f59e0b"][:len(model_names)]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, ax)

    bars = ax.bar(model_names, values, color=colors, alpha=0.9, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom",
                color=STYLE["text_color"], fontsize=10, fontweight="bold")

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.title()}", color=STYLE["text_color"])

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Attention map overlay (ViT)
# ---------------------------------------------------------------------------

def plot_attention_rollout(
    image: np.ndarray,
    rollout: np.ndarray,
    patch_size: int = 4,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Overlay ViT attention rollout heatmap on the original image.

    Args:
        image:      Original grayscale image (H, W), float [0, 1].
        rollout:    1-D attention rollout array (num_patches,).
        patch_size: Patch size used in the ViT model.
        save_path:  Optional save path.

    Returns:
        Matplotlib Figure.
    """
    h, w = image.shape
    num_patches_side = h // patch_size
    attn_map = rollout.reshape(num_patches_side, num_patches_side)

    # Upsample to image size
    import cv2
    attn_up = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)
    attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(STYLE["bg_color"])

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original", color=STYLE["text_color"])
    axes[0].axis("off")

    axes[1].imshow(attn_up, cmap="plasma")
    axes[1].set_title("Attention Rollout", color=STYLE["text_color"])
    axes[1].axis("off")

    overlay = plt.cm.plasma(attn_up)[..., :3] * 0.5 + np.stack([image] * 3, axis=-1) * 0.5
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay", color=STYLE["text_color"])
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig
