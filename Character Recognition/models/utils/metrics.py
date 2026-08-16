"""
Evaluation metrics for OCR models.

Implements:
  - Classification: Accuracy, Top-K Accuracy, Per-class Accuracy
  - Sequence recognition: Character Error Rate (CER), Word Error Rate (WER)
  - Confidence calibration: Expected Calibration Error (ECE)
  - Confusion matrix computation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks running mean of a scalar (loss, accuracy, etc.)."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> List[float]:
    """
    Compute top-k accuracy percentages.

    Args:
        output: Logit tensor of shape (N, C).
        target: Ground truth labels of shape (N,).
        topk:   Tuple of k values.

    Returns:
        List of accuracy percentages for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results: List[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(float(correct_k.mul_(100.0 / batch_size).item()))
        return results


def per_class_accuracy(
    all_preds: List[int],
    all_targets: List[int],
    num_classes: int,
    label_map: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """
    Compute per-class accuracy.

    Args:
        all_preds:   Flat list of predicted class indices.
        all_targets: Flat list of true class indices.
        num_classes: Total number of classes.
        label_map:   Optional mapping from class index to character string.

    Returns:
        Dict mapping class name (or index string) to accuracy float.
    """
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)

    for pred, target in zip(all_preds, all_targets):
        total[target] += 1
        if pred == target:
            correct[target] += 1

    result: Dict[str, float] = {}
    for i in range(num_classes):
        key = label_map[i] if (label_map and i in label_map) else str(i)
        result[key] = float(correct[i] / max(total[i], 1)) * 100.0

    return result


def compute_confusion_matrix(
    all_preds: List[int],
    all_targets: List[int],
    num_classes: int,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        all_preds:   Predicted class indices.
        all_targets: True class indices.
        num_classes: Number of classes.
        normalize:   Normalize rows to sum to 1.

    Returns:
        ndarray of shape (num_classes, num_classes).
        cm[i][j] = number (or fraction) of class i predicted as j.
    """
    labels = list(range(num_classes))
    try:
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        cm = sk_confusion_matrix(all_targets, all_preds, labels=labels)
    except ImportError:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(all_targets, all_preds):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[int(t), int(p)] += 1
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / np.maximum(row_sums, 1)
    return cm.astype(np.float32)


# ---------------------------------------------------------------------------
# Sequence / OCR metrics
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def character_error_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Character Error Rate (CER) — standard OCR accuracy metric.

    CER = edit_distance(prediction, reference) / len(reference)

    Averaged over all samples. Lower is better (0.0 = perfect).

    Args:
        predictions: List of predicted text strings.
        references:  List of ground truth text strings.

    Returns:
        Mean CER across all pairs (float in [0, ∞)).
    """
    if not predictions:
        return 0.0
    cer_sum = 0.0
    for pred, ref in zip(predictions, references):
        if not ref:
            continue
        cer_sum += _edit_distance(pred, ref) / len(ref)
    return cer_sum / len(predictions)


def word_error_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Word Error Rate (WER) — measures accuracy at the word level.

    WER = word_edit_distance(prediction, reference) / num_words_in_reference

    Args:
        predictions: List of predicted sentence strings.
        references:  List of ground truth sentence strings.

    Returns:
        Mean WER across all pairs (float in [0, ∞)).
    """
    if not predictions:
        return 0.0
    wer_sum = 0.0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        if not ref_words:
            continue
        wer_sum += _edit_distance(" ".join(pred_words), " ".join(ref_words)) / len(ref_words)
    return wer_sum / len(predictions)


def sequence_accuracy(
    predictions: List[str],
    references: List[str],
    case_sensitive: bool = True,
) -> float:
    """
    Exact-match sequence accuracy — fraction where prediction == reference.

    Args:
        predictions:    Predicted strings.
        references:     Ground truth strings.
        case_sensitive: If False, compare lowercase.

    Returns:
        Fraction of exactly correct predictions in [0, 1].
    """
    if not predictions:
        return 0.0
    if not case_sensitive:
        predictions = [p.lower() for p in predictions]
        references = [r.lower() for r in references]
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)


# ---------------------------------------------------------------------------
# Calibration metric
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    num_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures how well model confidence aligns with actual accuracy.
    A perfectly calibrated model has ECE = 0.

    Args:
        confidences: Array of predicted confidence scores in [0, 1].
        correct:     Binary array (1 = correct prediction, 0 = wrong).
        num_bins:    Number of confidence bins.

    Returns:
        ECE score (lower = better calibrated).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = correct[in_bin].mean()
        bin_weight = in_bin.sum() / n
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Batch-level evaluation helper
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Accumulates predictions and targets across batches for epoch-level metrics.

    Usage:
        tracker = MetricsTracker(num_classes=47)
        for batch in loader:
            logits = model(images)
            tracker.update(logits, targets)
        metrics = tracker.compute()
    """

    def __init__(self, num_classes: int, top_k: int = 5) -> None:
        self.num_classes = num_classes
        self.top_k = top_k
        self.reset()

    def reset(self) -> None:
        self.all_preds: List[int] = []
        self.all_targets: List[int] = []
        self.all_confs: List[float] = []
        self.loss_meter = AverageMeter("loss")

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ) -> None:
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        self.all_preds.extend(preds.cpu().tolist())
        self.all_targets.extend(targets.cpu().tolist())
        self.all_confs.extend(confs.cpu().tolist())

        if loss is not None:
            self.loss_meter.update(loss, n=logits.size(0))

    def compute(self) -> Dict[str, float]:
        preds_t = torch.tensor(self.all_preds)
        targets_t = torch.tensor(self.all_targets)
        logits_mock = torch.zeros(len(self.all_preds), self.num_classes)
        for i, p in enumerate(self.all_preds):
            logits_mock[i, p] = 1.0

        top1, top5 = accuracy(logits_mock, targets_t, topk=(1, min(5, self.num_classes)))

        confs = np.array(self.all_confs)
        correct_mask = np.array(self.all_preds) == np.array(self.all_targets)
        ece = expected_calibration_error(confs, correct_mask.astype(float))

        return {
            "accuracy": top1,
            "top5_accuracy": top5,
            "loss": self.loss_meter.avg,
            "ece": ece,
            "mean_confidence": float(confs.mean()),
            "num_samples": len(self.all_preds),
        }

    def get_confusion_matrix(self, normalize: bool = True) -> np.ndarray:
        return compute_confusion_matrix(
            self.all_preds, self.all_targets, self.num_classes, normalize
        )
