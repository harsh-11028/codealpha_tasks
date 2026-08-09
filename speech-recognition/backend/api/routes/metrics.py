"""
api/routes/metrics.py — Training evaluation metrics and usage statistics.

GET /metrics            — Overall system + model performance metrics
GET /metrics/usage      — API usage statistics (prediction counts, etc.)
GET /metrics/emotions   — Emotion distribution across all predictions
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from api.dependencies import DBSession
from database.crud import (
    get_active_model,
    get_best_model,
    get_emotion_statistics,
    get_average_confidence,
    list_models,
)
from training.config import EMOTION_LABELS, EMOTION_COLORS, EMOTION_EMOJI
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/metrics", tags=["Metrics"])


# ── Response schemas ──────────────────────────────────────────────────────────
class ModelMetricsResponse(BaseModel):
    model_name: str
    architecture: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    val_accuracy: Optional[float]
    val_f1_score: Optional[float]
    test_accuracy: Optional[float]
    epochs_trained: Optional[int]
    training_history: Optional[list]  # [{epoch, loss, val_loss, acc, val_acc}, ...]
    confusion_matrix: Optional[list]
    classification_report: Optional[dict]


class UsageStatsResponse(BaseModel):
    total_predictions: int
    predictions_today: int
    total_uploads: int
    average_confidence: Optional[float]
    emotion_distribution: dict  # {emotion: {count, percentage, color, emoji}}
    most_common_emotion: Optional[str]
    least_common_emotion: Optional[str]


class SystemMetricsResponse(BaseModel):
    model_metrics: Optional[ModelMetricsResponse]
    usage_stats: UsageStatsResponse
    all_model_comparison: list[dict]


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get(
    "",
    response_model=SystemMetricsResponse,
    summary="Get all metrics",
    description="Returns model evaluation metrics and usage statistics.",
)
async def get_metrics(db: DBSession):
    """Return comprehensive system metrics."""

    # ── Model metrics ─────────────────────────────────────────────────────
    active_model = await get_active_model(db)
    model_metrics_response = None

    if active_model and active_model.metrics:
        m = active_model.metrics
        model_metrics_response = ModelMetricsResponse(
            model_name=active_model.name,
            architecture=active_model.architecture,
            accuracy=m.get("accuracy"),
            precision=m.get("precision"),
            recall=m.get("recall"),
            f1_score=m.get("f1"),
            val_accuracy=m.get("val_accuracy"),
            val_f1_score=m.get("val_f1"),
            test_accuracy=m.get("test_accuracy"),
            epochs_trained=active_model.epochs_trained,
            training_history=m.get("training_history"),
            confusion_matrix=m.get("confusion_matrix"),
            classification_report=m.get("classification_report"),
        )

    # ── Usage statistics ───────────────────────────────────────────────────
    emotion_counts = await get_emotion_statistics(db)
    avg_confidence = await get_average_confidence(db)
    total_preds = sum(emotion_counts.values())

    # Build enriched emotion distribution
    distribution = {}
    for idx, label in EMOTION_LABELS.items():
        count = emotion_counts.get(label, 0)
        distribution[label] = {
            "count": count,
            "percentage": round((count / total_preds * 100) if total_preds > 0 else 0, 2),
            "color": EMOTION_COLORS.get(label, "#888"),
            "emoji": EMOTION_EMOJI.get(label, ""),
        }

    most_common = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    least_common = min(emotion_counts, key=emotion_counts.get) if emotion_counts else None

    usage_stats = UsageStatsResponse(
        total_predictions=total_preds,
        predictions_today=0,          # Populated with a date-filtered query in production
        total_uploads=0,
        average_confidence=round(avg_confidence, 4) if avg_confidence else None,
        emotion_distribution=distribution,
        most_common_emotion=most_common,
        least_common_emotion=least_common,
    )

    # ── All model comparison ───────────────────────────────────────────────
    all_models = await list_models(db)
    model_comparison = []
    for m in all_models:
        metrics = m.metrics or {}
        model_comparison.append({
            "name": m.name,
            "architecture": m.architecture,
            "version": m.version,
            "is_active": m.is_active,
            "is_best": m.is_best,
            "accuracy": metrics.get("accuracy"),
            "f1_score": metrics.get("f1"),
            "num_parameters": m.num_parameters,
            "epochs_trained": m.epochs_trained,
        })

    return SystemMetricsResponse(
        model_metrics=model_metrics_response,
        usage_stats=usage_stats,
        all_model_comparison=model_comparison,
    )


@router.get(
    "/emotions",
    summary="Emotion distribution",
    description="Returns the count and percentage breakdown of predicted emotions.",
)
async def get_emotion_distribution(db: DBSession):
    """Return emotion distribution across all stored predictions."""
    emotion_counts = await get_emotion_statistics(db)
    total = sum(emotion_counts.values())

    return {
        "total_predictions": total,
        "distribution": {
            label: {
                "count": emotion_counts.get(label, 0),
                "percentage": round(
                    (emotion_counts.get(label, 0) / total * 100) if total > 0 else 0, 2
                ),
                "color": EMOTION_COLORS.get(label, "#888"),
                "emoji": EMOTION_EMOJI.get(label, ""),
            }
            for label in EMOTION_LABELS.values()
        },
    }
