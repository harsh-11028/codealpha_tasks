"""
api/routes/model_info.py — Model information and registry endpoints.

GET /model-info         — Currently active model details
GET /models             — All registered models
GET /models/{model_id}  — Single model details
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from api.dependencies import DBSession
from database.crud import get_active_model, get_best_model, list_models, get_prediction
from utils.logger import get_logger
from training.config import EMOTION_LABELS, EMOTION_EMOJI, EMOTION_COLORS, NUM_CLASSES

logger = get_logger(__name__)
router = APIRouter(prefix="/model", tags=["Model"])


# ── Response schemas ─────────────────────────────────────────────────────────
class ModelInfoResponse(BaseModel):
    id: str
    name: str
    version: str
    architecture: str
    is_active: bool
    is_best: bool
    num_classes: int
    num_parameters: Optional[int]
    epochs_trained: Optional[int]
    metrics: Optional[dict]
    datasets_used: Optional[dict]
    framework: str
    created_at: str

    model_config = {"from_attributes": True}


class SystemInfoResponse(BaseModel):
    active_model: Optional[ModelInfoResponse]
    best_model: Optional[ModelInfoResponse]
    all_models: list[ModelInfoResponse]
    emotion_classes: dict
    num_classes: int


# ── Routes ───────────────────────────────────────────────────────────────────
@router.get(
    "-info",
    response_model=SystemInfoResponse,
    summary="Get model info",
    description="Returns the active model, best model, and all registered models.",
)
async def get_model_info(db: DBSession):
    """Return comprehensive model registry information."""
    active = await get_active_model(db)
    best = await get_best_model(db)
    all_models = await list_models(db)

    def _to_response(m) -> Optional[ModelInfoResponse]:
        if m is None:
            return None
        return ModelInfoResponse(
            id=m.id,
            name=m.name,
            version=m.version,
            architecture=m.architecture,
            is_active=m.is_active,
            is_best=m.is_best,
            num_classes=m.num_classes,
            num_parameters=m.num_parameters,
            epochs_trained=m.epochs_trained,
            metrics=m.metrics,
            datasets_used=m.datasets_used,
            framework=m.framework,
            created_at=m.created_at.isoformat(),
        )

    # Build emotion class info for frontend
    emotion_classes = {
        str(idx): {
            "label": label,
            "emoji": EMOTION_EMOJI.get(label, ""),
            "color": EMOTION_COLORS.get(label, "#888"),
        }
        for idx, label in EMOTION_LABELS.items()
    }

    return SystemInfoResponse(
        active_model=_to_response(active),
        best_model=_to_response(best),
        all_models=[_to_response(m) for m in all_models],
        emotion_classes=emotion_classes,
        num_classes=NUM_CLASSES,
    )
