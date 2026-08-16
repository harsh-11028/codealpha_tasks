"""
Health and model info routers — GET /health, GET /model-info.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.database import get_db
from backend.app.models.prediction import Prediction
from backend.app.schemas.prediction import HealthResponse, ModelInfoItem, ModelInfoResponse
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="/api", tags=["health"])
settings = get_settings()

_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
)
async def health_check():
    """
    Returns API health status and model readiness.
    Used by load balancers, monitoring systems, and the frontend.
    """
    svc = ModelService()
    return HealthResponse(
        status="ok",
        model_loaded=svc.is_ready(),
        uptime_seconds=round(time.time() - _start_time, 2),
        version=settings.app_version,
        device=settings.device,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="List loaded models and their metrics",
)
async def model_info(db: Session = Depends(get_db)):
    """Returns all loaded models with parameter counts and evaluation metrics."""
    svc = ModelService()
    total_preds = db.query(Prediction).count()
    model_items = [
        ModelInfoItem(**m) for m in svc.get_model_info()
    ]
    return ModelInfoResponse(
        active_model=svc.get_active_model(),
        all_models=model_items,
        total_predictions=total_preds,
    )
