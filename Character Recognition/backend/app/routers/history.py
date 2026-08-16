"""
History router — GET /history, DELETE /history/{id}.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.prediction import Prediction
from backend.app.schemas.prediction import PredictionRecord
from backend.app.services.export_service import ExportService
from fastapi.responses import Response

router = APIRouter(prefix="/api", tags=["history"])


@router.get(
    "/history",
    response_model=List[PredictionRecord],
    summary="Retrieve OCR prediction history",
)
async def get_history(
    db: Session = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0),
    input_type: Optional[str] = Query(default=None, description="Filter by: character|word|sentence"),
    session_id: Optional[str] = Query(default=None, description="Filter by session ID"),
):
    """
    Retrieve past OCR predictions, optionally filtered by type or session.
    Results are returned newest-first.
    """
    q = db.query(Prediction)
    if input_type:
        q = q.filter(Prediction.input_type == input_type)
    if session_id:
        q = q.filter(Prediction.session_id == session_id)

    records = q.order_by(Prediction.created_at.desc()).offset(offset).limit(limit).all()
    return [PredictionRecord.model_validate(r) for r in records]


@router.delete(
    "/history/{prediction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a prediction record",
)
async def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    record = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    db.delete(record)
    db.commit()


@router.get(
    "/history/stats",
    summary="Get aggregate stats from prediction history",
)
async def history_stats(db: Session = Depends(get_db)):
    """Returns total predictions, per-type breakdown, mean confidence, mean processing time."""
    from sqlalchemy import func
    total = db.query(Prediction).count()
    by_type = (
        db.query(Prediction.input_type, func.count(Prediction.id))
        .group_by(Prediction.input_type)
        .all()
    )
    avg_conf = db.query(func.avg(Prediction.confidence)).scalar() or 0.0
    avg_ms = db.query(func.avg(Prediction.processing_ms)).scalar() or 0.0

    return {
        "total_predictions": total,
        "by_type": {t: c for t, c in by_type},
        "mean_confidence": round(float(avg_conf), 4),
        "mean_processing_ms": round(float(avg_ms), 2),
    }


@router.post(
    "/export",
    summary="Export OCR text to TXT, PDF, or DOCX",
)
async def export_text(
    text: str,
    format: str = Query(default="txt", pattern="^(txt|pdf|docx)$"),
    filename: Optional[str] = None,
):
    """Generate and download a formatted export of OCR output text."""
    svc = ExportService()
    content, media_type, ext = svc.export(text, format, "OCR Export")
    fn = (filename or "ocr_result") + ext
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{fn}"'},
    )
