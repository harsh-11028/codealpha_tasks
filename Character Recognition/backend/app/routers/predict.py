"""
Prediction routers — POST /predict-character, /predict-word, /predict-sentence.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.middleware.security import rate_limit_middleware, validate_image_upload
from backend.app.schemas.prediction import CharPrediction, SentencePrediction, WordPrediction
from backend.app.services.ocr_service import OCRService

router = APIRouter(prefix="/api", tags=["predictions"])


def _session_id() -> str:
    return str(uuid.uuid4())


@router.post(
    "/predict-character",
    response_model=CharPrediction,
    summary="Predict a single handwritten character",
)
async def predict_character(
    request: Request,
    file: UploadFile = File(..., description="Image of a single handwritten character"),
    session_id: str = Form(default_factory=_session_id),
    db: Session = Depends(get_db),
):
    """
    Accepts a cropped image of a single character and returns:
    - Recognized character
    - Confidence score
    - Top-5 alternative predictions
    """
    rate_limit_middleware(request)
    validate_image_upload(
        file.content_type or "",
        file.size or 0,
        file.filename or "upload",
    )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")

    try:
        svc = OCRService(db)
        result = svc.predict_character(image_bytes, session_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Character prediction failed: {str(exc)}",
        )

    char_data = result.chars[0] if result.chars else {"char": result.text, "confidence": result.confidence}
    return CharPrediction(
        char=char_data.get("char", result.text),
        confidence=result.confidence,
        top_k=[],
    )


@router.post(
    "/predict-word",
    response_model=WordPrediction,
    summary="Predict a handwritten word",
)
async def predict_word(
    request: Request,
    file: UploadFile = File(..., description="Image containing a single handwritten word"),
    session_id: str = Form(default_factory=_session_id),
    db: Session = Depends(get_db),
):
    """
    Accepts an image of a single word and returns:
    - Recognized word text
    - Confidence
    - Per-character predictions
    - Processing time
    """
    rate_limit_middleware(request)
    validate_image_upload(
        file.content_type or "",
        file.size or 0,
        file.filename or "upload",
    )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")

    try:
        svc = OCRService(db)
        result = svc.predict_word(image_bytes, session_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Word prediction failed: {str(exc)}",
        )

    return WordPrediction(
        word=result.text,
        confidence=result.confidence,
        processing_ms=result.processing_ms,
        engine_used=result.engine_used,
        model_used=result.model_used,
    )


@router.post(
    "/predict-sentence",
    response_model=SentencePrediction,
    summary="Predict handwritten sentence or paragraph",
)
async def predict_sentence(
    request: Request,
    file: UploadFile = File(..., description="Image containing handwritten text (sentence/paragraph)"),
    session_id: str = Form(default_factory=_session_id),
    annotate: bool = Form(default=True, description="Return annotated image with bounding boxes"),
    db: Session = Depends(get_db),
):
    """
    Full OCR pipeline:
    - Line detection → Word detection → Character segmentation
    - Returns recognized text with bounding boxes at all levels
    - Includes base64-encoded annotated image
    """
    rate_limit_middleware(request)
    validate_image_upload(
        file.content_type or "",
        file.size or 0,
        file.filename or "upload",
    )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded.")

    try:
        svc = OCRService(db)
        result = svc.predict_sentence(image_bytes, session_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentence prediction failed: {str(exc)}",
        )

    return SentencePrediction(
        text=result.text,
        confidence=result.confidence,
        word_boxes=result.word_boxes,
        char_boxes=result.char_boxes,
        line_boxes=result.line_boxes,
        processing_ms=result.processing_ms,
        engine_used=result.engine_used,
        model_used=result.model_used,
        confidence_stats=result.confidence_stats,
        annotated_image=result.annotated_image if annotate else None,
    )
