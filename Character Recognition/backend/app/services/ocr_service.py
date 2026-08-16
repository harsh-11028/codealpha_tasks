"""
OCR service — business logic layer for OCR operations.
Wraps the OCRPipeline and handles session management.
"""

from __future__ import annotations

import uuid
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.models.prediction import Prediction
from models.ocr.pipeline import OCRPipeline, OCRPipelineResult
from models.training.config import config as ml_config

logger = logging.getLogger(__name__)
settings = get_settings()

# Module-level singleton pipeline
_pipeline: Optional[OCRPipeline] = None


def get_pipeline() -> OCRPipeline:
    """Return initialized OCRPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = OCRPipeline(ml_config)
        _pipeline.initialize()
    return _pipeline


class OCRService:
    """Thin service layer wrapping OCRPipeline with DB persistence."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.pipeline = get_pipeline()

    def _persist(
        self,
        session_id: str,
        image_path: Optional[str],
        input_type: str,
        result: OCRPipelineResult,
    ) -> Prediction:
        record = Prediction(
            session_id=session_id,
            image_path=image_path,
            input_type=input_type,
            raw_text=result.text,
            confidence=result.confidence,
            model_used=result.model_used,
            engine_used=result.engine_used,
            processing_ms=int(result.processing_ms),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def predict_character(
        self,
        image_bytes: bytes,
        session_id: str,
        image_path: Optional[str] = None,
    ) -> OCRPipelineResult:
        result = self.pipeline.run(image_bytes, task="character")
        self._persist(session_id, image_path, "character", result)
        return result

    def predict_word(
        self,
        image_bytes: bytes,
        session_id: str,
        image_path: Optional[str] = None,
    ) -> OCRPipelineResult:
        result = self.pipeline.run(image_bytes, task="word")
        self._persist(session_id, image_path, "word", result)
        return result

    def predict_sentence(
        self,
        image_bytes: bytes,
        session_id: str,
        image_path: Optional[str] = None,
    ) -> OCRPipelineResult:
        result = self.pipeline.run(image_bytes, task="sentence")
        self._persist(session_id, image_path, "sentence", result)
        return result
