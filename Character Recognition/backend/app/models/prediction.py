"""SQLAlchemy ORM models for the OCR system database."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from backend.app.database import Base


class Prediction(Base):
    """Stores each OCR prediction made through the API."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    image_path = Column(String(512), nullable=True)
    input_type = Column(String(16), nullable=False)   # 'character' | 'word' | 'sentence'
    raw_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    model_used = Column(String(64), nullable=True)
    engine_used = Column(String(64), nullable=True)   # 'custom' | 'easyocr' | 'tesseract'
    processing_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "image_path": self.image_path,
            "input_type": self.input_type,
            "raw_text": self.raw_text,
            "confidence": round(self.confidence, 4) if self.confidence else None,
            "model_used": self.model_used,
            "engine_used": self.engine_used,
            "processing_ms": self.processing_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelMetric(Base):
    """Stores evaluation metrics for trained models."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(64), nullable=False, index=True)
    accuracy = Column(Float, nullable=True)
    cer = Column(Float, nullable=True)      # Character Error Rate
    wer = Column(Float, nullable=True)      # Word Error Rate
    ece = Column(Float, nullable=True)      # Expected Calibration Error
    num_params = Column(Integer, nullable=True)
    eval_date = Column(DateTime, server_default=func.now(), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "cer": self.cer,
            "wer": self.wer,
            "ece": self.ece,
            "num_params": self.num_params,
            "eval_date": self.eval_date.isoformat() if self.eval_date else None,
        }
