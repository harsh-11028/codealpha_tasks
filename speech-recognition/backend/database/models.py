"""
database/models.py — SQLAlchemy ORM models for the SER application.

Tables:
  - uploaded_files       : Tracks every audio file uploaded to the system
  - prediction_history   : Stores every emotion prediction result
  - model_registry       : Tracks trained model versions and their metrics
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.database import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ── UploadedFile ─────────────────────────────────────────────────────────────
class UploadedFile(Base):
    """Tracks every audio file uploaded through the API."""

    __tablename__ = "uploaded_files"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    filename: Mapped[str] = mapped_column(String(256), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channels: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship
    predictions: Mapped[list["PredictionHistory"]] = relationship(
        "PredictionHistory", back_populates="uploaded_file", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<UploadedFile id={self.id} name={self.original_filename}>"


# ── PredictionHistory ─────────────────────────────────────────────────────────
class PredictionHistory(Base):
    """Stores every emotion prediction result with full detail."""

    __tablename__ = "prediction_history"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    # Source info
    file_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("uploaded_files.id"), nullable=True, index=True
    )
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="upload"
    )  # upload | live | stream
    request_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # Prediction result
    predicted_emotion: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON: {"happy": 0.87, "sad": 0.05, ...}
    all_probabilities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Model used
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Audio metadata
    audio_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Feature data for visualization (stored as JSON)
    mfcc_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    spectrogram_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    waveform_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Explainability
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Performance
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    preprocessing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # Relationship
    uploaded_file: Mapped[Optional["UploadedFile"]] = relationship(
        "UploadedFile", back_populates="predictions"
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} emotion={self.predicted_emotion} "
            f"confidence={self.confidence:.2f}>"
        )


# ── ModelRegistry ─────────────────────────────────────────────────────────────
class ModelRegistry(Base):
    """Tracks all trained model versions, their metrics, and active status."""

    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    architecture: Mapped[str] = mapped_column(String(64), nullable=False)
    # e.g. cnn | cnn_lstm | bilstm | cnn_attention | wav2vec2

    # File paths
    model_path: Mapped[str] = mapped_column(String(512), nullable=False)
    config_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Training metrics (JSON for extensibility)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # {"accuracy": 0.87, "f1": 0.86, "precision": 0.87, "recall": 0.86}

    # Dataset info
    datasets_used: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    num_classes: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    num_parameters: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Training metadata
    epochs_trained: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    training_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    framework: Mapped[str] = mapped_column(String(32), default="pytorch")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry name={self.name} v={self.version} "
            f"active={self.is_active} best={self.is_best}>"
        )
