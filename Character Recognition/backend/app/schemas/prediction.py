"""Pydantic schemas for request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., ge=1)
    h: int = Field(..., ge=1)


# ---------------------------------------------------------------------------
# Prediction responses
# ---------------------------------------------------------------------------

class CharPrediction(BaseModel):
    char: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = None
    top_k: List[Tuple[str, float]] = []


class WordPrediction(BaseModel):
    word: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = None
    char_predictions: List[CharPrediction] = []
    processing_ms: float = 0.0
    engine_used: str = "custom"
    model_used: str = ""


class SentencePrediction(BaseModel):
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    word_boxes: List[Dict[str, Any]] = []
    char_boxes: List[Dict[str, Any]] = []
    line_boxes: List[Dict[str, Any]] = []
    processing_ms: float = 0.0
    engine_used: str = "custom"
    model_used: str = ""
    confidence_stats: Dict[str, float] = {}
    annotated_image: Optional[str] = None   # base64 PNG


# ---------------------------------------------------------------------------
# Upload response
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    session_id: str
    image_url: str
    filename: str
    size_bytes: int
    message: str = "Upload successful"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class PredictionRecord(BaseModel):
    id: int
    session_id: str
    input_type: str
    raw_text: Optional[str] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    engine_used: Optional[str] = None
    processing_ms: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------

class WebcamRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image (JPEG/PNG)")
    task: str = Field(default="sentence", pattern="^(character|word|sentence)$")
    session_id: Optional[str] = None

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        import base64
        # Strip data URI prefix if present
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("Invalid base64 image data")
        return v


# ---------------------------------------------------------------------------
# Health / model info
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    uptime_seconds: float
    version: str
    device: str


class ModelInfoItem(BaseModel):
    name: str
    parameters: int
    metrics: Dict[str, float] = {}
    device: str


class ModelInfoResponse(BaseModel):
    active_model: str
    all_models: List[ModelInfoItem]
    total_predictions: int


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    text: str
    format: str = Field(default="txt", pattern="^(txt|pdf|docx)$")
    session_id: Optional[str] = None
    filename: Optional[str] = None
