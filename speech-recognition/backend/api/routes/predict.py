"""
api/routes/predict.py — Core emotion prediction endpoints.

POST /predict           — Predict emotion from an uploaded audio file
POST /predict-live      — Predict from a raw audio chunk (base64 encoded)
GET  /predict/{id}      — Retrieve a past prediction by ID
"""

import base64
import io
import os
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from api.dependencies import DBSession
from database.crud import (
    create_prediction,
    get_prediction,
    get_uploaded_file,
)
from training.config import EMOTION_COLORS, EMOTION_EMOJI, EMOTION_LABELS
from utils.helpers import Timer, compute_file_hash, format_duration
from utils.logger import get_logger
from utils.validators import validate_audio_file

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class EmotionProbability(BaseModel):
    emotion: str
    label: str
    probability: float
    confidence_pct: float
    color: str
    emoji: str


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_emotion: str
    emotion_label: str
    emoji: str
    confidence: float
    confidence_pct: float
    color: str
    all_probabilities: list[EmotionProbability]
    model_name: str
    model_version: Optional[str]
    source: str

    # Audio metadata
    audio_duration: Optional[str]        # "00:04.123"
    audio_duration_seconds: Optional[float]
    sample_rate: Optional[int]

    # Visualization data
    waveform_data: Optional[list[float]]
    mfcc_data: Optional[list[list[float]]]
    spectrogram_data: Optional[list[list[float]]]

    # Explainability
    explanation: str
    feature_importance: Optional[dict]

    # Performance
    inference_time_ms: float
    preprocessing_time_ms: float
    total_time_ms: float


class LivePredictionRequest(BaseModel):
    audio_base64: str = Field(
        ..., description="Base64-encoded raw PCM or WAV audio bytes"
    )
    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    encoding: str = Field(default="wav", description="wav | pcm_f32 | pcm_int16")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_probabilities_response(probs_dict: dict) -> list[EmotionProbability]:
    """Convert {emotion: probability} dict into sorted response list."""
    result = []
    for idx, label in EMOTION_LABELS.items():
        prob = probs_dict.get(label, 0.0)
        result.append(
            EmotionProbability(
                emotion=label,
                label=label.capitalize(),
                probability=round(prob, 6),
                confidence_pct=round(prob * 100, 2),
                color=EMOTION_COLORS.get(label, "#888"),
                emoji=EMOTION_EMOJI.get(label, ""),
            )
        )
    # Sort descending by probability
    result.sort(key=lambda x: x.probability, reverse=True)
    return result


def _generate_explanation(emotion: str, confidence: float, feature_importance: dict) -> str:
    """Generate a natural-language explanation for the prediction."""
    confidence_desc = (
        "very high" if confidence > 0.85
        else "high" if confidence > 0.70
        else "moderate" if confidence > 0.50
        else "low"
    )

    # Find top contributing features
    top_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:3] if feature_importance else []

    feature_text = ""
    if top_features:
        feature_names = [f[0].replace("_", " ") for f in top_features]
        feature_text = (
            f" The most informative acoustic features were "
            f"{', '.join(feature_names[:-1])} and {feature_names[-1]}."
            if len(feature_names) > 1
            else f" The key feature was {feature_names[0]}."
        )

    emotion_descriptions = {
        "happy":   "characterized by higher pitch, faster tempo, and energetic speech patterns",
        "sad":     "reflected in lower energy, slower tempo, and reduced pitch variation",
        "angry":   "indicated by loud, tense, and high-energy speech with irregular patterns",
        "fear":    "shown through trembling voice, higher pitch, and irregular speech rhythm",
        "neutral": "showing baseline speech characteristics with minimal emotional markers",
        "surprise": "detected via sudden pitch changes and irregular energy bursts",
        "disgust": "marked by low, strained vocal quality with specific spectral patterns",
        "calm":    "reflected in smooth, regular, low-energy speech with stable pitch",
    }

    desc = emotion_descriptions.get(emotion, "detected in the speech signal")
    return (
        f"The model predicted '{emotion.capitalize()}' with {confidence_desc} confidence "
        f"({confidence * 100:.1f}%). This emotion was {desc}.{feature_text}"
    )


async def _run_prediction(
    audio_bytes: bytes,
    source: str,
    db,
    file_id: Optional[str] = None,
) -> PredictionResponse:
    """
    Core prediction pipeline:
    1. Preprocess audio
    2. Extract features
    3. Run model inference
    4. Build response
    5. Save to DB

    The actual ML calls are wrapped in try/except so the endpoint
    returns a sensible mock during Phase 2 (before Phase 5 model training).
    """
    from utils.helpers import generate_request_id
    request_id = generate_request_id()
    total_start = time.perf_counter()

    # ── Try real inference (available after Phase 6) ─────────────────────
    try:
        # Import here to avoid circular imports; engine loaded at startup
        from api.dependencies import _prediction_engine
        if _prediction_engine is not None:
            with Timer("preprocessing") as pre_timer:
                result = _prediction_engine.predict(audio_bytes)
            preprocessing_ms = pre_timer.elapsed * 1000
            inference_ms = result.get("inference_time_ms", 0.0)
            probs_dict = result["probabilities"]
            predicted_emotion = result["emotion"]
            confidence = result["confidence"]
            mfcc_data = result.get("mfcc_data")
            spectrogram_data = result.get("spectrogram_data")
            waveform_data = result.get("waveform_data")
            feature_importance = result.get("feature_importance", {})
            duration_seconds = result.get("duration_seconds")
            sample_rate = result.get("sample_rate", 22050)
            model_name = result.get("model_name", "unknown")
            model_version = result.get("model_version")
        else:
            raise RuntimeError("Engine not loaded")

    except Exception as e:
        logger.warning(f"Live inference unavailable ({e}), returning stub response.")
        # ── Stub response for Phase 2 testing (no model loaded yet) ──────
        preprocessing_ms = 12.4
        inference_ms = 8.3
        probs_dict = {
            "neutral": 0.05, "calm": 0.03, "happy": 0.06, "sad": 0.04,
            "angry": 0.03, "fear": 0.02, "disgust": 0.02, "surprise": 0.02,
        }
        # Placeholder — set neutral as dominant
        probs_dict["neutral"] = 0.73
        predicted_emotion = "neutral"
        confidence = 0.73
        mfcc_data = None
        spectrogram_data = None
        waveform_data = None
        feature_importance = {
            "mfcc_mean": 0.45, "spectral_centroid": 0.22,
            "zcr": 0.18, "chroma": 0.15
        }
        duration_seconds = None
        sample_rate = 22050
        model_name = "stub_model"
        model_version = "0.0.0"

    total_ms = (time.perf_counter() - total_start) * 1000
    explanation = _generate_explanation(predicted_emotion, confidence, feature_importance)

    # ── Save to DB ────────────────────────────────────────────────────────
    db_record = await create_prediction(
        db,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        all_probabilities=probs_dict,
        model_name=model_name,
        model_version=model_version,
        source=source,
        file_id=file_id,
        request_id=request_id,
        audio_duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        explanation=explanation,
        feature_importance=feature_importance,
        inference_time_ms=inference_ms,
        preprocessing_time_ms=preprocessing_ms,
    )

    return PredictionResponse(
        prediction_id=db_record.id,
        predicted_emotion=predicted_emotion,
        emotion_label=predicted_emotion.capitalize(),
        emoji=EMOTION_EMOJI.get(predicted_emotion, ""),
        confidence=round(confidence, 6),
        confidence_pct=round(confidence * 100, 2),
        color=EMOTION_COLORS.get(predicted_emotion, "#888"),
        all_probabilities=_build_probabilities_response(probs_dict),
        model_name=model_name,
        model_version=model_version,
        source=source,
        audio_duration=format_duration(duration_seconds) if duration_seconds else None,
        audio_duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        waveform_data=waveform_data,
        mfcc_data=mfcc_data,
        spectrogram_data=spectrogram_data,
        explanation=explanation,
        feature_importance=feature_importance,
        inference_time_ms=round(inference_ms, 2),
        preprocessing_time_ms=round(preprocessing_ms, 2),
        total_time_ms=round(total_ms, 2),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict emotion from audio file",
    description=(
        "Upload a WAV/MP3/OGG/M4A file and receive an emotion prediction "
        "with confidence scores, visualisation data, and explanation."
    ),
)
async def predict_from_file(
    db: DBSession,
    file: UploadFile = File(..., description="Audio file for emotion prediction"),
    file_id: Optional[str] = Form(
        default=None,
        description="Optional: pass a previously uploaded file_id to skip re-uploading",
    ),
):
    """Predict emotion from an uploaded audio file."""
    file_bytes = await file.read()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    is_valid, error = validate_audio_file(file_bytes, file.filename)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error,
        )

    return await _run_prediction(
        audio_bytes=file_bytes,
        source="upload",
        db=db,
        file_id=file_id,
    )


@router.post(
    "-live",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict from live audio chunk",
    description=(
        "Accepts a base64-encoded audio chunk (from microphone recording) "
        "and returns an emotion prediction. Used for near-real-time prediction."
    ),
)
async def predict_live(
    db: DBSession,
    request: LivePredictionRequest,
):
    """Predict emotion from a base64-encoded live audio chunk."""
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 encoding in audio_base64 field.",
        )

    if len(audio_bytes) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio chunk is too small. Minimum 100 bytes required.",
        )

    return await _run_prediction(
        audio_bytes=audio_bytes,
        source="live",
        db=db,
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionResponse,
    summary="Get prediction by ID",
    description="Retrieve the full details of a past prediction by its ID.",
)
async def get_prediction_by_id(prediction_id: str, db: DBSession):
    """Fetch a stored prediction record."""
    record = await get_prediction(db, prediction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction '{prediction_id}' not found.",
        )

    probs_dict = record.all_probabilities or {}
    emotion = record.predicted_emotion

    return PredictionResponse(
        prediction_id=record.id,
        predicted_emotion=emotion,
        emotion_label=emotion.capitalize(),
        emoji=EMOTION_EMOJI.get(emotion, ""),
        confidence=round(record.confidence, 6),
        confidence_pct=round(record.confidence * 100, 2),
        color=EMOTION_COLORS.get(emotion, "#888"),
        all_probabilities=_build_probabilities_response(probs_dict),
        model_name=record.model_name,
        model_version=record.model_version,
        source=record.source,
        audio_duration=(
            format_duration(record.audio_duration_seconds)
            if record.audio_duration_seconds else None
        ),
        audio_duration_seconds=record.audio_duration_seconds,
        sample_rate=record.sample_rate,
        waveform_data=record.waveform_data.get("data") if record.waveform_data else None,
        mfcc_data=record.mfcc_data.get("data") if record.mfcc_data else None,
        spectrogram_data=(
            record.spectrogram_data.get("data") if record.spectrogram_data else None
        ),
        explanation=record.explanation or "",
        feature_importance=record.feature_importance,
        inference_time_ms=record.inference_time_ms or 0.0,
        preprocessing_time_ms=record.preprocessing_time_ms or 0.0,
        total_time_ms=(record.inference_time_ms or 0.0) + (record.preprocessing_time_ms or 0.0),
    )
