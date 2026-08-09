"""
api/routes/upload.py — Audio file upload and management.

POST /upload            — Upload an audio file (validate, store, return metadata)
GET  /upload/{file_id}  — Retrieve metadata for a previously uploaded file
GET  /history           — Paginated prediction history
"""

import io
import os
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from api.dependencies import DBSession
from database.crud import (
    create_uploaded_file,
    get_file_by_hash,
    get_prediction_history,
    get_uploaded_file,
)
from utils.helpers import compute_file_hash, ensure_dir
from utils.logger import get_logger
from utils.validators import sanitize_filename, validate_audio_file

logger = get_logger(__name__)
router = APIRouter(tags=["Upload & History"])

# Upload storage directory
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
ensure_dir(UPLOAD_DIR)


# ── Response schemas ──────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    file_id: str
    original_filename: str
    filename: str
    mime_type: str
    file_size_kb: float
    duration_seconds: Optional[float]
    sample_rate: Optional[int]
    channels: Optional[int]
    is_duplicate: bool
    storage_path: str
    created_at: str


class PredictionSummary(BaseModel):
    id: str
    predicted_emotion: str
    confidence: float
    model_name: str
    source: str
    audio_duration_seconds: Optional[float]
    created_at: str


class HistoryResponse(BaseModel):
    predictions: list[PredictionSummary]
    total: int
    limit: int
    offset: int
    has_more: bool


# ── Helper ────────────────────────────────────────────────────────────────────
def _probe_audio(file_bytes: bytes) -> tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Use soundfile to extract audio metadata (duration, sample_rate, channels).
    Returns (duration_seconds, sample_rate, channels) — all None on failure.
    """
    try:
        with sf.SoundFile(io.BytesIO(file_bytes)) as f:
            duration = len(f) / f.samplerate
            return duration, f.samplerate, f.channels
    except Exception:
        return None, None, None


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an audio file",
    description=(
        "Accepts WAV, MP3, OGG, M4A audio files up to 50 MB. "
        "Validates file type via magic bytes. Detects duplicates by content hash."
    ),
)
async def upload_audio(
    db: DBSession,
    file: UploadFile = File(..., description="Audio file to upload"),
):
    """
    Upload and validate an audio file.

    1. Read bytes
    2. Security validate (size, extension, MIME magic bytes)
    3. Compute SHA-256 hash for duplicate detection
    4. Probe audio metadata
    5. Save to disk
    6. Persist record to DB
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    # 1. Read file bytes
    file_bytes = await file.read()

    # 2. Validate
    is_valid, error_msg = validate_audio_file(file_bytes, file.filename)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_msg,
        )

    mime_type = file.content_type or "audio/unknown"

    # 3. Duplicate detection
    file_hash = compute_file_hash(file_bytes)
    existing = await get_file_by_hash(db, file_hash)
    is_duplicate = existing is not None

    if is_duplicate and existing:
        logger.info(f"Duplicate upload detected: {file.filename} → {existing.id}")
        return UploadResponse(
            file_id=existing.id,
            original_filename=existing.original_filename,
            filename=existing.filename,
            mime_type=existing.mime_type,
            file_size_kb=round(existing.file_size_bytes / 1024, 2),
            duration_seconds=existing.duration_seconds,
            sample_rate=existing.sample_rate,
            channels=existing.channels,
            is_duplicate=True,
            storage_path=existing.storage_path,
            created_at=existing.created_at.isoformat(),
        )

    # 4. Probe audio metadata
    duration, sample_rate, channels = _probe_audio(file_bytes)

    # 5. Save to disk
    safe_name = sanitize_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{safe_name}"
    storage_path = UPLOAD_DIR / unique_filename
    storage_path.write_bytes(file_bytes)
    logger.info(f"Saved upload: {storage_path} ({len(file_bytes) / 1024:.1f} KB)")

    # 6. Persist to DB
    record = await create_uploaded_file(
        db,
        filename=unique_filename,
        original_filename=file.filename,
        mime_type=mime_type,
        file_size_bytes=len(file_bytes),
        storage_path=str(storage_path),
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        file_hash=file_hash,
    )

    return UploadResponse(
        file_id=record.id,
        original_filename=record.original_filename,
        filename=record.filename,
        mime_type=record.mime_type,
        file_size_kb=round(record.file_size_bytes / 1024, 2),
        duration_seconds=record.duration_seconds,
        sample_rate=record.sample_rate,
        channels=record.channels,
        is_duplicate=False,
        storage_path=str(storage_path),
        created_at=record.created_at.isoformat(),
    )


@router.get(
    "/upload/{file_id}",
    summary="Get file metadata",
    description="Returns metadata for a previously uploaded audio file.",
)
async def get_file_metadata(file_id: str, db: DBSession):
    """Retrieve stored metadata for an uploaded file."""
    record = await get_uploaded_file(db, file_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID '{file_id}' not found.",
        )
    return {
        "file_id": record.id,
        "original_filename": record.original_filename,
        "mime_type": record.mime_type,
        "file_size_kb": round(record.file_size_bytes / 1024, 2),
        "duration_seconds": record.duration_seconds,
        "sample_rate": record.sample_rate,
        "channels": record.channels,
        "is_processed": record.is_processed,
        "created_at": record.created_at.isoformat(),
    }


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Prediction history",
    description="Returns paginated prediction history with optional emotion and source filters.",
)
async def get_history(
    db: DBSession,
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    offset: int = Query(default=0, ge=0, description="Items to skip"),
    emotion: Optional[str] = Query(default=None, description="Filter by emotion"),
    source: Optional[str] = Query(
        default=None, description="Filter by source: upload | live | stream"
    ),
):
    """Return paginated prediction history."""
    records, total = await get_prediction_history(
        db, limit=limit, offset=offset,
        emotion_filter=emotion, source_filter=source
    )

    predictions = [
        PredictionSummary(
            id=r.id,
            predicted_emotion=r.predicted_emotion,
            confidence=round(r.confidence, 4),
            model_name=r.model_name,
            source=r.source,
            audio_duration_seconds=r.audio_duration_seconds,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]

    return HistoryResponse(
        predictions=predictions,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + limit) < total,
    )
