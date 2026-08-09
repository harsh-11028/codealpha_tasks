"""
database/crud.py — Full CRUD operations for all ORM models.
All functions are async and accept an AsyncSession.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import ModelRegistry, PredictionHistory, UploadedFile
from utils.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  UploadedFile CRUD
# ════════════════════════════════════════════════════════════════════════════

async def create_uploaded_file(
    db: AsyncSession,
    *,
    filename: str,
    original_filename: str,
    mime_type: str,
    file_size_bytes: int,
    storage_path: str,
    duration_seconds: Optional[float] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    file_hash: Optional[str] = None,
) -> UploadedFile:
    """Create and persist a new UploadedFile record."""
    record = UploadedFile(
        id=str(uuid4()),
        filename=filename,
        original_filename=original_filename,
        mime_type=mime_type,
        file_size_bytes=file_size_bytes,
        storage_path=storage_path,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        file_hash=file_hash,
    )
    db.add(record)
    await db.flush()
    logger.info(f"Created UploadedFile: {record.id} ({original_filename})")
    return record


async def get_uploaded_file(
    db: AsyncSession, file_id: str
) -> Optional[UploadedFile]:
    """Fetch a single uploaded file by ID."""
    result = await db.execute(
        select(UploadedFile).where(UploadedFile.id == file_id)
    )
    return result.scalar_one_or_none()


async def get_file_by_hash(
    db: AsyncSession, file_hash: str
) -> Optional[UploadedFile]:
    """Find duplicate uploads by content hash."""
    result = await db.execute(
        select(UploadedFile).where(UploadedFile.file_hash == file_hash)
    )
    return result.scalar_one_or_none()


async def mark_file_processed(db: AsyncSession, file_id: str) -> None:
    """Mark a file as processed after feature extraction."""
    await db.execute(
        update(UploadedFile)
        .where(UploadedFile.id == file_id)
        .values(is_processed=True)
    )


# ════════════════════════════════════════════════════════════════════════════
#  PredictionHistory CRUD
# ════════════════════════════════════════════════════════════════════════════

async def create_prediction(
    db: AsyncSession,
    *,
    predicted_emotion: str,
    confidence: float,
    all_probabilities: dict,
    model_name: str,
    source: str = "upload",
    file_id: Optional[str] = None,
    request_id: Optional[str] = None,
    model_version: Optional[str] = None,
    audio_duration_seconds: Optional[float] = None,
    sample_rate: Optional[int] = None,
    mfcc_data: Optional[dict] = None,
    spectrogram_data: Optional[dict] = None,
    waveform_data: Optional[dict] = None,
    explanation: Optional[str] = None,
    feature_importance: Optional[dict] = None,
    inference_time_ms: Optional[float] = None,
    preprocessing_time_ms: Optional[float] = None,
) -> PredictionHistory:
    """Create and persist a new prediction result."""
    record = PredictionHistory(
        id=str(uuid4()),
        file_id=file_id,
        source=source,
        request_id=request_id,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        all_probabilities=all_probabilities,
        model_name=model_name,
        model_version=model_version,
        audio_duration_seconds=audio_duration_seconds,
        sample_rate=sample_rate,
        mfcc_data=mfcc_data,
        spectrogram_data=spectrogram_data,
        waveform_data=waveform_data,
        explanation=explanation,
        feature_importance=feature_importance,
        inference_time_ms=inference_time_ms,
        preprocessing_time_ms=preprocessing_time_ms,
    )
    db.add(record)
    await db.flush()
    logger.info(
        f"Saved prediction: {record.id} — {predicted_emotion} ({confidence:.2%})"
    )
    return record


async def get_prediction(
    db: AsyncSession, prediction_id: str
) -> Optional[PredictionHistory]:
    """Fetch a single prediction by ID."""
    result = await db.execute(
        select(PredictionHistory).where(PredictionHistory.id == prediction_id)
    )
    return result.scalar_one_or_none()


async def get_prediction_history(
    db: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    emotion_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> tuple[list[PredictionHistory], int]:
    """
    Fetch paginated prediction history with optional filters.

    Returns:
        (records, total_count)
    """
    query = select(PredictionHistory)
    count_query = select(func.count()).select_from(PredictionHistory)

    if emotion_filter:
        query = query.where(PredictionHistory.predicted_emotion == emotion_filter)
        count_query = count_query.where(
            PredictionHistory.predicted_emotion == emotion_filter
        )
    if source_filter:
        query = query.where(PredictionHistory.source == source_filter)
        count_query = count_query.where(PredictionHistory.source == source_filter)

    total = (await db.execute(count_query)).scalar_one()
    records = (
        await db.execute(
            query.order_by(desc(PredictionHistory.created_at))
            .limit(limit)
            .offset(offset)
        )
    ).scalars().all()

    return list(records), total


async def get_emotion_statistics(db: AsyncSession) -> dict:
    """
    Aggregate emotion counts across all predictions.
    Returns: {"happy": 42, "sad": 18, ...}
    """
    result = await db.execute(
        select(
            PredictionHistory.predicted_emotion,
            func.count(PredictionHistory.id).label("count"),
        ).group_by(PredictionHistory.predicted_emotion)
    )
    rows = result.all()
    return {row.predicted_emotion: row.count for row in rows}


async def get_average_confidence(db: AsyncSession) -> Optional[float]:
    """Compute average confidence across all predictions."""
    result = await db.execute(
        select(func.avg(PredictionHistory.confidence))
    )
    return result.scalar_one_or_none()


# ════════════════════════════════════════════════════════════════════════════
#  ModelRegistry CRUD
# ════════════════════════════════════════════════════════════════════════════

async def register_model(
    db: AsyncSession,
    *,
    name: str,
    version: str,
    architecture: str,
    model_path: str,
    metrics: Optional[dict] = None,
    datasets_used: Optional[dict] = None,
    num_parameters: Optional[int] = None,
    epochs_trained: Optional[int] = None,
    training_duration_seconds: Optional[float] = None,
    config_path: Optional[str] = None,
    num_classes: int = 8,
) -> ModelRegistry:
    """Register a newly trained model in the registry."""
    record = ModelRegistry(
        id=str(uuid4()),
        name=name,
        version=version,
        architecture=architecture,
        model_path=model_path,
        config_path=config_path,
        metrics=metrics,
        datasets_used=datasets_used,
        num_parameters=num_parameters,
        epochs_trained=epochs_trained,
        training_duration_seconds=training_duration_seconds,
        num_classes=num_classes,
        is_active=False,
        is_best=False,
    )
    db.add(record)
    await db.flush()
    logger.info(f"Registered model: {name} v{version} ({architecture})")
    return record


async def set_active_model(db: AsyncSession, model_id: str) -> None:
    """Deactivate all models then activate the specified one."""
    # Deactivate all
    await db.execute(
        update(ModelRegistry).values(is_active=False)
    )
    # Activate specified
    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.id == model_id)
        .values(is_active=True)
    )
    logger.info(f"Set active model: {model_id}")


async def set_best_model(db: AsyncSession, model_id: str) -> None:
    """Mark a model as the best (by validation F1)."""
    await db.execute(update(ModelRegistry).values(is_best=False))
    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.id == model_id)
        .values(is_best=True, is_active=True)
    )
    logger.info(f"Set best model: {model_id}")


async def get_active_model(db: AsyncSession) -> Optional[ModelRegistry]:
    """Fetch the currently active model."""
    result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.is_active == True)
    )
    return result.scalar_one_or_none()


async def get_best_model(db: AsyncSession) -> Optional[ModelRegistry]:
    """Fetch the best-performing model."""
    result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.is_best == True)
    )
    return result.scalar_one_or_none()


async def list_models(db: AsyncSession) -> list[ModelRegistry]:
    """List all registered models ordered by creation date desc."""
    result = await db.execute(
        select(ModelRegistry).order_by(desc(ModelRegistry.created_at))
    )
    return list(result.scalars().all())


async def update_model_metrics(
    db: AsyncSession, model_id: str, metrics: dict
) -> None:
    """Update evaluation metrics for a model."""
    await db.execute(
        update(ModelRegistry)
        .where(ModelRegistry.id == model_id)
        .values(metrics=metrics, updated_at=datetime.now(timezone.utc))
    )
