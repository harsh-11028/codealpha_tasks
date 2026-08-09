"""
api/dependencies.py — FastAPI dependency injection providers.

Provides reusable dependencies for:
  - Database sessions
  - Settings
  - Prediction engine (singleton)
  - Rate limiting state
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Database session dependency ──────────────────────────────────────────────
DBSession = Annotated[AsyncSession, Depends(get_db)]


# ── Prediction engine singleton ──────────────────────────────────────────────
# Loaded once at startup via lifespan, accessed via this dependency.
_prediction_engine = None


def set_prediction_engine(engine) -> None:
    """Called at app startup to inject the loaded model."""
    global _prediction_engine
    _prediction_engine = engine
    logger.info("PredictionEngine injected into dependency provider.")


def get_prediction_engine():
    """
    FastAPI dependency: returns the global PredictionEngine.
    Raises RuntimeError if not loaded yet (startup failure).
    """
    if _prediction_engine is None:
        raise RuntimeError(
            "PredictionEngine is not loaded. The model may still be initializing."
        )
    return _prediction_engine


PredictionEngineDep = Annotated[object, Depends(get_prediction_engine)]
