"""
database/database.py — Async SQLAlchemy engine, session factory, and base model.
Supports SQLite (dev) and PostgreSQL (prod) via DATABASE_URL env var.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Build async database URL ─────────────────────────────────────────────────
_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./speech_emotion.db")

# SQLAlchemy async drivers use different prefixes
if _DATABASE_URL.startswith("sqlite:///"):
    _ASYNC_DATABASE_URL = _DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
elif _DATABASE_URL.startswith("postgresql://"):
    _ASYNC_DATABASE_URL = _DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _DATABASE_URL.startswith("postgresql+asyncpg://"):
    _ASYNC_DATABASE_URL = _DATABASE_URL
else:
    _ASYNC_DATABASE_URL = _DATABASE_URL

logger.info(f"Database driver: {_ASYNC_DATABASE_URL.split('://')[0]}")

# ── Engine configuration ─────────────────────────────────────────────────────
_is_sqlite = "sqlite" in _ASYNC_DATABASE_URL

engine = create_async_engine(
    _ASYNC_DATABASE_URL,
    echo=os.getenv("APP_ENV", "development") == "development",
    # SQLite-specific: disable connection pool (single-file DB)
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    poolclass=StaticPool if _is_sqlite else None,
)

# ── Session factory ──────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── Declarative base ─────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── DB lifecycle helpers ─────────────────────────────────────────────────────
async def create_all_tables() -> None:
    """Create all tables defined via ORM models (dev/test convenience)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified.")


async def drop_all_tables() -> None:
    """Drop all tables (test teardown only)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: yields an async DB session.
    Rolls back on exception; always closes session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
