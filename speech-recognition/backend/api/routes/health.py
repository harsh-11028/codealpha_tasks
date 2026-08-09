"""
api/routes/health.py — Health check and readiness endpoints.

GET /health  — Liveness probe (always returns 200 if process is alive)
GET /ready   — Readiness probe (checks DB and model are available)
"""

import os
import time
from datetime import datetime, timezone

import torch
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import DBSession
from database.crud import get_active_model
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

# Track server start time for uptime reporting
_START_TIME = time.time()


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 if the server is running. Used by load balancers and Docker health checks.",
)
async def health_check():
    """Lightweight liveness probe — never checks external dependencies."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - _START_TIME, 2),
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "environment": os.getenv("APP_ENV", "development"),
    }


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Checks database connectivity and model availability. Returns 503 if not ready.",
)
async def readiness_check(
    db: DBSession,
    request: Request,
):
    """
    Deep readiness probe:
    - Checks DB connectivity with a simple SELECT 1
    - Checks that at least one model is registered
    """
    checks: dict[str, dict] = {}
    all_ok = True

    # ── Database check ────────────────────────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok", "message": "Connected"}
    except Exception as e:
        checks["database"] = {"status": "error", "message": str(e)}
        all_ok = False
        logger.error(f"Readiness: DB check failed — {e}")

    # ── Model check ──────────────────────────────────────────────────────
    try:
        active = await get_active_model(db)
        if active:
            checks["model"] = {
                "status": "ok",
                "active_model": active.name,
                "architecture": active.architecture,
            }
        else:
            checks["model"] = {
                "status": "warning",
                "message": "No active model. Upload or train a model first.",
            }
            # Don't mark not ready — app can still accept uploads
    except Exception as e:
        checks["model"] = {"status": "error", "message": str(e)}
        all_ok = False
        logger.error(f"Readiness: Model check failed — {e}")

    # ── System check ──────────────────────────────────────────────────────
    checks["system"] = {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }

    http_status = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=http_status,
        content={
            "status": "ready" if all_ok else "not_ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        },
    )
