"""
FastAPI application entry point.

Registers all routers, middleware, CORS, lifespan events,
and serves Swagger UI at /docs.

Start with:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import get_settings
from backend.app.database import create_tables
from backend.app.middleware.security import global_exception_handler
from backend.app.routers import health, history, predict, upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown events
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables and warm up OCR pipeline."""
    logger.info("🚀 Starting %s v%s", settings.app_name, settings.app_version)

    # Create SQLite tables
    create_tables()
    logger.info("✅ Database tables created/verified.")

    # Warm up OCR pipeline in background
    try:
        from backend.app.services.ocr_service import get_pipeline
        get_pipeline()
        logger.info("✅ OCR pipeline initialized.")
    except Exception as exc:
        logger.warning("⚠️  OCR pipeline init deferred: %s", exc)

    yield

    logger.info("👋 Shutting down %s.", settings.app_name)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## AI Handwritten Character Recognition & OCR System

A production-ready OCR API capable of recognizing:
- **Digits** (0–9)
- **English alphabets** (A–Z, a–z)
- **Handwritten words** and **sentences**

### Key Endpoints
| Method | Path                   | Description               |
|--------|------------------------|---------------------------|
| POST   | /api/predict-character | Recognize single character |
| POST   | /api/predict-word      | Recognize a word          |
| POST   | /api/predict-sentence  | Full document OCR         |
| POST   | /api/upload            | Upload an image           |
| POST   | /api/webcam            | Submit webcam frame       |
| GET    | /api/history           | Prediction history        |
| POST   | /api/export            | Export to TXT/PDF/DOCX    |
| GET    | /api/health            | Health check              |
| GET    | /api/model-info        | Model info and metrics    |
""",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# Request timing header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response

# Gzip compression for large responses (base64 images, history)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS (Must be added last in Starlette/FastAPI so it wraps all inner middleware and exception handlers as the outermost layer)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
app.add_exception_handler(Exception, global_exception_handler)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(upload.router)
app.include_router(history.router)

# ---------------------------------------------------------------------------
# Static files (uploaded images)
# ---------------------------------------------------------------------------

import pathlib
pathlib.Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"])
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
    }
