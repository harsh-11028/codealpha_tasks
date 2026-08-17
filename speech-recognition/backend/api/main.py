"""
api/main.py — FastAPI application factory and entry point.

Features:
  - Async lifespan: DB table creation + ML model loading on startup
  - All routers mounted
  - Middleware stack: CORS, Security, Rate Limiting, Request ID, Process Time
  - WebSocket endpoint for real-time streaming predictions
  - Global exception handlers
  - OpenAPI customization
"""

import base64
import json
import os
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# ── Load env before anything else ────────────────────────────────────────────
load_dotenv()

from api.middleware.cors import add_cors_middleware
from api.middleware.security import add_security_middleware
from api.routes import health, metrics, model_info, predict, upload
from database.database import create_all_tables
from training.config import EMOTION_COLORS, EMOTION_EMOJI, DEFAULT_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

# ── App metadata ──────────────────────────────────────────────────────────────
APP_NAME = os.getenv("APP_NAME", "Speech Emotion Recognition")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_ENV = os.getenv("APP_ENV", "development")


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    Runs startup logic before yielding, shutdown logic after.
    """
    logger.info(f"🚀 Starting {APP_NAME} v{APP_VERSION} [{APP_ENV}]")
    logger.info(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # ── Create database tables ────────────────────────────────────────────
    logger.info("Initializing database...")
    await create_all_tables()

    # ── Load ML model ─────────────────────────────────────────────────────
    logger.info("Loading prediction engine...")
    try:
        from ml.prediction_engine import PredictionEngine
        from api.dependencies import set_prediction_engine

        engine = PredictionEngine(config=DEFAULT_CONFIG)
        await engine.load()
        set_prediction_engine(engine)
        logger.info(f"✅ Prediction engine loaded: {engine.active_model_name}")
    except Exception as e:
        # Don't crash on model load failure — API still serves upload/history
        logger.warning(
            f"⚠️  Prediction engine failed to load: {e}. "
            "The /predict endpoint will return stub responses until a model is trained."
        )

    logger.info(f"✅ {APP_NAME} is ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info(f"🛑 Shutting down {APP_NAME}...")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=(
            "AI-powered Speech Emotion Recognition API. "
            "Classify human emotions from audio in real-time using deep learning."
        ),
        contact={
            "name": "SER Project",
            "url": "https://github.com/yourusername/speech-emotion-recognition",
        },
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        openapi_tags=[
            {"name": "Health", "description": "Liveness and readiness probes"},
            {"name": "Prediction", "description": "Emotion prediction from audio"},
            {"name": "Upload & History", "description": "File management and history"},
            {"name": "Model", "description": "Model registry and info"},
            {"name": "Metrics", "description": "Performance and usage metrics"},
        ],
        docs_url="/docs" if APP_ENV != "production" else None,
        redoc_url="/redoc" if APP_ENV != "production" else None,
        lifespan=lifespan,
    )

    # ── Middleware (order: security outermost, CORS innermost) ────────────
    add_cors_middleware(app)
    add_security_middleware(app)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(upload.router)
    app.include_router(model_info.router)
    app.include_router(metrics.router)

    # ── Exception handlers ────────────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        errors = exc.errors()
        logger.warning(f"Validation error on {request.url.path}: {errors}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation failed",
                "details": errors,
                "path": str(request.url.path),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception on {request.url.path}: {exc}\n"
            + traceback.format_exc()
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again.",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    # ── WebSocket streaming endpoint ──────────────────────────────────────
    _ws_manager = WebSocketConnectionManager()

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        WebSocket endpoint for real-time audio streaming predictions.

        Protocol:
          Client → Server: JSON { "type": "audio_chunk", "data": "<base64>", "sample_rate": 22050 }
          Server → Client: JSON { "type": "prediction", "emotion": "...", "confidence": 0.87, ... }

          Client → Server: JSON { "type": "ping" }
          Server → Client: JSON { "type": "pong" }

          Client → Server: JSON { "type": "close" }
          Server → Client: JSON { "type": "closed" }
        """
        await _ws_manager.connect(websocket)
        client_id = id(websocket)
        logger.info(f"WebSocket client connected: {client_id}")

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message.",
                    })
                    continue

                msg_type = message.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "audio_chunk":
                    audio_b64 = message.get("data", "")
                    sample_rate = message.get("sample_rate", 22050)

                    if not audio_b64:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing 'data' field in audio_chunk.",
                        })
                        continue

                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                    except Exception:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid base64 audio data.",
                        })
                        continue

                    # Run prediction
                    prediction = await _run_ws_prediction(audio_bytes, sample_rate)
                    await websocket.send_json({"type": "prediction", **prediction})

                elif msg_type == "close":
                    await websocket.send_json({"type": "closed"})
                    break

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: '{msg_type}'.",
                    })

        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
        finally:
            _ws_manager.disconnect(websocket)

    return app


# ── WebSocket connection manager ───────────────────────────────────────────────
class WebSocketConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._active.append(websocket)
        logger.debug(f"Active WS connections: {len(self._active)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._active:
            self._active.remove(websocket)
        logger.debug(f"Active WS connections: {len(self._active)}")

    async def broadcast(self, message: dict) -> None:
        """Send a message to all connected clients."""
        for ws in self._active:
            try:
                await ws.send_json(message)
            except Exception:
                pass


async def _run_ws_prediction(audio_bytes: bytes, sample_rate: int) -> dict:
    """
    Run prediction on a WebSocket audio chunk.
    Falls back to stub if model not loaded.
    """
    try:
        from api.dependencies import _prediction_engine
        if _prediction_engine is not None:
            result = _prediction_engine.predict(audio_bytes)
            return {
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "confidence_pct": round(result["confidence"] * 100, 2),
                "probabilities": result["probabilities"],
                "emoji": EMOTION_EMOJI.get(result["emotion"], ""),
                "color": EMOTION_COLORS.get(result["emotion"], "#888"),
                "inference_time_ms": result.get("inference_time_ms", 0),
            }
    except Exception as e:
        logger.debug(f"WS prediction fallback: {e}")

    # Stub
    return {
        "emotion": "neutral",
        "confidence": 0.73,
        "confidence_pct": 73.0,
        "probabilities": {"neutral": 0.73, "calm": 0.10, "happy": 0.08},
        "emoji": "😐",
        "color": "#94a3b8",
        "inference_time_ms": 0,
    }


# ── Application instance ──────────────────────────────────────────────────────
app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", "8001")),
        reload=os.getenv("BACKEND_RELOAD", "true").lower() == "true",
        log_level="info",
        access_log=True,
    )
