"""
api/middleware/cors.py — CORS configuration for FastAPI.
Reads allowed origins from environment for security.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.logger import get_logger

logger = get_logger(__name__)


def add_cors_middleware(app: FastAPI) -> None:
    """
    Attach CORS middleware to the FastAPI app.
    Allowed origins are read from the ALLOWED_ORIGINS env variable
    (comma-separated list of URLs).
    """
    raw_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5174,http://localhost:3000,http://localhost:80",
    )
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    app_env = os.getenv("APP_ENV", "development")
    if app_env == "development":
        # In dev, allow all localhost variants
        allowed_origins = ["*"]
        logger.warning(
            "CORS: Running in development mode — all origins allowed. "
            "Set APP_ENV=production and ALLOWED_ORIGINS in production."
        )
    else:
        logger.info(f"CORS: Allowed origins → {allowed_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Request-ID",
            "X-API-Key",
        ],
        expose_headers=["X-Request-ID", "X-Process-Time"],
        max_age=600,  # 10 minutes preflight cache
    )
