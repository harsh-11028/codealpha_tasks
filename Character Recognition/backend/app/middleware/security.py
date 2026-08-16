"""
Security middleware — upload validation, rate limiting, CORS.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.app.config import get_settings

settings = get_settings()

# In-memory rate limiter (replace with Redis for multi-instance production)
_request_counts: Dict[str, list] = defaultdict(list)
_WINDOW_SECONDS = 60


def rate_limit_middleware(request: Request) -> None:
    """
    Simple sliding-window rate limiter per IP.
    Raises HTTP 429 if limit exceeded.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - _WINDOW_SECONDS

    # Prune old requests
    _request_counts[client_ip] = [
        t for t in _request_counts[client_ip] if t > window_start
    ]
    _request_counts[client_ip].append(now)

    if len(_request_counts[client_ip]) > settings.rate_limit_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {settings.rate_limit_requests} requests/minute.",
        )


def validate_image_upload(
    content_type: str,
    content_length: int,
    filename: str,
) -> None:
    """
    Validate an uploaded image file.

    Raises HTTP 400/413 on invalid inputs.
    """
    # Size check
    if content_length > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB.",
        )

    # Extension check
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. "
                   f"Allowed: {', '.join(sorted(settings.allowed_extensions_set))}",
        )

    # MIME type check
    allowed_mimes = {
        "image/jpeg", "image/png", "image/bmp",
        "image/tiff", "image/webp", "application/octet-stream",
    }
    if content_type and content_type.split(";")[0].strip() not in allowed_mimes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {content_type}",
        )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler for unexpected errors."""
    origin = request.headers.get("origin", "*")
    allow_origin = origin if (origin in settings.allowed_origins_list or "*" in settings.allowed_origins_list) else settings.allowed_origins_list[0] if settings.allowed_origins_list else "*"
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"{type(exc).__name__}: {str(exc)}", "type": type(exc).__name__},
        headers={
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Credentials": "true",
        },
    )
