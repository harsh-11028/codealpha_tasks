"""
api/middleware/security.py — Security middleware for the FastAPI app.

Implements:
  - Request ID injection (X-Request-ID header)
  - Process-time header (X-Process-Time)
  - Basic rate limiting per IP (in-memory token bucket)
  - Trusted host enforcement in production
"""

import os
import time
import uuid
from collections import defaultdict
from threading import Lock

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from utils.logger import get_logger

logger = get_logger(__name__)


# ── In-Memory Rate Limiter ────────────────────────────────────────────────────
class RateLimiter:
    """
    Token-bucket rate limiter keyed by client IP.

    Defaults: 60 requests / minute per IP.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._store: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        """
        Check if a request from `client_ip` is within the rate limit.

        Returns:
            (allowed, remaining_requests)
        """
        now = time.time()
        cutoff = now - self._window

        with self._lock:
            timestamps = self._store[client_ip]
            # Remove expired timestamps
            self._store[client_ip] = [t for t in timestamps if t > cutoff]
            count = len(self._store[client_ip])

            if count >= self._max:
                return False, 0

            self._store[client_ip].append(now)
            return True, self._max - count - 1


# Global rate limiter instance
_rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_MAX", "60")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
)


# ── Process Time + Request ID Middleware ──────────────────────────────────────
class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Attaches:
      - X-Request-ID: unique UUID per request (echoed back in response)
      - X-Process-Time: total server-side latency in milliseconds
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Assign or echo request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
        return response


# ── Rate Limiting Middleware ──────────────────────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Enforces per-IP rate limits.
    Health-check and docs endpoints are exempt.
    """

    _EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        allowed, remaining = _rate_limiter.is_allowed(client_ip)

        if not allowed:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please wait before retrying.",
                    "retry_after": os.getenv("RATE_LIMIT_WINDOW", "60"),
                },
                headers={"Retry-After": str(os.getenv("RATE_LIMIT_WINDOW", "60"))},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract real client IP, respecting X-Forwarded-For behind proxy."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


# ── Security Headers Middleware ────────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects recommended security headers on every response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response


# ── Factory ───────────────────────────────────────────────────────────────────
def add_security_middleware(app: FastAPI) -> None:
    """Attach all security middleware to the FastAPI app."""

    # Order matters: outermost middleware runs first on request / last on response.
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestContextMiddleware)

    app_env = os.getenv("APP_ENV", "development")
    if app_env == "production":
        allowed_hosts_raw = os.getenv("ALLOWED_HOSTS", "")
        if allowed_hosts_raw:
            allowed_hosts = [h.strip() for h in allowed_hosts_raw.split(",")]
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
            logger.info(f"TrustedHost middleware: {allowed_hosts}")

    logger.info("Security middleware attached.")
