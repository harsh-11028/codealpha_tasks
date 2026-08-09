"""
tests/test_api.py — API integration tests using FastAPI's async test client.

Tests cover:
  - Health and readiness endpoints
  - File upload (valid + invalid)
  - Predict endpoint (valid + invalid)
  - History endpoint
  - Model info endpoint
  - Metrics endpoint
"""

import base64
import io
import os
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Use a test SQLite DB that gets cleaned up
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_app.db"
os.environ["APP_ENV"] = "test"

from api.main import app
from database.database import create_all_tables, drop_all_tables


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_database():
    """Create test DB tables before tests, drop them after."""
    await create_all_tables()
    yield
    await drop_all_tables()
    # Clean up test DB file
    db_file = Path("test_app.db")
    if db_file.exists():
        db_file.unlink()


@pytest_asyncio.fixture(scope="module")
async def client():
    """Async HTTP test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def _make_wav_bytes(
    duration_seconds: float = 1.0,
    sample_rate: int = 22050,
    channels: int = 1,
) -> bytes:
    """Generate a minimal valid WAV file with a sine wave."""
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


# ── Health Tests ──────────────────────────────────────────────────────────────
class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_has_request_id_header(self, client: AsyncClient):
        response = await client.get("/health")
        assert "x-request-id" in response.headers

    @pytest.mark.asyncio
    async def test_health_has_process_time_header(self, client: AsyncClient):
        response = await client.get("/health")
        assert "x-process-time" in response.headers

    @pytest.mark.asyncio
    async def test_readiness_returns_200_or_503(self, client: AsyncClient):
        response = await client.get("/ready")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "checks" in data
        assert "database" in data["checks"]


# ── Upload Tests ──────────────────────────────────────────────────────────────
class TestUploadEndpoint:
    @pytest.mark.asyncio
    async def test_upload_valid_wav(self, client: AsyncClient):
        wav_bytes = _make_wav_bytes(duration_seconds=2.0)
        response = await client.post(
            "/upload",
            files={"file": ("test_audio.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 201
        data = response.json()
        assert "file_id" in data
        assert data["original_filename"] == "test_audio.wav"
        assert data["is_duplicate"] is False
        assert data["file_size_kb"] > 0

    @pytest.mark.asyncio
    async def test_upload_duplicate_detected(self, client: AsyncClient):
        wav_bytes = _make_wav_bytes(duration_seconds=1.0)
        # Upload once
        r1 = await client.post(
            "/upload",
            files={"file": ("dup_test.wav", wav_bytes, "audio/wav")},
        )
        assert r1.status_code == 201

        # Upload same bytes again
        r2 = await client.post(
            "/upload",
            files={"file": ("dup_test.wav", wav_bytes, "audio/wav")},
        )
        assert r2.status_code == 201
        data = r2.json()
        assert data["is_duplicate"] is True

    @pytest.mark.asyncio
    async def test_upload_rejects_text_file(self, client: AsyncClient):
        response = await client.post(
            "/upload",
            files={"file": ("evil.wav", b"This is not audio", "audio/wav")},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_rejects_empty_file(self, client: AsyncClient):
        response = await client.post(
            "/upload",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_file_metadata(self, client: AsyncClient):
        wav_bytes = _make_wav_bytes()
        upload_r = await client.post(
            "/upload",
            files={"file": ("meta_test.wav", wav_bytes, "audio/wav")},
        )
        file_id = upload_r.json()["file_id"]

        response = await client.get(f"/upload/{file_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["file_id"] == file_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_file_returns_404(self, client: AsyncClient):
        response = await client.get("/upload/nonexistent-id")
        assert response.status_code == 404


# ── Prediction Tests ──────────────────────────────────────────────────────────
class TestPredictEndpoint:
    @pytest.mark.asyncio
    async def test_predict_returns_valid_response(self, client: AsyncClient):
        wav_bytes = _make_wav_bytes(duration_seconds=3.0)
        response = await client.post(
            "/predict",
            files={"file": ("emotion_test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction_id" in data
        assert "predicted_emotion" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "all_probabilities" in data
        assert len(data["all_probabilities"]) == 8  # 8 emotion classes

    @pytest.mark.asyncio
    async def test_predict_live_with_base64(self, client: AsyncClient):
        wav_bytes = _make_wav_bytes(duration_seconds=2.0)
        audio_b64 = base64.b64encode(wav_bytes).decode()
        response = await client.post(
            "/predict-live",
            json={"audio_base64": audio_b64, "sample_rate": 22050, "encoding": "wav"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "live"

    @pytest.mark.asyncio
    async def test_predict_rejects_invalid_file(self, client: AsyncClient):
        response = await client.post(
            "/predict",
            files={"file": ("fake.wav", b"not an audio file at all", "audio/wav")},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_invalid_base64_live(self, client: AsyncClient):
        response = await client.post(
            "/predict-live",
            json={"audio_base64": "!!! invalid base64 !!!", "sample_rate": 22050},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_prediction_by_id(self, client: AsyncClient):
        wav_bytes = _make_wav_bytes()
        pred_r = await client.post(
            "/predict",
            files={"file": ("get_test.wav", wav_bytes, "audio/wav")},
        )
        pred_id = pred_r.json()["prediction_id"]

        response = await client.get(f"/predict/{pred_id}")
        assert response.status_code == 200
        assert response.json()["prediction_id"] == pred_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_prediction_returns_404(self, client: AsyncClient):
        response = await client.get("/predict/does-not-exist")
        assert response.status_code == 404


# ── History Tests ─────────────────────────────────────────────────────────────
class TestHistoryEndpoint:
    @pytest.mark.asyncio
    async def test_history_returns_list(self, client: AsyncClient):
        response = await client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert isinstance(data["predictions"], list)

    @pytest.mark.asyncio
    async def test_history_pagination(self, client: AsyncClient):
        response = await client.get("/history?limit=5&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) <= 5


# ── Model Info Tests ──────────────────────────────────────────────────────────
class TestModelInfoEndpoint:
    @pytest.mark.asyncio
    async def test_model_info_returns_emotion_classes(self, client: AsyncClient):
        response = await client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "emotion_classes" in data
        assert "num_classes" in data
        assert data["num_classes"] == 8


# ── Metrics Tests ─────────────────────────────────────────────────────────────
class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_usage_stats(self, client: AsyncClient):
        response = await client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "usage_stats" in data
        assert "emotion_distribution" in data["usage_stats"]

    @pytest.mark.asyncio
    async def test_emotion_distribution_endpoint(self, client: AsyncClient):
        response = await client.get("/metrics/emotions")
        assert response.status_code == 200
        data = response.json()
        assert "distribution" in data
        assert "total_predictions" in data
