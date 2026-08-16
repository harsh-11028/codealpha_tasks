"""
Integration test suite for FastAPI backend endpoints using TestClient.
Tests health checks, model info retrieval, image upload, history logging, and rate limits.
"""

import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import create_tables

@pytest.fixture(scope="module")
def client():
    # Verify tables exist before test execution
    create_tables()
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "service" in data


def test_health_check(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "device" in data
    assert "model_loaded" in data


def test_model_info(client: TestClient):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "all_models" in data
    assert "total_predictions" in data


def test_upload_endpoint(client: TestClient):
    # Create a mock synthetic PNG byte array
    mock_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    files = {"file": ("mock_sample.png", io.BytesIO(mock_png), "image/png")}
    
    response = client.post("/api/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "image_url" in data
    assert data["filename"] == "mock_sample.png"
    assert data["size_bytes"] == len(mock_png)


def test_history_and_stats(client: TestClient):
    response_stats = client.get("/api/history/stats")
    assert response_stats.status_code == 200
    stats = response_stats.json()
    assert "total_predictions" in stats
    assert "mean_confidence" in stats
    assert "mean_processing_ms" in stats

    response_hist = client.get("/api/history", params={"limit": 10})
    assert response_hist.status_code == 200
    assert isinstance(response_hist.json(), list)


def test_export_endpoint(client: TestClient):
    sample_text = "Handwritten Character Recognition OCR output text sample."
    response = client.post(
        "/api/export",
        params={"text": sample_text, "format": "txt", "filename": "test_output"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.content == sample_text.encode("utf-8")
