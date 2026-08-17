"""
Tests for the FastAPI backend of the Disease Prediction System.
Run with: pytest tests/backend/ -v
"""
import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from app.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def client():
    """Create an async test client."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(scope="session")
async def auth_token(client):
    """Register and login a test user, return the token."""
    # Register
    register_resp = await client.post("/api/auth/register", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "testpassword123"
    })
    # May fail if user already exists (409/400), that's ok
    
    # Login
    login_resp = await client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword123"
    })
    assert login_resp.status_code == 200
    data = login_resp.json()
    assert data["success"] is True
    return data["data"]["access_token"]


class TestHealth:
    async def test_health_check(self, client):
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestAuth:
    async def test_register_new_user(self, client):
        """Test user registration."""
        response = await client.post("/api/auth/register", json={
            "name": "New Test User",
            "email": "newtest_unique@example.com",
            "password": "password123"
        })
        assert response.status_code in [201, 400]  # 400 if already exists
        if response.status_code == 201:
            data = response.json()
            assert data["success"] is True
            assert "access_token" in data["data"]

    async def test_register_duplicate_email(self, client, auth_token):
        """Test that duplicate email registration fails."""
        # First register
        await client.post("/api/auth/register", json={
            "name": "Dup Test",
            "email": "duptest@example.com",
            "password": "password123"
        })
        # Second attempt should fail
        response = await client.post("/api/auth/register", json={
            "name": "Dup Test 2",
            "email": "duptest@example.com",
            "password": "password456"
        })
        assert response.status_code == 400

    async def test_login_success(self, client):
        """Test successful login."""
        # Register first
        await client.post("/api/auth/register", json={
            "name": "Login Test",
            "email": "logintest@example.com",
            "password": "password123"
        })
        response = await client.post("/api/auth/login", json={
            "email": "logintest@example.com",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "access_token" in data["data"]

    async def test_login_wrong_password(self, client):
        """Test login with wrong password."""
        response = await client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

    async def test_get_me_with_token(self, client, auth_token):
        """Test getting current user profile."""
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "email" in data["data"]

    async def test_get_me_without_token(self, client):
        """Test that protected routes require auth."""
        response = await client.get("/api/auth/me")
        assert response.status_code == 401


class TestPredictions:
    async def test_heart_prediction_valid(self, client, auth_token):
        """Test valid heart disease prediction."""
        response = await client.post(
            "/api/predict/heart",
            json={
                "age": 55,
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 160,
                "exang": 0,
                "oldpeak": 1.5,
                "slope": 1,
                "ca": 0,
                "thal": 2
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["disease"] == "heart"
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["label"] in ["Positive", "Negative"]

    async def test_heart_prediction_invalid_field(self, client, auth_token):
        """Test heart prediction with invalid field value."""
        response = await client.post(
            "/api/predict/heart",
            json={
                "age": 200,  # Invalid: > 120
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 160,
                "exang": 0,
                "oldpeak": 1.5,
                "slope": 1,
                "ca": 0,
                "thal": 2
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 422

    async def test_diabetes_prediction_valid(self, client, auth_token):
        """Test valid diabetes prediction."""
        response = await client.post(
            "/api/predict/diabetes",
            json={
                "Pregnancies": 2,
                "Glucose": 120,
                "BloodPressure": 70,
                "SkinThickness": 20,
                "Insulin": 80,
                "BMI": 25.5,
                "DiabetesPedigreeFunction": 0.5,
                "Age": 35
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["disease"] == "diabetes"
        assert 0.0 <= data["probability"] <= 1.0

    async def test_breast_cancer_prediction_valid(self, client, auth_token):
        """Test valid breast cancer prediction."""
        response = await client.post(
            "/api/predict/breast-cancer",
            json={
                "mean_radius": 13.0,
                "mean_texture": 18.0,
                "mean_perimeter": 83.0,
                "mean_area": 525.0,
                "mean_smoothness": 0.1,
                "mean_compactness": 0.09,
                "mean_concavity": 0.05,
                "mean_concave_points": 0.03,
                "mean_symmetry": 0.17,
                "mean_fractal_dimension": 0.06,
                "radius_error": 0.28,
                "texture_error": 1.1,
                "perimeter_error": 1.9,
                "area_error": 22.0,
                "smoothness_error": 0.007,
                "compactness_error": 0.015,
                "concavity_error": 0.02,
                "concave_points_error": 0.009,
                "symmetry_error": 0.025,
                "fractal_dimension_error": 0.003,
                "worst_radius": 15.0,
                "worst_texture": 25.0,
                "worst_perimeter": 97.0,
                "worst_area": 680.0,
                "worst_smoothness": 0.13,
                "worst_compactness": 0.20,
                "worst_concavity": 0.20,
                "worst_concave_points": 0.09,
                "worst_symmetry": 0.27,
                "worst_fractal_dimension": 0.08
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["disease"] == "breast_cancer"

    async def test_prediction_without_auth(self, client):
        """Test that prediction requires authentication."""
        response = await client.post(
            "/api/predict/heart",
            json={"age": 50, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
                  "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
                  "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 1}
        )
        assert response.status_code == 401


class TestHistory:
    async def test_get_history(self, client, auth_token):
        """Test getting prediction history."""
        response = await client.get(
            "/api/predictions",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data["data"]
        assert "total" in data["data"]

    async def test_history_filter_by_disease(self, client, auth_token):
        """Test filtering history by disease."""
        response = await client.get(
            "/api/predictions?disease=heart",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200


class TestDashboard:
    async def test_dashboard_stats(self, client, auth_token):
        """Test dashboard statistics endpoint."""
        response = await client.get(
            "/api/dashboard/stats",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        stats = data["data"]
        assert "total_predictions" in stats
        assert "heart_predictions" in stats
        assert "diabetes_predictions" in stats
        assert "disease_distribution" in stats


class TestModels:
    async def test_get_model_performance(self, client, auth_token):
        """Test model performance endpoint."""
        for disease in ["heart", "diabetes", "breast_cancer"]:
            response = await client.get(
                f"/api/models/{disease}/performance",
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            assert response.status_code == 200

    async def test_invalid_disease_model(self, client, auth_token):
        """Test model performance with invalid disease."""
        response = await client.get(
            "/api/models/invalid_disease/performance",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404
