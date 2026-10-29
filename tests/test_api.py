"""
Тестове за REST API.

Използва FastAPI TestClient — не изисква реален сървър.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient с заредено API."""
    from powerguard.api.rest_api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def normal_reading():
    """Нормално сензорно показание."""
    return {
        "U": 11000.0,
        "I": 100.0,
        "R": 110.0,
        "P": 1012000.0,
        "Q": 430000.0,
        "T": 25.0,
        "f": 50.0,
        "cos_phi": 0.92,
    }


# ═══════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════

def test_health_endpoint(client):
    """GET /health връща 200 и правилен shape."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "uptime_seconds" in data
    assert data["version"] == "1.0.0"


# ═══════════════════════════════════════════════════════
# MODEL INFO
# ═══════════════════════════════════════════════════════

def test_model_info_returns_metadata(client):
    """GET /model_info връща metadata (ако моделът е зареден)."""
    response = client.get("/model_info")

    # Ако моделът не е зареден → 503
    if response.status_code == 503:
        pytest.skip("Моделът не е зареден — пусни scripts/train_model.py")

    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "n_features" in data
    assert data["n_features"] > 0
    assert isinstance(data["feature_names"], list)


# ═══════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════

def test_predict_normal(client, normal_reading):
    """POST /predict с нормални данни."""
    response = client.post("/predict", json=normal_reading)

    if response.status_code == 503:
        pytest.skip("Моделът не е зареден")

    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "anomaly_probability" in data
    assert 0 <= data["anomaly_probability"] <= 1
    assert data["confidence"] in ["low", "medium", "high"]


def test_predict_invalid_input_missing_field(client):
    """POST /predict без задължително поле → 422."""
    response = client.post("/predict", json={"U": 11000})
    assert response.status_code == 422


def test_predict_invalid_range(client, normal_reading):
    """POST /predict с отрицателно U → 422."""
    bad = {**normal_reading, "U": -100}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_batch(client, normal_reading):
    """POST /predict_batch с 3 samples."""
    payload = {"readings": [normal_reading] * 3}
    response = client.post("/predict_batch", json=payload)

    if response.status_code == 503:
        pytest.skip("Моделът не е зареден")

    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3
    assert "n_anomalies" in data
    assert "anomaly_ratio" in data


def test_predict_batch_empty(client):
    """POST /predict_batch с празен списък → 422."""
    response = client.post("/predict_batch", json={"readings": []})
    assert response.status_code == 422


def test_predict_batch_too_large(client, normal_reading):
    """POST /predict_batch с > 1000 samples → 422."""
    payload = {"readings": [normal_reading] * 1001}
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 422