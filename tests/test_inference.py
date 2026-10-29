"""Тестове за inference модула (ModelStore, predict_single)."""
import pytest
from powerguard.api.inference import ModelStore, predict_single
from powerguard.api.schemas import SensorReading


def test_model_store_loads():
    """ModelStore се зарежда успешно."""
    ModelStore.load()
    assert ModelStore.is_loaded()
    assert ModelStore.model is not None
    assert ModelStore.scaler is not None
    assert ModelStore.model_type == "XGBClassifier"


def test_model_store_clear():
    """ModelStore.clear() изчиства всичко."""
    ModelStore.load()
    assert ModelStore.is_loaded()

    ModelStore.clear()
    assert not ModelStore.is_loaded()
    assert ModelStore.model is None


def test_predict_single_normal():
    """Predict връща нормален резултат за нормални данни."""
    ModelStore.load()

    reading = SensorReading(
        U=11000, I=100, R=110,
        P=1012000, Q=430000, T=25, f=50, cos_phi=0.92,
    )
    result = predict_single(reading)

    assert result.is_anomaly is False
    assert result.anomaly_type == "normal"
    assert 0 <= result.anomaly_probability <= 1


def test_predict_single_anomaly():
    """Predict връща аномалия за overheat."""
    ModelStore.load()

    reading = SensorReading(
        U=11000, I=160, R=110,
        P=1600000, Q=700000, T=90, f=50, cos_phi=0.92,
    )
    result = predict_single(reading)

    assert result.is_anomaly is True
    assert result.anomaly_type == "overheat"


def test_predict_without_model_raises():
    """Predict без зареден модел хвърля RuntimeError."""
    ModelStore.clear()

    reading = SensorReading(
        U=11000, I=100, R=110,
        P=1012000, Q=430000, T=25, f=50, cos_phi=0.92,
    )

    with pytest.raises(RuntimeError, match="Моделът не е зареден"):
        predict_single(reading)