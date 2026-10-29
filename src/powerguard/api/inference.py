"""
Inference logic за PowerGuard.

Съдържа ModelStore (singleton) и helper функции за prediction.
Използва се от rest_api.py и тестовете.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

from powerguard.config import MODELS_DIR
from powerguard.api.schemas import SensorReading, PredictionResult
from powerguard.data.features import build_features, get_feature_columns


logger = logging.getLogger("powerguard.inference")


MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
SCALER_PATH = MODELS_DIR / "xgb_scaler.joblib"


class ModelStore:
    """Singleton — зарежда модела веднъж при startup."""

    model = None
    scaler = None
    feature_names: list[str] = []
    model_type: str = "unknown"

    @classmethod
    def load(cls, model_path: Path | None = None, scaler_path: Path | None = None):
        """Зарежда модела и scaler-а от disk."""
        model_path = model_path or MODEL_PATH
        scaler_path = scaler_path or SCALER_PATH

        if not model_path.exists():
            logger.warning(f"Липсва модел: {model_path}")
            return

        if not scaler_path.exists():
            logger.warning(f"Липсва scaler: {scaler_path}")
            return

        cls.model = joblib.load(model_path)
        cls.scaler = joblib.load(scaler_path)
        cls.model_type = type(cls.model).__name__

        if hasattr(cls.scaler, "feature_names_in_"):
            cls.feature_names = list(cls.scaler.feature_names_in_)

        logger.info(f"Model loaded: {cls.model_type}, {len(cls.feature_names)} features")

    @classmethod
    def is_loaded(cls) -> bool:
        return cls.model is not None and cls.scaler is not None

    @classmethod
    def clear(cls):
        cls.model = None
        cls.scaler = None
        cls.feature_names = []
        cls.model_type = "unknown"


def _infer_anomaly_type(row: pd.Series) -> str:
    """Евристика за тип аномалия."""
    if row.get("T", 0) > 60:
        return "overheat"

    if row.get("U", 0) < 5000 and abs(row.get("ohm_residual", 0)) > 1000:
        return "short_circuit"

    if row.get("I", 0) < 30:
        return "load_loss"

    if abs(row.get("power_residual", 0)) > 1000:
        return "sensor_fault"

    return "unknown"


def predict_single(reading: SensorReading) -> PredictionResult:
    """Прави предсказание за един sample."""
    if not ModelStore.is_loaded():
        raise RuntimeError("Моделът не е зареден")

    # 1. Input → DataFrame
    df = pd.DataFrame([reading.model_dump()])

    # 2. Feature engineering
    features = build_features(df)

    # 3. Feature columns
    feature_cols = get_feature_columns(features, exclude_leakage=True)
    X = features[feature_cols]

    # 4. Scale
    X_scaled = ModelStore.scaler.transform(X)

    # 5. Predict
    prob = float(ModelStore.model.predict_proba(X_scaled)[0, 1])
    is_anomaly = prob > 0.5

    # 6. Anomaly type
    anomaly_type = "normal" if not is_anomaly else _infer_anomaly_type(features.iloc[0])

    # 7. Confidence
    if prob < 0.2 or prob > 0.8:
        confidence = "high"
    elif prob < 0.4 or prob > 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    return PredictionResult(
        is_anomaly=is_anomaly,
        anomaly_probability=prob,
        anomaly_type=anomaly_type,
        confidence=confidence,
    )