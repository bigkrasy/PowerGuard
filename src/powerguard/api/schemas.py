"""
Pydantic schemas за REST API.

Дефинира валидация за:
- Входни данни (сензорни показания)
- Изходни данни (предсказания)
- Metadata (информация за модела)
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ═══════════════════════════════════════════════════════
# ВХОДНИ SCHEMAS
# ═══════════════════════════════════════════════════════

class SensorReading(BaseModel):
    """Едно сензорно показание от трансформатор.

    Всички стойности са задължителни и валидирани.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "U": 11000.0,
                "I": 100.0,
                "R": 110.0,
                "P": 1012000.0,
                "Q": 430000.0,
                "T": 25.0,
                "f": 50.0,
                "cos_phi": 0.92,
            }
        }
    )

    U: float = Field(..., gt=0, description="Напрежение (V)")
    I: float = Field(..., ge=0, description="Ток (A)")
    R: float = Field(..., gt=0, description="Съпротивление (Ohm)")
    P: float = Field(..., description="Активна мощност (W)")
    Q: float = Field(..., description="Реактивна мощност (VAR)")
    T: float = Field(..., description="Температура (C)")
    f: float = Field(..., gt=0, description="Честота (Hz)")
    cos_phi: float = Field(..., ge=0, le=1, description="Фактор на мощността")


class BatchSensorReadings(BaseModel):
    """Batch от сензорни показания."""
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Списък със сензорни показания (1-1000)",
    )


# ═══════════════════════════════════════════════════════
# ИЗХОДНИ SCHEMAS
# ═══════════════════════════════════════════════════════

class PredictionResult(BaseModel):
    """Резултат от предсказание за един sample."""
    is_anomaly: bool = Field(..., description="Има ли аномалия")
    anomaly_probability: float = Field(
        ..., ge=0, le=1, description="Вероятност за аномалия [0, 1]"
    )
    anomaly_type: str = Field(
        ...,
        description="Тип на аномалията (или 'normal')",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="Ниво на сигурност на предсказанието"
    )


class BatchPredictionResult(BaseModel):
    """Резултат от batch предсказание."""
    predictions: list[PredictionResult]
    n_anomalies: int
    anomaly_ratio: float


# ═══════════════════════════════════════════════════════
# METADATA SCHEMAS
# ═══════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Отговор от /health endpoint."""
    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    uptime_seconds: float
    version: str


class ModelInfoResponse(BaseModel):
    """Информация за заредения модел."""
    model_type: str
    n_features: int
    feature_names: list[str]
    training_date: str
    auc_pr: float | None = None
    notes: str | None = None