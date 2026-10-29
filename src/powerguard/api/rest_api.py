"""
REST API за PowerGuard anomaly detection.

Endpoints:
- GET  /health         — проверка на статуса
- GET  /model_info     — информация за модела
- POST /predict        — single prediction
- POST /predict_batch  — batch prediction
- GET  /metrics        — Prometheus metrics
- GET  /docs           — Swagger UI
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from powerguard.api.schemas import (
    BatchPredictionResult,
    BatchSensorReadings,
    HealthResponse,
    ModelInfoResponse,
    PredictionResult,
    SensorReading,
)
from powerguard.api.inference import ModelStore, predict_single


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("powerguard.api")


_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PowerGuard API стартира...")
    ModelStore.load()
    yield
    logger.info("PowerGuard API спира...")
    ModelStore.clear()


app = FastAPI(
    title="PowerGuard API",
    description="Real-time anomaly detection за електрическа подстанция.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/docs", "/openapi.json", "/redoc"],
    inprogress_name="powerguard_requests_inprogress",
    inprogress_labels=True,
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok" if ModelStore.is_loaded() else "degraded",
        model_loaded=ModelStore.is_loaded(),
        uptime_seconds=time.time() - _start_time,
        version="1.0.0",
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    if not ModelStore.is_loaded():
        raise HTTPException(503, "Моделът не е зареден.")

    return ModelInfoResponse(
        model_type=ModelStore.model_type,
        n_features=len(ModelStore.feature_names),
        feature_names=ModelStore.feature_names,
        training_date="2024-01-01",
        auc_pr=None,
        notes="XGBoost classifier с физично-обосновани features",
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(reading: SensorReading):
    if not ModelStore.is_loaded():
        raise HTTPException(503, "Моделът не е зареден.")
    try:
        return predict_single(reading)
    except Exception as e:
        logger.exception("Грешка при предсказание")
        raise HTTPException(500, f"Грешка при предсказание: {e}")


@app.post("/predict_batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch(payload: BatchSensorReadings):
    if not ModelStore.is_loaded():
        raise HTTPException(503, "Моделът не е зареден.")
    try:
        predictions = [predict_single(r) for r in payload.readings]
        n_anomalies = sum(p.is_anomaly for p in predictions)
        return BatchPredictionResult(
            predictions=predictions,
            n_anomalies=n_anomalies,
            anomaly_ratio=n_anomalies / len(predictions),
        )
    except Exception as e:
        logger.exception("Грешка при batch предсказание")
        raise HTTPException(500, f"Грешка при batch предсказание: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("powerguard.api.rest_api:app", host="0.0.0.0", port=8000, reload=True)