# ============================================================
# PowerGuard API - Production Dockerfile (multi-stage, non-root)
# ============================================================
# Multi-stage:
#   Stage 1 (builder) — компилира wheels с build-essential
#   Stage 2 (runtime) — инсталира само готовите wheels, без build tools
#
# xgboost се инсталира отделно с --no-deps + pin <3.0:
#   - --no-deps (при wheel) → xgboost 2.x не тегли nvidia-* при build
#   - <3.0 → избягва nvidia-nccl (xgboost 3.x го тегли като вградена зависимост)
# ============================================================

# ─── Stage 1: Builder ───
FROM python:3.11-slim AS builder

WORKDIR /build

# Build deps (само в builder stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build wheels за всички production dependencies (БЕЗ xgboost)
COPY requirements-prod.txt .
RUN pip wheel --no-cache-dir -r requirements-prod.txt -w /wheels

# Build xgboost wheel БЕЗ dependencies (без nvidia-*)
RUN pip wheel --no-cache-dir --no-deps "xgboost>=2.0.0,<3.0.0" -w /wheels

# ─── Stage 2: Runtime ───
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime deps (само curl за healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install wheels from builder (БЕЗ --no-deps — uvicorn има нужда от click, h11)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# App code + install as package (1:1 с GridForecast)
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Config + models + data
COPY pytest.ini .
COPY .coveragerc .
COPY models/ ./models/
COPY data/ ./data/

# Non-root user
RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run (1:1 с GridForecast)
CMD ["uvicorn", "powerguard.api.rest_api:app", "--host", "0.0.0.0", "--port", "8000"]