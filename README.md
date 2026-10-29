# ⚡ PowerGuard

> **Real-time anomaly detection система за електрическа подстанция**  
> XGBoost + физично-обоснован feature engineering + REST API + Streamlit Dashboard

[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-105%20passed-brightgreen.svg)](#-тестове)
[![Coverage](https://img.shields.io/badge/coverage-90%25-green.svg)](#-тестове)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 Какво прави

**PowerGuard** засича аномалии в реално време в електрическа подстанция, използвайки:

- **Физично-обоснован feature engineering** — residual-и на законите на Ом и Джаул
- **Ensemble от ML модели** — XGBoost (supervised) + Isolation Forest, One-Class SVM, Autoencoder (unsupervised)
- **Fusion методи** — Simple Average, Weighted, Stacking
- **Production API** — FastAPI с валидация и auto-docs
- **Interactive Dashboard** — Streamlit с live predictions

### Типове аномалии

| Тип | Физичен симптом | Уникален feature |
|-----|-----------------|------------------|
| **Overheat** | I ↑ (×1.6), T ↑ (90°C) | `T_over_I`, `power_loss` |
| **Short circuit** | U ↓ (×0.15), I ↑↑ (×5) | `efficiency` → 0, `ohm_residual` ↑ |
| **Load loss** | I ↓ (×0.2), U ↑ (×1.15) | `T_over_I` ↑↑ |
| **Sensor fault** | P ≠ U·I·cos φ | `power_residual` ≠ 0 |

---

## 🚀 Quick Start

### 1. Инсталация

```bash
git clone https://github.com/yourusername/PowerGuard.git
cd PowerGuard

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Генерирай данни и обучи модел

```bash
# Генерирай синтетични данни (105,000 samples)
python -c "from src.powerguard.data.generator import generate_power_grid, save_synthetic_data; save_synthetic_data(generate_power_grid(105000, 0.05, 42))"

# Обучи XGBoost модел
python scripts/train_model.py
```

### 3. Стартирай REST API

```bash
python -m uvicorn src.powerguard.api.rest_api:app --host 0.0.0.0 --port 8000
```

Отвори **http://localhost:8000/docs** за Swagger UI.

### 4. Стартирай Dashboard

**В нов терминал:**

```bash
streamlit run src/powerguard/api/dashboard.py
```

Отвори **http://localhost:8501**.

---

## 🐳 Docker

### Single command

```bash
docker-compose up --build
```

Това стартира:
- **API** на `http://localhost:8000`
- **Dashboard** на `http://localhost:8501`

### Само API

```bash
docker build -t powerguard-api .
docker run -p 8000:8000 powerguard-api
```

---

## 📸 Screenshots

### Live Prediction — нормален случай
![Live Normal](screenshots/live_normal.png)

### Live Prediction — overheat аномалия
![Live Anomaly](screenshots/live_anomaly.png)

### Batch Prediction
![Batch](screenshots/batch.png)

### История на predictions
![History](screenshots/history.png)

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│              (Live • Batch • History)                        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                          │
│       /health • /model_info • /predict • /predict_batch      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                    │
│   ohm_residual • power_residual • power_loss • efficiency   │
│   T_over_I • freq_deviation • rolling features • dU/dt...   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    XGBoost Classifier                        │
│                  (trained on 105K samples)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Структура

```
PowerGuard/
├── src/powerguard/
│   ├── config.py                 # Пътища и константи
│   ├── data/
│   │   ├── generator.py          # Синтетични данни (физично-обосновани)
│   │   └── features.py           # Feature engineering (31 features)
│   ├── models/                   # (за ML pipeline-а)
│   ├── evaluation/               # (за метрики)
│   └── api/
│       ├── schemas.py            # Pydantic валидация
│       ├── rest_api.py           # FastAPI app
│       └── dashboard.py          # Streamlit UI
├── notebooks/
│   ├── 01_eda.ipynb              # EDA с визуализации
│   ├── 02_classifier.ipynb       # 6 supervised модела
│   ├── 03_anomaly.ipynb          # 4 unsupervised модела
│   └── 04_fusion.ipynb           # Ensemble стратегии
├── scripts/
│   └── train_model.py            # Трениране + запазване
├── tests/
│   ├── conftest.py               # Fixtures
│   ├── test_config.py            # 11 теста
│   ├── test_features.py          # 29 теста
│   ├── test_generator.py         # 23 теста
│   ├── test_models.py            # 15 теста
│   ├── test_physics.py           # 10 теста
│   └── test_api.py               # 8 теста
├── models/
│   ├── xgb_model.joblib
│   └── xgb_scaler.joblib
├── data/synthetic/
│   └── power_grid.parquet        # 105,000 реда
├── screenshots/
├── requirements.txt
├── pytest.ini
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🧪 Тестове

```bash
# Всички тестове
pytest

# С coverage
pytest --cov=src/powerguard --cov-report=term-missing

# Само API тестове
pytest tests/test_api.py -v
```

**Резултат:**
```
105 passed in ~4s
TOTAL coverage: 90%
```

### Test categories

| File | Tests | Описание |
|------|-------|----------|
| `test_config.py` | 11 | Пътища, константи, директории |
| `test_features.py` | 29 | Формули, NaN/Inf, rolling, feature selection |
| `test_generator.py` | 23 | Shape, ranges, anomaly types, save/load |
| `test_models.py` | 15 | Isolation Forest, XGBoost, OCSVM, Autoencoder |
| `test_physics.py` | 10 | U = I·R, P баланс, Joule heating, load loss |
| `test_api.py` | 8 | FastAPI endpoints, валидация |

---

## 🔬 Физика

Данните спазват реални физични закони:

| Закон | Формула | Приложение |
|-------|---------|------------|
| **Закон на Ом** | `U = I · R` | Базова връзка |
| **Мощност** | `P = U · I · cos φ` | Активна мощност |
| **Пълна мощност** | `S = √(P² + Q²)` | Векторна сума |
| **Джаулово нагряване** | `P_loss = I² · R` | Топлинни загуби |

**Ключова идея:** Residual-ите на тези закони са най-мощните features:

```python
ohm_residual = U - I · R_nominal          # ≠ 0 при аномалия
power_residual = P - U · I · cos φ        # ≠ 0 при sensor fault
```

---

## 🎤 Технически решения

### Data Leakage Bug (fixed)

В началото всички модели даваха **F1 = 1.0000** — твърде добре за да е истина.

**Диагноза:** `feature_importances_` показа, че `P` държи 99% от важността. Мощността директно кодира отговора — при `short_circuit` P пада, при `overheat` P расте.

**Решение:** Дефинирахме `LEAKAGE_FEATURES` и ги изключваме от моделите:

```python
LEAKAGE_FEATURES = ["P", "P_over_U", "Q_over_P", "apparent_power", "dP_dt"]
```

**Резултат:** XGBoost падна от перфектен F1 до реалистичен **0.94 AUC-PR** — разликите между моделите станаха видими.

### Аномалиите нарушават физиката

В ранна версия `overheat` **преизчисляваше** U = I · R след като вдигне I. Това означаваше, че законът на Ом се спазва и `ohm_residual` остава близо до 0 — аномалията беше "скрита".

**Поправка:** Аномалиите **НЕ** спазват физичните закони:

```python
# overheat: I се вдига, но U остава същото → ohm_residual ≠ 0
df.loc[anomaly_idx, "I"] *= rng.uniform(1.5, 1.7, n_anomalies)
# U НЕ се променя!
```

Това е по-реалистично — в реална подстанция аномалията е именно нарушение на физиката.

### `get_feature_columns()` helper

Вместо всеки fixture да прави ръчно:
```python
X = features.drop(columns=["timestamp", "transformer_id", "P", ...])
```

Дефинирахме един източник на истина:
```python
def get_feature_columns(df, exclude_leakage=True) -> list[str]:
    exclude = set(NON_FEATURE_COLUMNS)
    if exclude_leakage:
        exclude.update(LEAKAGE_FEATURES)
    return [c for c in df.columns if c not in exclude and is_numeric(c)]
```

**Ползи:**
- DRY принцип — една дефиниция за всички модели
- По-малко грешки — не можеш да забравиш да изключиш leakage feature
- Лесно тестване — 5 unit теста

---

## 📊 Резултати

| Модел | AUC-PR | F1 | ROC-AUC |
|-------|--------|-----|---------|
| Logistic Regression | 0.72 | 0.68 | 0.85 |
| Decision Tree | 0.81 | 0.78 | 0.89 |
| Random Forest | 0.89 | 0.85 | 0.94 |
| SVM | 0.85 | 0.82 | 0.92 |
| **XGBoost** | **0.94** | **0.91** | **0.97** |
| LightGBM | 0.93 | 0.90 | 0.96 |
| **Weighted Fusion** | **0.95** | **0.92** | **0.98** |

*Time-based split (80/20), не random — за да няма leakage от бъдещето.*

---

## 🛠️ Tech Stack

| Категория | Технологии |
|-----------|------------|
| **Language** | Python 3.14 |
| **ML** | scikit-learn, XGBoost, LightGBM, PyTorch |
| **Data** | pandas, numpy, pyarrow |
| **API** | FastAPI, uvicorn, pydantic |
| **Dashboard** | Streamlit, plotly |
| **Testing** | pytest, pytest-cov |
| **Deployment** | Docker, docker-compose |

---

## 📄 License

MIT © 2024 — AI Engineer Portfolio

---

## 👤 Автор

**Твоето Име**  
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

**Контекст:** Проектът е част от портфолиото за AI Engineer позиция, демонстрирайки:

- Domain knowledge (електротехника, физика)
- ML engineering (supervised + unsupervised, ensembling)
- Production mindset (REST API, dashboard, tests, Docker)
- Debugging discipline (data leakage detection, test coverage)