"""
Тренира XGBoost модел и го запазва в models/.

Стартиране:
    python scripts/train_model.py
"""
import sys
from pathlib import Path

# Добави root към path (за да намери src.powerguard)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from powerguard.config import MODELS_DIR, SYNTHETIC_DATA_FILE
from powerguard.data.generator import load_synthetic_data
from powerguard.data.features import build_features, get_feature_columns


def main():
    print("Зареждане на данни...")
    df = load_synthetic_data(SYNTHETIC_DATA_FILE)
    print(f"   {len(df):,} реда")

    print("Feature engineering...")
    features = build_features(df)
    feature_cols = get_feature_columns(features, exclude_leakage=True)
    X = features[feature_cols]
    y = df["is_anomaly"].values
    print(f"   {len(feature_cols)} features")

    print("Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Трениране на XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)

    # Запази
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "xgb_scaler.joblib")

    print(f"Модел запазен в {MODELS_DIR}")


if __name__ == "__main__":
    main()