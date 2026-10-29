"""
Споделени fixtures за всички тестове в PowerGuard.
"""
import pytest


# ═══════════════════════════════════════════════════════
# DATA FIXTURES
# ═══════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def sample_df():
    """Малък dataset за тестове (1000 samples)."""
    from powerguard.data.generator import generate_power_grid
    return generate_power_grid(n_samples=1000, anomaly_ratio=0.05, seed=42)


@pytest.fixture(scope="session")
def normal_df(sample_df):
    """Само нормални данни."""
    return sample_df[sample_df["is_anomaly"] == 0].copy()


@pytest.fixture(scope="session")
def anomaly_df(sample_df):
    """Само аномалии."""
    return sample_df[sample_df["is_anomaly"] == 1].copy()


@pytest.fixture(scope="session")
def sample_features(sample_df):
    """Features от sample_df."""
    from powerguard.data.features import build_features
    return build_features(sample_df)


# ═══════════════════════════════════════════════════════
# MODEL FIXTURES
# ═══════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def trained_iforest(sample_features):
    """Обучен Isolation Forest."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from powerguard.data.features import get_feature_columns

    feature_cols = get_feature_columns(sample_features, exclude_leakage=True)
    X = sample_features[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    return model, scaler, X


@pytest.fixture(scope="session")
def trained_xgb(sample_features, sample_df):
    """Обучен XGBoost."""
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from powerguard.data.features import get_feature_columns

    feature_cols = get_feature_columns(sample_features, exclude_leakage=True)
    X = sample_features[feature_cols]
    y = sample_df["is_anomaly"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=50, max_depth=4,
        scale_pos_weight=19, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_scaled, y)

    return model, scaler, X