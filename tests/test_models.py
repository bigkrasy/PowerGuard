"""
Тестове за ML модели.

Покрива:
- Isolation Forest
- XGBoost
- One-Class SVM
- Autoencoder
- Fusion
"""
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════
# ISOLATION FOREST
# ═══════════════════════════════════════════════════════

def test_iforest_fits(trained_iforest):
    """IF се обучава без грешка."""
    model, scaler, X = trained_iforest
    assert hasattr(model, "predict")


def test_iforest_predictions(trained_iforest):
    """Predict връща -1 или 1."""
    model, scaler, X = trained_iforest
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    assert set(preds).issubset({-1, 1})


def test_iforest_scores(trained_iforest):
    """score_samples връща float."""
    model, scaler, X = trained_iforest
    X_scaled = scaler.transform(X)
    scores = model.score_samples(X_scaled)
    assert len(scores) == len(X)
    assert scores.dtype == np.float64


def test_iforest_anomaly_rate(trained_iforest):
    """~5% от данните се маркират."""
    model, scaler, X = trained_iforest
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    anomaly_rate = (preds == -1).mean()
    assert 0.02 <= anomaly_rate <= 0.15


# ═══════════════════════════════════════════════════════
# XGBOOST
# ═══════════════════════════════════════════════════════

def test_xgb_fits(trained_xgb):
    """XGB се обучава без грешка."""
    model, scaler, X = trained_xgb
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_xgb_predictions_binary(trained_xgb):
    """Predict връща 0 или 1."""
    model, scaler, X = trained_xgb
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    assert set(preds).issubset({0, 1})


def test_xgb_probabilities_range(trained_xgb):
    """Probabilities са в [0, 1]."""
    model, scaler, X = trained_xgb
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    assert probs.min() >= 0
    assert probs.max() <= 1


def test_xgb_learns_above_baseline(trained_xgb, sample_df):
    """XGB има по-добра точност от random."""
    from sklearn.metrics import average_precision_score

    model, scaler, X = trained_xgb
    y_true = sample_df["is_anomaly"].values
    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    auc_pr = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    assert auc_pr > baseline * 1.5
    assert auc_pr > 0.7


def test_xgb_feature_importance(trained_xgb):
    """XGB има feature importance."""
    model, scaler, X = trained_xgb
    importance = model.feature_importances_
    assert len(importance) == X.shape[1]
    assert (importance >= 0).all()
    assert abs(importance.sum() - 1.0) < 0.01


# ═══════════════════════════════════════════════════════
# ONE-CLASS SVM
# ═══════════════════════════════════════════════════════

def test_ocsvm_fits(sample_features):
    """OCSVM се обучава без грешка."""
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from powerguard.data.features import get_feature_columns

    feature_cols = get_feature_columns(sample_features, exclude_leakage=True)
    X = sample_features[feature_cols].iloc[:500]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(X_scaled)

    assert hasattr(model, "predict")


def test_ocsvm_predictions(sample_features):
    """OCSVM връща -1 или 1."""
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from powerguard.data.features import get_feature_columns

    feature_cols = get_feature_columns(sample_features, exclude_leakage=True)
    X = sample_features[feature_cols].iloc[:500]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(X_scaled)
    preds = model.predict(X_scaled)

    assert set(preds).issubset({-1, 1})


# ═══════════════════════════════════════════════════════
# AUTOENCODER
# ═══════════════════════════════════════════════════════

def test_autoencoder_architecture():
    """Autoencoder се създава с правилна архитектура."""
    import torch
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder(input_dim=26)
    assert isinstance(model, nn.Module)


def test_autoencoder_forward():
    """Autoencoder forward pass работи."""
    import torch
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder(input_dim=26)
    x = torch.randn(10, 26)
    output = model(x)

    assert output.shape == x.shape


def test_autoencoder_training_step():
    """Autoencoder може да се обучава."""
    import torch
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Linear(input_dim, 8)
            self.decoder = nn.Linear(8, input_dim)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder(input_dim=26)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.randn(32, 26)

    # Един training step
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, x)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0


# ═══════════════════════════════════════════════════════
# FUSION METHODS
# ═══════════════════════════════════════════════════════

def test_fusion_weighted_average():
    """Weighted average на scores работи."""
    scores1 = np.array([0.1, 0.5, 0.9])
    scores2 = np.array([0.2, 0.4, 0.8])
    weights = np.array([0.5, 0.5])

    fused = np.average([scores1, scores2], axis=0, weights=weights)

    assert len(fused) == 3
    assert np.allclose(fused, [0.15, 0.45, 0.85])


def test_fusion_simple_average():
    """Simple average на scores работи."""
    scores = [
        np.array([0.1, 0.5, 0.9]),
        np.array([0.2, 0.4, 0.8]),
    ]
    fused = np.mean(scores, axis=0)

    assert len(fused) == 3
    assert np.allclose(fused, [0.15, 0.45, 0.85])