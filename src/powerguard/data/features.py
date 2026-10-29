"""
Feature engineering за PowerGuard.

Създава физично-обосновани features от суровите сензорни данни.
"""
import numpy as np
import pandas as pd

from powerguard.config import NOMINAL_R, NOMINAL_F


# ═══════════════════════════════════════════════════════
# FEATURE SELECTION CONSTANTS
# ═══════════════════════════════════════════════════════

# Features, които директно кодират отговора (data leakage)
LEAKAGE_FEATURES = [
    "P",
    "P_over_U",
    "Q_over_P",
    "apparent_power",
    "dP_dt",
]

# Колони, които никога не са features (metadata / target)
NON_FEATURE_COLUMNS = [
    "timestamp",
    "transformer_id",
    "is_anomaly",
    "anomaly_type",
]


# ═══════════════════════════════════════════════════════
# CREATE_POWER_FEATURES
# ═══════════════════════════════════════════════════════

def create_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """Създава физични features от сензорните данни.

    Включва:
    - Residuals на физичните закони (ohm_residual, power_residual)
    - Ефективност, загуби, отклонения
    - Временни производни (dU/dt, dI/dt, ...)
    - Съотношения (P/U, Q/P, I/U)
    """
    features = df.copy()

    # ─── Основни величини ───
    U = features["U"]
    I = features["I"]
    R = features["R"]
    P = features["P"]
    Q = features["Q"]
    T = features["T"]
    f = features["f"]
    cos_phi = features["cos_phi"]

    # ─── Физични изчисления ───
    features["R_calc"] = U / (I + 1e-6)
    features["apparent_power"] = np.sqrt(P ** 2 + Q ** 2)
    features["cos_phi_calc"] = P / (features["apparent_power"] + 1e-6)

    # ─── Residuals (нарушения на физичните закони) ───
    # Закон на Ом: U = I · R → residual = U - I · R_nominal
    features["ohm_residual"] = U - I * NOMINAL_R

    # Мощностен баланс: P = U · I · cos φ → residual = P - U · I · cos φ
    features["power_residual"] = P - U * I * cos_phi

    # Джаулови загуби: P_loss = I² · R
    features["power_loss"] = I ** 2 * R

    # Ефективност: P / (P + P_loss)
    features["efficiency"] = P / (P + features["power_loss"] + 1e-6)

    # ─── Отклонения ───
    features["freq_deviation"] = np.abs(f - NOMINAL_F)
    features["T_deviation"] = T - 25.0
    features["T_over_I"] = T / (I + 1e-6)

    # ─── Временни производни (промяна във времето) ───
    features["dU_dt"] = U.diff().fillna(0)
    features["dI_dt"] = I.diff().fillna(0)
    features["dP_dt"] = P.diff().fillna(0)
    features["dT_dt"] = T.diff().fillna(0)

    # ─── Съотношения ───
    features["P_over_U"] = P / (U + 1e-6)
    features["Q_over_P"] = Q / (P + 1e-6)
    features["I_over_U"] = I / (U + 1e-6)

    # Почисти Inf/NaN от деление
    features = features.replace([np.inf, -np.inf], 0)
    features = features.fillna(0)

    return features


# ═══════════════════════════════════════════════════════
# ADD_ROLLING_FEATURES
# ═══════════════════════════════════════════════════════

def add_rolling_features(
    df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Добавя rolling mean и std за ключови features.

    Args:
        df: DataFrame с features
        window: Прозорец за rolling

    Returns:
        DataFrame с добавени rolling колони
    """
    features = df.copy()

    cols_to_roll = ["U", "I", "T", "ohm_residual"]

    for col in cols_to_roll:
        if col not in features.columns:
            continue
        features[f"{col}_rolling_mean_{window}"] = (
            features[col].rolling(window=window, min_periods=1).mean()
        )
        features[f"{col}_rolling_std_{window}"] = (
            features[col].rolling(window=window, min_periods=1).std().fillna(0)
        )

    features = features.replace([np.inf, -np.inf], 0)
    features = features.fillna(0)

    return features


# ═══════════════════════════════════════════════════════
# BUILD_FEATURES (combined)
# ═══════════════════════════════════════════════════════

def build_features(
    df: pd.DataFrame,
    rolling_window: int = 10,
) -> pd.DataFrame:
    """Пълна pipeline: create_power_features + add_rolling_features.

    Args:
        df: Суров DataFrame от generator-а
        rolling_window: Прозорец за rolling features

    Returns:
        DataFrame с всички features
    """
    features = create_power_features(df)
    features = add_rolling_features(features, window=rolling_window)
    return features


# ═══════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════

def get_feature_columns(
    df: pd.DataFrame,
    exclude_leakage: bool = True,
) -> list[str]:
    """Връща списък с feature колони за ML моделите.

    Args:
        df: DataFrame с features
        exclude_leakage: Ако True, изключва LEAKAGE_FEATURES

    Returns:
        Списък с имена на колони (само числови)
    """
    exclude = set(NON_FEATURE_COLUMNS)
    if exclude_leakage:
        exclude.update(LEAKAGE_FEATURES)

    return [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]