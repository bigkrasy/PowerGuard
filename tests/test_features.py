"""
Тестове за feature engineering.

Покрива:
- create_power_features (всички features)
- add_rolling_features
- build_features
- Формули
- Edge cases
"""
import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════
# CREATE_POWER_FEATURES
# ═══════════════════════════════════════════════════════

def test_create_power_features_shape(sample_df):
    """Правилен брой features."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    assert features.shape[0] == len(sample_df)
    assert features.shape[1] > 20


def test_create_power_features_expected(sample_df):
    """Всички очаквани features присъстват."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)

    expected = [
        "U", "I", "P", "Q", "T", "f",
        "R_calc", "apparent_power", "cos_phi_calc",
        "ohm_residual", "power_residual", "power_loss",
        "efficiency", "freq_deviation", "T_over_I", "T_deviation",
        "dU_dt", "dI_dt", "dP_dt", "dT_dt",
        "P_over_U", "Q_over_P", "I_over_U",
    ]
    for col in expected:
        assert col in features.columns, f"Липсва: {col}"


def test_create_power_features_no_nan(sample_df):
    """Няма NaN."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    assert features.isnull().sum().sum() == 0


def test_create_power_features_no_inf(sample_df):
    """Няма Inf."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    numeric = features.select_dtypes(include=[np.number])
    assert np.isinf(numeric).sum().sum() == 0


def test_ohm_residual_formula(sample_df):
    """ohm_residual = U - I * NOMINAL_R."""
    from powerguard.data.features import create_power_features
    from powerguard.config import NOMINAL_R

    features = create_power_features(sample_df)
    expected = sample_df["U"].values - sample_df["I"].values * NOMINAL_R
    actual = features["ohm_residual"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_power_residual_formula(sample_df):
    """power_residual = P - U * I * cos φ."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    expected = (
        sample_df["P"].values
        - sample_df["U"].values * sample_df["I"].values * sample_df["cos_phi"].values
    )
    actual = features["power_residual"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_power_loss_formula(sample_df):
    """power_loss = I² * R."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    expected = sample_df["I"].values ** 2 * sample_df["R"].values
    actual = features["power_loss"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_apparent_power_formula(sample_df):
    """apparent_power = sqrt(P² + Q²)."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    expected = np.sqrt(sample_df["P"].values ** 2 + sample_df["Q"].values ** 2)
    actual = features["apparent_power"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_r_calc_formula(sample_df):
    """R_calc = U / (I + 1e-6)."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    expected = sample_df["U"].values / (sample_df["I"].values + 1e-6)
    actual = features["R_calc"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_cos_phi_calc_formula(sample_df):
    """cos_phi_calc = P / (apparent_power + 1e-6)."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    S = np.sqrt(sample_df["P"].values ** 2 + sample_df["Q"].values ** 2)
    expected = sample_df["P"].values / (S + 1e-6)
    actual = features["cos_phi_calc"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_freq_deviation_formula(sample_df):
    """freq_deviation = |f - 50|."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    expected = np.abs(sample_df["f"].values - 50.0)
    actual = features["freq_deviation"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_freq_deviation_positive(sample_df):
    """freq_deviation >= 0."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    assert (features["freq_deviation"] >= 0).all()


def test_T_over_I_formula(sample_df):
    """T_over_I = T / (I + 1e-6)."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    expected = sample_df["T"].values / (sample_df["I"].values + 1e-6)
    actual = features["T_over_I"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_T_deviation_formula(sample_df):
    """T_deviation = T - 25."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    expected = sample_df["T"].values - 25.0
    actual = features["T_deviation"].values
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_efficiency_range(sample_df):
    """efficiency е в [0, 1.5]."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    assert (features["efficiency"] >= 0).all()
    assert (features["efficiency"] <= 1.5).all()


def test_temporal_features(sample_df):
    """dU_dt, dI_dt, dP_dt, dT_dt са с правилен shape."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    for col in ["dU_dt", "dI_dt", "dP_dt", "dT_dt"]:
        assert col in features.columns
        assert len(features[col]) == len(sample_df)


def test_ratios_positive(sample_df):
    """P_over_U, Q_over_P, I_over_U са разумни."""
    from powerguard.data.features import create_power_features

    features = create_power_features(sample_df)
    for col in ["P_over_U", "Q_over_P", "I_over_U"]:
        assert col in features.columns
        assert np.isfinite(features[col]).all()


# ═══════════════════════════════════════════════════════
# ADD_ROLLING_FEATURES
# ═══════════════════════════════════════════════════════

def test_add_rolling_features(sample_df):
    """add_rolling_features добавя rolling колони."""
    from powerguard.data.features import (
        create_power_features, add_rolling_features,
    )

    features = create_power_features(sample_df)
    features_rolling = add_rolling_features(features, window=10)

    # Провери колони
    expected_cols = [
        "U_rolling_std_10", "U_rolling_mean_10",
        "I_rolling_std_10", "I_rolling_mean_10",
        "T_rolling_std_10", "T_rolling_mean_10",
        "ohm_residual_rolling_std_10", "ohm_residual_rolling_mean_10",
    ]
    for col in expected_cols:
        assert col in features_rolling.columns, f"Липсва: {col}"

    # Провери shape
    assert len(features_rolling) == len(features)


def test_add_rolling_features_custom_window(sample_df):
    """add_rolling_features работи с custom window."""
    from powerguard.data.features import (
        create_power_features, add_rolling_features,
    )

    features = create_power_features(sample_df)
    features_rolling = add_rolling_features(features, window=5)

    assert "U_rolling_std_5" in features_rolling.columns
    assert "U_rolling_mean_5" in features_rolling.columns


def test_add_rolling_features_no_nan(sample_df):
    """Rolling features нямат NaN."""
    from powerguard.data.features import (
        create_power_features, add_rolling_features,
    )

    features = create_power_features(sample_df)
    features_rolling = add_rolling_features(features, window=10)

    assert features_rolling.isnull().sum().sum() == 0


# ═══════════════════════════════════════════════════════
# BUILD_FEATURES (combined)
# ═══════════════════════════════════════════════════════

def test_build_features(sample_df):
    """build_features комбинира всички features."""
    from powerguard.data.features import build_features

    features = build_features(sample_df)

    # Основни features
    assert "ohm_residual" in features.columns
    assert "power_residual" in features.columns

    # Rolling features
    assert "U_rolling_std_10" in features.columns

    # Shape
    assert len(features) == len(sample_df)
    assert features.shape[1] > 30


def test_build_features_custom_window(sample_df):
    """build_features работи с custom window."""
    from powerguard.data.features import build_features

    features = build_features(sample_df, rolling_window=5)
    assert "U_rolling_std_5" in features.columns


def test_build_features_no_nan(sample_df):
    """build_features няма NaN."""
    from powerguard.data.features import build_features

    features = build_features(sample_df)
    assert features.isnull().sum().sum() == 0


# ═══════════════════════════════════════════════════════
# RESIDUALS за НОРМАЛНИ vs АНОМАЛИИ
# ═══════════════════════════════════════════════════════

@pytest.mark.parametrize("residual_col", ["ohm_residual", "power_residual"])
def test_residuals_near_zero_for_normal(sample_df, sample_features, residual_col):
    """Residuals са близо до 0 за нормални."""
    normal_mask = sample_df["is_anomaly"] == 0
    normal_residuals = sample_features.loc[normal_mask, residual_col]

    assert normal_residuals.abs().mean() < 100


def test_ohm_residual_large_for_anomaly(sample_df, sample_features):
    """Ohm residual е голям за short_circuit."""
    mask = sample_df["anomaly_type"] == "short_circuit"
    if mask.sum() == 0:
        pytest.skip("Няма short_circuit")

    short_residuals = sample_features.loc[mask, "ohm_residual"]
    assert short_residuals.abs().mean() > 1000
    
    
# ═══════════════════════════════════════════════════════
# GET_FEATURE_COLUMNS
# ═══════════════════════════════════════════════════════

def test_get_feature_columns_excludes_leakage(sample_features):
    """Leakage features се изключват при exclude_leakage=True."""
    from powerguard.data.features import (
        get_feature_columns, LEAKAGE_FEATURES,
    )

    cols = get_feature_columns(sample_features, exclude_leakage=True)
    for leak in LEAKAGE_FEATURES:
        assert leak not in cols, f"Leakage feature присъства: {leak}"


def test_get_feature_columns_includes_leakage(sample_features):
    """Leakage features се включват при exclude_leakage=False."""
    from powerguard.data.features import (
        get_feature_columns, LEAKAGE_FEATURES,
    )

    cols = get_feature_columns(sample_features, exclude_leakage=False)
    for leak in LEAKAGE_FEATURES:
        if leak in sample_features.columns:
            assert leak in cols, f"Leakage feature липсва: {leak}"


def test_get_feature_columns_excludes_metadata(sample_features):
    """Metadata колони не са features."""
    from powerguard.data.features import (
        get_feature_columns, NON_FEATURE_COLUMNS,
    )

    cols = get_feature_columns(sample_features)
    for meta in NON_FEATURE_COLUMNS:
        assert meta not in cols, f"Metadata присъства: {meta}"


def test_get_feature_columns_returns_list_of_str(sample_features):
    """Връща list от str."""
    from powerguard.data.features import get_feature_columns

    cols = get_feature_columns(sample_features)
    assert isinstance(cols, list)
    assert all(isinstance(c, str) for c in cols)
    assert len(cols) > 0


def test_get_feature_columns_matches_dataframe(sample_features):
    """Всички върнати колони съществуват в df."""
    from powerguard.data.features import get_feature_columns

    cols = get_feature_columns(sample_features)
    for c in cols:
        assert c in sample_features.columns
        
        
# ═══════════════════════════════════════════════════════
# EDGE CASES — покритие на непокрити редове
# ═══════════════════════════════════════════════════════

def test_add_rolling_features_missing_column():
    """add_rolling_features работи дори ако колона липсва."""
    import pandas as pd
    from powerguard.data.features import add_rolling_features

    # DataFrame БЕЗ 'U' и 'ohm_residual' колони
    df = pd.DataFrame({
        "I": [1.0, 2.0, 3.0, 4.0],
        "T": [20.0, 21.0, 22.0, 23.0],
    })

    result = add_rolling_features(df, window=2)

    # Липсващите колони НЕ трябва да създават rolling features
    assert "U_rolling_mean_2" not in result.columns
    assert "ohm_residual_rolling_mean_2" not in result.columns
    # Наличните трябва да са обработени
    assert "I_rolling_mean_2" in result.columns
    assert "T_rolling_mean_2" in result.columns