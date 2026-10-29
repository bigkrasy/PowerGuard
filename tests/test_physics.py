"""
Тестове за физични закони.

Това са уникални тестове за нашия проект —
проверяваме че данните спазват физиката.
"""
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════
# ЗАКОН НА ОМ
# ═══════════════════════════════════════════════════════

def test_ohm_law_normal(sample_df):
    """U = I · R за нормални."""
    normal = sample_df[sample_df["is_anomaly"] == 0]
    residual = (normal["U"] - normal["I"] * normal["R"]).abs()
    relative_error = residual.mean() / normal["U"].mean()
    assert relative_error < 0.001


def test_ohm_law_anomalies(sample_df):
    """Аномалиите нарушават закона на Ом."""
    for atype in ["overheat", "short_circuit", "load_loss"]:
        subset = sample_df[sample_df["anomaly_type"] == atype]
        if len(subset) == 0:
            pytest.skip(f"Няма {atype}")
        residual = (subset["U"] - subset["I"] * subset["R"]).abs()
        large = (residual > 100).sum()
        assert large / len(subset) > 0.5


def test_ohm_law_sensor_fault_respected(sample_df):
    """Sensor fault НЕ нарушава закона на Ом."""
    sf = sample_df[sample_df["anomaly_type"] == "sensor_fault"]
    if len(sf) == 0:
        pytest.skip("Няма sensor_fault")

    residual = (sf["U"] - sf["I"] * sf["R"]).abs()
    # Sensor fault не пипа U, I, R → законът се спазва
    assert residual.mean() < 10


# ═══════════════════════════════════════════════════════
# МОЩНОСТЕН БАЛАНС
# ═══════════════════════════════════════════════════════

def test_power_balance_normal(sample_df):
    """P = U · I · cos φ за нормални."""
    normal = sample_df[sample_df["is_anomaly"] == 0]
    expected_P = normal["U"] * normal["I"] * normal["cos_phi"]
    actual_P = normal["P"]
    np.testing.assert_allclose(actual_P, expected_P, rtol=1e-6)


def test_power_balance_sensor_fault(sample_df):
    """Sensor fault нарушава мощностния баланс."""
    sf = sample_df[sample_df["anomaly_type"] == "sensor_fault"]
    if len(sf) == 0:
        pytest.skip("Няма sensor_fault")

    expected_P = sf["U"] * sf["I"] * sf["cos_phi"]
    actual_P = sf["P"]
    residual = (actual_P - expected_P).abs()

    large_residual = (residual > 1000).sum()
    assert large_residual / len(sf) > 0.5


# ═══════════════════════════════════════════════════════
# ДЖАУЛОВИ ЗАГУБИ
# ═══════════════════════════════════════════════════════

def test_joule_heating(sample_df):
    """P_loss = I² · R, overheat има по-високи загуби."""
    normal = sample_df[sample_df["anomaly_type"] == "normal"]
    overheat = sample_df[sample_df["anomaly_type"] == "overheat"]

    if len(overheat) == 0:
        pytest.skip("Няма overheat")

    P_loss_normal = (normal["I"] ** 2 * normal["R"]).mean()
    P_loss_overheat = (overheat["I"] ** 2 * overheat["R"]).mean()

    assert P_loss_overheat > P_loss_normal * 2


# ═══════════════════════════════════════════════════════
# SHORT CIRCUIT
# ═══════════════════════════════════════════════════════

def test_short_circuit_physics(sample_df):
    """Short circuit: U намалява, I расте."""
    normal = sample_df[sample_df["anomaly_type"] == "normal"]
    short = sample_df[sample_df["anomaly_type"] == "short_circuit"]

    if len(short) == 0:
        pytest.skip("Няма short_circuit")

    assert short["I"].mean() > normal["I"].mean() * 2
    assert short["U"].mean() < normal["U"].mean() * 0.5


def test_energy_efficiency_short_circuit(sample_df):
    """Short circuit има много ниска ефективност."""
    short = sample_df[sample_df["anomaly_type"] == "short_circuit"]
    normal = sample_df[sample_df["anomaly_type"] == "normal"]

    if len(short) == 0:
        pytest.skip("Няма short_circuit")

    eff_short = (short["P"] / (short["P"] + short["I"] ** 2 * short["R"])).mean()
    eff_normal = (normal["P"] / (normal["P"] + normal["I"] ** 2 * normal["R"])).mean()

    assert eff_short < eff_normal


# ═══════════════════════════════════════════════════════
# LOAD LOSS
# ═══════════════════════════════════════════════════════

def test_load_loss_physics(sample_df):
    """Load loss намалява I, увеличава U."""
    normal = sample_df[sample_df["anomaly_type"] == "normal"]
    load_loss = sample_df[sample_df["anomaly_type"] == "load_loss"]

    if len(load_loss) == 0:
        pytest.skip("Няма load_loss")

    assert load_loss["I"].mean() < normal["I"].mean() * 0.5
    assert load_loss["U"].mean() > normal["U"].mean()


# ═══════════════════════════════════════════════════════
# ЧЕСТОТА
# ═══════════════════════════════════════════════════════

def test_frequency_stable(sample_df):
    """Честотата е близо до 50 Hz."""
    assert sample_df["f"].mean() == pytest.approx(50.0, abs=0.1)
    assert sample_df["f"].std() < 0.1