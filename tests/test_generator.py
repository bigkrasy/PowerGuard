"""
Тестове за data generator.

Покрива:
- generate_normal_operation
- inject_anomaly (всички типове)
- generate_power_grid
- save_synthetic_data
- load_synthetic_data
"""
import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════
# GENERATE_NORMAL_OPERATION
# ═══════════════════════════════════════════════════════

def test_generate_normal_operation():
    """generate_normal_operation връща правилен DataFrame."""
    from powerguard.data.generator import generate_normal_operation

    df = generate_normal_operation(n_samples=500, seed=42)

    assert len(df) == 500
    assert "is_anomaly" in df.columns
    assert (df["is_anomaly"] == 0).all()
    assert (df["anomaly_type"] == "normal").all()


def test_generate_normal_operation_columns():
    """Всички очаквани колони присъстват."""
    from powerguard.data.generator import generate_normal_operation

    df = generate_normal_operation(n_samples=100, seed=42)
    expected = [
        "timestamp", "transformer_id", "U", "I", "R", "P", "Q", "T", "f",
        "cos_phi", "is_anomaly", "anomaly_type",
    ]
    for col in expected:
        assert col in df.columns


# ═══════════════════════════════════════════════════════
# INJECT_ANOMALY
# ═══════════════════════════════════════════════════════

@pytest.mark.parametrize("anomaly_type", [
    "overheat", "short_circuit", "load_loss", "sensor_fault",
])
def test_inject_anomaly_types(anomaly_type):
    """inject_anomaly работи за всеки тип."""
    from powerguard.data.generator import (
        generate_normal_operation, inject_anomaly,
    )

    df_normal = generate_normal_operation(n_samples=500, seed=42)
    df_anomaly = inject_anomaly(df_normal, anomaly_type, n_anomalies=50, seed=42)

    assert len(df_anomaly) == 500
    assert (df_anomaly["is_anomaly"] == 1).sum() == 50
    assert anomaly_type in df_anomaly["anomaly_type"].values


def test_inject_anomaly_overheat():
    """Overheat увеличава I и T."""
    from powerguard.data.generator import (
        generate_normal_operation, inject_anomaly,
    )

    df_normal = generate_normal_operation(n_samples=500, seed=42)
    df_anomaly = inject_anomaly(df_normal, "overheat", n_anomalies=50, seed=42)

    anomalies = df_anomaly[df_anomaly["is_anomaly"] == 1]
    assert anomalies["T"].mean() > 80


def test_inject_anomaly_short_circuit():
    """Short circuit увеличава I, намалява U."""
    from powerguard.data.generator import (
        generate_normal_operation, inject_anomaly,
    )

    df_normal = generate_normal_operation(n_samples=500, seed=42)
    df_anomaly = inject_anomaly(df_normal, "short_circuit", n_anomalies=50, seed=42)

    anomalies = df_anomaly[df_anomaly["is_anomaly"] == 1]
    assert anomalies["I"].mean() > 300
    assert anomalies["U"].mean() < 5000


def test_inject_anomaly_load_loss():
    """Load loss намалява I."""
    from powerguard.data.generator import (
        generate_normal_operation, inject_anomaly,
    )

    df_normal = generate_normal_operation(n_samples=500, seed=42)
    df_anomaly = inject_anomaly(df_normal, "load_loss", n_anomalies=50, seed=42)

    anomalies = df_anomaly[df_anomaly["is_anomaly"] == 1]
    assert anomalies["I"].mean() < 40


def test_inject_anomaly_sensor_fault():
    """Sensor fault увеличава P."""
    from powerguard.data.generator import (
        generate_normal_operation, inject_anomaly,
    )

    df_normal = generate_normal_operation(n_samples=500, seed=42)
    df_anomaly = inject_anomaly(df_normal, "sensor_fault", n_anomalies=50, seed=42)

    anomalies = df_anomaly[df_anomaly["is_anomaly"] == 1]
    normal = df_anomaly[df_anomaly["is_anomaly"] == 0]
    assert anomalies["P"].mean() > normal["P"].mean()


def test_inject_anomaly_unknown_type():
    """Непознат тип хвърля ValueError."""
    from powerguard.data.generator import (
        generate_normal_operation, inject_anomaly,
    )

    df = generate_normal_operation(n_samples=100, seed=42)
    with pytest.raises(ValueError, match="Непознат тип аномалия"):
        inject_anomaly(df, "unknown_type", n_anomalies=10, seed=42)


# ═══════════════════════════════════════════════════════
# GENERATE_POWER_GRID
# ═══════════════════════════════════════════════════════

def test_generate_power_grid_shape(sample_df):
    """Точен брой редове — n_samples общо (нормални + аномалии)."""
    # n_samples=1000 → точно 1000 реда (аномалиите заменят нормални)
    assert len(sample_df) == 1000


def test_generate_power_grid_columns(sample_df):
    """Всички колони присъстват."""
    expected_cols = [
        "timestamp", "transformer_id", "U", "I", "R", "P", "Q", "T", "f",
        "cos_phi", "is_anomaly", "anomaly_type",
    ]
    for col in expected_cols:
        assert col in sample_df.columns


def test_generate_power_grid_no_nan(sample_df):
    """Няма missing values."""
    assert not sample_df.isnull().any().any()


def test_generate_power_grid_sensor_ranges(sample_df):
    """Сензорите са в разумни диапазони."""
    assert sample_df["U"].min() > 0
    assert sample_df["U"].max() < 50000
    assert sample_df["I"].min() > 0
    assert sample_df["I"].max() < 2000
    assert sample_df["T"].min() > -50
    assert sample_df["T"].max() < 200
    assert sample_df["f"].min() > 45
    assert sample_df["f"].max() < 55


def test_generate_power_grid_cos_phi(sample_df):
    """cos φ е в [0, 1]."""
    assert sample_df["cos_phi"].min() >= 0
    assert sample_df["cos_phi"].max() <= 1


def test_generate_power_grid_anomaly_ratio(sample_df):
    """~5% аномалии."""
    ratio = sample_df["is_anomaly"].mean()
    assert 0.03 <= ratio <= 0.08


def test_generate_power_grid_anomaly_types(sample_df):
    """4 типа аномалии."""
    counts = sample_df["anomaly_type"].value_counts()
    for atype in ["overheat", "short_circuit", "load_loss", "sensor_fault"]:
        assert atype in counts
        assert counts[atype] > 0
    assert "normal" in counts


def test_generate_power_grid_timestamps(sample_df):
    """Timestamp-и са в 2024."""
    min_ts = sample_df["timestamp"].min()
    max_ts = sample_df["timestamp"].max()
    assert min_ts.year == 2024
    assert max_ts.year == 2024


def test_generate_power_grid_reproducible():
    """Един и същ seed → един и същ резултат."""
    from powerguard.data.generator import generate_power_grid

    df1 = generate_power_grid(n_samples=100, seed=42)
    df2 = generate_power_grid(n_samples=100, seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_generate_power_grid_anomalies_differ(sample_df):
    """Аномалиите имат различни стойности."""
    normal = sample_df[sample_df["is_anomaly"] == 0]

    overheat = sample_df[sample_df["anomaly_type"] == "overheat"]
    assert overheat["T"].mean() > normal["T"].mean()

    short = sample_df[sample_df["anomaly_type"] == "short_circuit"]
    assert short["I"].mean() > normal["I"].mean() * 2
    assert short["U"].mean() < normal["U"].mean() * 0.5

    load_loss = sample_df[sample_df["anomaly_type"] == "load_loss"]
    assert load_loss["I"].mean() < normal["I"].mean() * 0.5


# ═══════════════════════════════════════════════════════
# SAVE / LOAD SYNTHETIC DATA
# ═══════════════════════════════════════════════════════

def test_save_synthetic_data(tmp_path, sample_df):
    """save_synthetic_data записва parquet файл."""
    from powerguard.data.generator import save_synthetic_data

    out_path = tmp_path / "test.parquet"
    result = save_synthetic_data(sample_df, out_path)

    assert result is not None
    assert out_path.exists()
    assert out_path.suffix == ".parquet"

    loaded = pd.read_parquet(out_path)
    assert len(loaded) == len(sample_df)
    assert list(loaded.columns) == list(sample_df.columns)


def test_save_synthetic_data_creates_dirs(tmp_path, sample_df):
    """save_synthetic_data създава липсващи директории."""
    from powerguard.data.generator import save_synthetic_data

    out_path = tmp_path / "nested" / "deep" / "test.parquet"
    save_synthetic_data(sample_df, out_path)

    assert out_path.exists()
    assert out_path.parent.is_dir()


def test_save_synthetic_data_default_path(tmp_path, sample_df, monkeypatch):
    """save_synthetic_data с path=None използва SYNTHETIC_DATA_FILE."""
    from powerguard.data import generator

    fake_path = tmp_path / "default.parquet"
    monkeypatch.setattr(generator, "SYNTHETIC_DATA_FILE", fake_path)

    generator.save_synthetic_data(sample_df, path=None)
    assert fake_path.exists()


def test_load_synthetic_data(tmp_path, sample_df):
    """load_synthetic_data зарежда записания файл."""
    from powerguard.data.generator import (
        save_synthetic_data, load_synthetic_data,
    )

    out_path = tmp_path / "test.parquet"
    save_synthetic_data(sample_df, out_path)

    loaded = load_synthetic_data(out_path)
    assert len(loaded) == len(sample_df)
    pd.testing.assert_frame_equal(loaded, sample_df.reset_index(drop=True))


def test_load_synthetic_data_missing_file(tmp_path):
    """load_synthetic_data хвърля FileNotFoundError."""
    from powerguard.data.generator import load_synthetic_data

    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        load_synthetic_data(missing)
        
        
def test_inject_anomaly_more_than_available():
    """inject_anomaly с n_anomalies > налични нормални → ползва всички."""
    from powerguard.data.generator import (
        generate_normal_operation, inject_anomaly,
    )

    df = generate_normal_operation(n_samples=10, seed=42)
    # Искаме 100 аномалии, но имаме само 10 нормални
    result = inject_anomaly(df, "overheat", n_anomalies=100, seed=42)

    # Всичките 10 трябва да са аномалии (не повече)
    assert len(result) == 10
    assert (result["is_anomaly"] == 1).sum() == 10


def test_load_synthetic_data_default_path_missing(monkeypatch, tmp_path):
    """load_synthetic_data с path=None и липсващ default файл."""
    from powerguard.data import generator
    import pytest

    fake_path = tmp_path / "does_not_exist.parquet"
    monkeypatch.setattr(generator, "SYNTHETIC_DATA_FILE", fake_path)

    with pytest.raises(FileNotFoundError):
        generator.load_synthetic_data(path=None)