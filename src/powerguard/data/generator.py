"""
Генератор на синтетични данни за електрическа подстанция.

Физично-обоснован:
- Закон на Ом: U = I · R
- Мощност: P = U · I · cos φ
- Джаулови загуби: P_loss = I² · R
"""
from pathlib import Path

import numpy as np
import pandas as pd

from powerguard.config import (
    NOMINAL_U,
    NOMINAL_I,
    NOMINAL_R,
    NOMINAL_F,
    RANDOM_SEED,
    SYNTHETIC_DATA_FILE,
)


# ═══════════════════════════════════════════════════════
# NORMAL OPERATION
# ═══════════════════════════════════════════════════════

def generate_normal_operation(
    n_samples: int = 1000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Генерира нормална работа на трансформатор.

    Args:
        n_samples: Брой проби
        seed: Random seed

    Returns:
        DataFrame с колони: timestamp, transformer_id, U, I, R, P, Q, T, f,
                            cos_phi, is_anomaly, anomaly_type
    """
    rng = np.random.default_rng(seed)

    # Timestamps — 1 минута интервал, започвайки от 2024-01-01
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=n_samples,
        freq="1min",
    )

    # Сензори с малък шум около номиналните стойности
    I = NOMINAL_I + rng.normal(0, NOMINAL_I * 0.05, n_samples)
    R = NOMINAL_R + rng.normal(0, NOMINAL_R * 0.005, n_samples)
    T = 25.0 + rng.normal(0, 2.0, n_samples)
    f = NOMINAL_F + rng.normal(0, 0.02, n_samples)
    cos_phi = 0.92 + rng.normal(0, 0.01, n_samples)
    cos_phi = np.clip(cos_phi, 0.85, 0.99)

    # Физични закони
    # U = I · R (закон на Ом)
    U = I * R

    # P = U · I · cos φ
    P = U * I * cos_phi

    # Q = U · I · sin φ, sin φ = sqrt(1 - cos²φ)
    sin_phi = np.sqrt(1 - cos_phi ** 2)
    Q = U * I * sin_phi

    df = pd.DataFrame({
        "timestamp": timestamps,
        "transformer_id": "TR-001",
        "U": U,
        "I": I,
        "R": R,
        "P": P,
        "Q": Q,
        "T": T,
        "f": f,
        "cos_phi": cos_phi,
        "is_anomaly": 0,
        "anomaly_type": "normal",
    })

    return df


# ═══════════════════════════════════════════════════════
# ANOMALY INJECTION
# ═══════════════════════════════════════════════════════

def inject_anomaly(
    df: pd.DataFrame,
    anomaly_type: str,
    n_anomalies: int = 100,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Инжектира аномалии от даден тип в DataFrame.

    ВАЖНО: Аномалиите НАРУШАВАТ физичните закони (това ги прави аномалии!):
    - overheat: I↑(×1.6), T↑(~90°C) → нарушава закона на Ом (U остава същото)
    - short_circuit: U↓(×0.15), I↑↑(×5) → нарушава закона на Ом
    - load_loss: I↓(×0.2), U↑(×1.15) → ниска мощност, нарушава закона на Ом
    - sensor_fault: P ≠ U·I·cos φ → нарушава мощностния баланс
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Избери random индекси за аномалии (само от нормални)
    normal_idx = df[df["is_anomaly"] == 0].index
    if len(normal_idx) < n_anomalies:
        n_anomalies = len(normal_idx)

    anomaly_idx = rng.choice(normal_idx, size=n_anomalies, replace=False)

    if anomaly_type == "overheat":
        # I расте, T скача. U НЕ се променя → законът на Ом (U = I·R) е нарушен!
        df.loc[anomaly_idx, "I"] *= rng.uniform(1.5, 1.7, n_anomalies)
        df.loc[anomaly_idx, "T"] = rng.uniform(85, 95, n_anomalies)
        # U остава същото → ohm_residual = U - I·R_nominal ще е голям
        # Преизчисляваме само P, Q с новия I
        cos_phi = df.loc[anomaly_idx, "cos_phi"]
        U_anom = df.loc[anomaly_idx, "U"]
        I_anom = df.loc[anomaly_idx, "I"]
        df.loc[anomaly_idx, "P"] = U_anom * I_anom * cos_phi
        df.loc[anomaly_idx, "Q"] = U_anom * I_anom * np.sqrt(1 - cos_phi ** 2)

    elif anomaly_type == "short_circuit":
        # U пада драстично, I скача, законът на Ом е нарушен
        df.loc[anomaly_idx, "U"] *= rng.uniform(0.10, 0.20, n_anomalies)
        df.loc[anomaly_idx, "I"] *= rng.uniform(4.0, 6.0, n_anomalies)
        # T расте заради загубите
        df.loc[anomaly_idx, "T"] = rng.uniform(60, 80, n_anomalies)
        # P се преизчислява от новите U, I (но законът на Ом вече не важи)
        cos_phi = df.loc[anomaly_idx, "cos_phi"]
        U_anom = df.loc[anomaly_idx, "U"]
        I_anom = df.loc[anomaly_idx, "I"]
        df.loc[anomaly_idx, "P"] = U_anom * I_anom * cos_phi
        df.loc[anomaly_idx, "Q"] = U_anom * I_anom * np.sqrt(1 - cos_phi ** 2)

    elif anomaly_type == "load_loss":
        # I пада, U РАСТЕ (по-малко товар → по-високо напрежение в реална мрежа)
        df.loc[anomaly_idx, "I"] *= rng.uniform(0.15, 0.25, n_anomalies)
        df.loc[anomaly_idx, "U"] *= rng.uniform(1.10, 1.20, n_anomalies)
        cos_phi = df.loc[anomaly_idx, "cos_phi"]
        U_anom = df.loc[anomaly_idx, "U"]
        I_anom = df.loc[anomaly_idx, "I"]
        df.loc[anomaly_idx, "P"] = U_anom * I_anom * cos_phi
        df.loc[anomaly_idx, "Q"] = U_anom * I_anom * np.sqrt(1 - cos_phi ** 2)

    elif anomaly_type == "sensor_fault":
        # P не съответства на U · I · cos φ
        # U, I, R остават нормални → законът на Ом се спазва
        # Но P е "счупен"
        df.loc[anomaly_idx, "P"] = (
            df.loc[anomaly_idx, "P"] * rng.uniform(2.0, 4.0, n_anomalies)
        )

    else:
        raise ValueError(f"Непознат тип аномалия: {anomaly_type}")

    df.loc[anomaly_idx, "is_anomaly"] = 1
    df.loc[anomaly_idx, "anomaly_type"] = anomaly_type

    return df


# ═══════════════════════════════════════════════════════
# GENERATE_POWER_GRID (combined)
# ═══════════════════════════════════════════════════════

def generate_power_grid(
    n_samples: int = 105000,
    anomaly_ratio: float = 0.05,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Генерира пълен dataset с нормални + аномалии.

    Генерира n_samples нормални, после инжектира n_samples * anomaly_ratio
    аномалии (които ЗАМЕНЯТ част от нормалните). Общият брой редове
    остава точно n_samples.

    Args:
        n_samples: Общ брой проби (крайният брой редове)
        anomaly_ratio: Дял аномалии (0.0 - 1.0)
        seed: Random seed

    Returns:
        DataFrame с точно n_samples реда
    """
    n_anomalies_total = int(n_samples * anomaly_ratio)

    # Генерираме n_samples нормални, после инжектираме аномалии (заменят част от тях)
    df = generate_normal_operation(n_samples=n_samples, seed=seed)

    # Разпредели аномалиите по равно между 4 типа
    anomaly_types = ["overheat", "short_circuit", "load_loss", "sensor_fault"]
    per_type = n_anomalies_total // len(anomaly_types)

    for i, atype in enumerate(anomaly_types):
        # Последният тип получава остатъка
        n = per_type if i < len(anomaly_types) - 1 else (
            n_anomalies_total - per_type * (len(anomaly_types) - 1)
        )
        df = inject_anomaly(df, atype, n_anomalies=n, seed=seed + i + 1)

    # Разбъркай
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════

def save_synthetic_data(
    df: pd.DataFrame,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Записва синтетичните данни в parquet файл.

    Args:
        df: DataFrame с данните
        path: Път за запис. По подразбиране: SYNTHETIC_DATA_FILE

    Returns:
        Същият DataFrame (за удобство при chaining)
    """
    if path is None:
        path = SYNTHETIC_DATA_FILE

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=False)

    return df


def load_synthetic_data(
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Зарежда синтетичните данни от parquet файл.

    Args:
        path: Път за четене. По подразбиране: SYNTHETIC_DATA_FILE

    Returns:
        DataFrame
    """
    if path is None:
        path = SYNTHETIC_DATA_FILE

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Липсва файл: {path}")

    return pd.read_parquet(path)