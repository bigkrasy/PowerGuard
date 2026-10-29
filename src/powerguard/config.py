"""
Конфигурация на PowerGuard проекта.
Всички пътища се строят от PROJECT_ROOT.
"""
from pathlib import Path

# ═══════════════════════════════════════════════════════
# PROJECT ROOT
# ═══════════════════════════════════════════════════════
# config.py е в src/powerguard/config.py
# PROJECT_ROOT е 3 нива нагоре
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ═══════════════════════════════════════════════════════
# DATA DIRECTORIES
# ═══════════════════════════════════════════════════════
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# ═══════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════
MODELS_DIR = PROJECT_ROOT / "models"

# ═══════════════════════════════════════════════════════
# NOTEBOOKS / OUTPUTS
# ═══════════════════════════════════════════════════════
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# ═══════════════════════════════════════════════════════
# SPECIFIC FILES
# ═══════════════════════════════════════════════════════
SYNTHETIC_DATA_FILE = SYNTHETIC_DIR / "power_grid.parquet"
MODEL_V1 = MODELS_DIR / "powerguard_v1.pkl"

# ═══════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════
RANDOM_SEED = 42

# Номинални стойности на веригата
NOMINAL_U = 11000.0     # V
NOMINAL_I = 100.0       # A
NOMINAL_R = 110.0       # Ω
NOMINAL_F = 50.0        # Hz

# ═══════════════════════════════════════════════════════
# AUTO-CREATE DIRECTORIES
# ═══════════════════════════════════════════════════════
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR,
          MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)