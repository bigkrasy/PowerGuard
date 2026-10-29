"""
Тестове за config.py.

Покрива:
- PROJECT_ROOT
- Всички пътища
- Номинални стойности
- Auto-create на директории
"""
from pathlib import Path


def test_project_root_exists():
    """PROJECT_ROOT съществува."""
    from powerguard.config import PROJECT_ROOT
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_project_root_correct():
    """PROJECT_ROOT сочи към PowerGuard."""
    from powerguard.config import PROJECT_ROOT
    assert PROJECT_ROOT.name == "PowerGuard"


def test_data_directories_created():
    """Всички data директории съществуват."""
    from powerguard.config import (
        DATA_DIR, RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR,
    )
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR]:
        assert d.exists(), f"Липсва: {d}"
        assert d.is_dir()


def test_models_directory_created():
    """Models директорията съществува."""
    from powerguard.config import MODELS_DIR
    assert MODELS_DIR.exists()
    assert MODELS_DIR.is_dir()


def test_figures_directory_created():
    """Figures директорията съществува."""
    from powerguard.config import FIGURES_DIR
    assert FIGURES_DIR.exists()


def test_results_directory_created():
    """Results директорията съществува."""
    from powerguard.config import RESULTS_DIR
    assert RESULTS_DIR.exists()


def test_paths_are_path_objects():
    """Всички пътища са Path обекти."""
    from powerguard.config import (
        PROJECT_ROOT, DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR,
    )
    for p in [PROJECT_ROOT, DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
        assert isinstance(p, Path)


def test_synthetic_data_file():
    """SYNTHETIC_DATA_FILE е правилен път."""
    from powerguard.config import SYNTHETIC_DATA_FILE
    assert isinstance(SYNTHETIC_DATA_FILE, Path)
    assert SYNTHETIC_DATA_FILE.name == "power_grid.parquet"


def test_random_seed():
    """RANDOM_SEED е зададен."""
    from powerguard.config import RANDOM_SEED
    assert RANDOM_SEED == 42


def test_nominal_values():
    """Номиналните стойности са разумни."""
    from powerguard.config import NOMINAL_U, NOMINAL_I, NOMINAL_R, NOMINAL_F

    assert NOMINAL_U == 11000.0
    assert NOMINAL_I == 100.0
    assert NOMINAL_R == 110.0
    assert NOMINAL_F == 50.0


def test_ohm_law_nominal():
    """U = I · R за номинални стойности."""
    from powerguard.config import NOMINAL_U, NOMINAL_I, NOMINAL_R
    assert abs(NOMINAL_U - NOMINAL_I * NOMINAL_R) < 1e-6