from __future__ import annotations

from pathlib import Path


# Project root is the directory containing this package's parent
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = ROOT_DIR / "data"
ENCODINGS_DIR: Path = DATA_DIR / "encodings"

# Canonical datasets
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed_data.csv"
LATEST_RACE_DATA_PATH: Path = DATA_DIR / "latest_race_data.csv"

# Model artifacts (kept at project root to preserve current workflow)
XGB_MODEL_PATH: Path = ROOT_DIR / "xgb_model.json"
PREPROCESSOR_PATH: Path = ROOT_DIR / "preprocessor.pkl"
LABEL_ENCODER_PATH: Path = ROOT_DIR / "label_encoder.pkl"


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ENCODINGS_DIR.mkdir(parents=True, exist_ok=True)


def session_year_file(year: int) -> Path:
    return DATA_DIR / f"session_data_{year}.csv"


def list_session_year_files() -> list[Path]:
    return sorted(DATA_DIR.glob("session_data_*.csv"))


