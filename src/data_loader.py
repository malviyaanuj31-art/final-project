"""
data_loader.py — Load and validate datasets for AgriSmart AI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CROP_CSV      = _DATA_DIR / "crop_recommendation.csv"
REGRESSION_CSV = _DATA_DIR / "crop_yield_regression.csv"

# Required columns in the crop dataset
CROP_FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_TARGET_COL   = "label"
CROP_REQUIRED_COLS = CROP_FEATURE_COLS + [CROP_TARGET_COL]


class DataLoadError(RuntimeError):
    """Raised when a dataset cannot be loaded or is malformed."""


def load_crop_data() -> pd.DataFrame:
    """
    Load the crop recommendation CSV and validate its structure.

    Returns
    -------
    pd.DataFrame with columns N, P, K, temperature, humidity, ph, rainfall, label.

    Raises
    ------
    DataLoadError if the file is missing or columns are wrong.
    """
    if not CROP_CSV.exists():
        raise DataLoadError(
            f"Crop dataset not found at: {CROP_CSV}\n"
            "Please place 'crop_recommendation.csv' in the data/ folder.\n"
            "You can download the dataset from:\n"
            "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset\n"
            "Or run:  python data/generate_sample_data.py"
        )

    try:
        df = pd.read_csv(CROP_CSV)
    except Exception as exc:
        raise DataLoadError(f"Could not read crop dataset: {exc}") from exc

    missing = [c for c in CROP_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataLoadError(
            f"Crop dataset is missing required columns: {missing}\n"
            f"Expected: {CROP_REQUIRED_COLS}\n"
            f"Found:    {list(df.columns)}"
        )

    return df


def load_regression_data() -> pd.DataFrame:
    """
    Load the crop yield regression demo CSV.

    Returns
    -------
    pd.DataFrame or raises DataLoadError.
    """
    if not REGRESSION_CSV.exists():
        raise DataLoadError(
            f"Regression dataset not found at: {REGRESSION_CSV}\n"
            "Run:  python data/generate_regression_data.py"
        )
    try:
        df = pd.read_csv(REGRESSION_CSV)
    except Exception as exc:
        raise DataLoadError(f"Could not read regression dataset: {exc}") from exc
    return df


def get_crop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the input feature columns (no target)."""
    return df[CROP_FEATURE_COLS].copy()


def get_crop_labels(df: pd.DataFrame) -> pd.Series:
    """Return the target label column."""
    return df[CROP_TARGET_COL].copy()
