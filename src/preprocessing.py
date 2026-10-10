"""
preprocessing.py — Feature scaling and label encoding for AgriSmart AI.

All preprocessing steps are fit on training data only to prevent data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def build_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit and return a StandardScaler on training features."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """Apply a pre-fit scaler to a feature DataFrame."""
    return scaler.transform(X)


def build_label_encoder(y_train: pd.Series) -> LabelEncoder:
    """Fit and return a LabelEncoder on training labels."""
    le = LabelEncoder()
    le.fit(y_train)
    return le


def encode_labels(le: LabelEncoder, y: pd.Series) -> np.ndarray:
    """Encode a label series using a pre-fit LabelEncoder."""
    return le.transform(y)


def validate_input_row(values: dict) -> list[str]:
    """
    Validate a single crop recommendation input.

    Parameters
    ----------
    values : dict with keys N, P, K, temperature, humidity, ph, rainfall

    Returns
    -------
    List of error strings (empty if all inputs are valid).
    """
    errors: list[str] = []

    ranges = {
        "N":           (0,   200, "kg/ha"),
        "P":           (0,   200, "kg/ha"),
        "K":           (0,   300, "kg/ha"),
        "temperature": (0,   50,  "°C"),
        "humidity":    (0,   100, "%"),
        "ph":          (0,   14,  ""),
        "rainfall":    (0,   500, "mm"),
    }

    for field, (lo, hi, unit) in ranges.items():
        val = values.get(field)
        if val is None:
            errors.append(f"{field}: value is required.")
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"{field}: must be a number.")
            continue
        if not (lo <= val <= hi):
            errors.append(
                f"{field}: {val} is outside the expected range [{lo}–{hi} {unit}]."
            )

    return errors
