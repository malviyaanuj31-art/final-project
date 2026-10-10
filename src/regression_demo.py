"""
regression_demo.py — Crop yield regression demonstration for AgriSmart AI.

Dataset: synthetic crop yield dataset (rainfall, temperature, fertilizer,
         soil nitrogen, irrigation count → yield in tonnes/ha).

NOTE: This is a clearly labelled demonstration dataset, not real farm measurements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_regression_data, DataLoadError

RANDOM_SEED = 42
FEATURE_COLS = [
    "rainfall_mm",
    "temperature_C",
    "fertilizer_kg_per_ha",
    "soil_nitrogen",
    "irrigation_count",
]
TARGET_COL = "yield_tph"
FEATURE_DESCRIPTIONS = {
    "rainfall_mm":          "Annual rainfall (mm)",
    "temperature_C":        "Average temperature (°C)",
    "fertilizer_kg_per_ha": "Fertilizer applied (kg/ha)",
    "soil_nitrogen":        "Soil nitrogen level (kg/ha)",
    "irrigation_count":     "Number of irrigation cycles",
}


def train_regression_models() -> dict:
    """
    Train Linear Regression and Gradient Boosting Regressor on the yield dataset.

    Returns
    -------
    dict with models, evaluation metrics, split sizes and feature importances.
    """
    df = load_regression_data()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    # Linear Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)

    # Gradient Boosting
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
        )),
    ])
    gb_pipe.fit(X_train, y_train)
    gb_pred = gb_pipe.predict(X_test)

    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE":  round(mean_absolute_error(y_true, y_pred), 3),
            "RMSE": round(np.sqrt(mse), 3),
            "R2":   round(r2_score(y_true, y_pred), 4),
        }

    gb_importances = dict(zip(
        FEATURE_COLS,
        gb_pipe.named_steps["model"].feature_importances_.round(4)
    ))

    return {
        "lr_pipeline":     lr_pipe,
        "gb_pipeline":     gb_pipe,
        "lr_metrics":      metrics(y_test, lr_pred),
        "gb_metrics":      metrics(y_test, gb_pred),
        "y_test":          y_test.values,
        "lr_pred":         lr_pred,
        "gb_pred":         gb_pred,
        "train_size":      len(X_train),
        "test_size":       len(X_test),
        "feature_cols":    FEATURE_COLS,
        "target_col":      TARGET_COL,
        "feature_descs":   FEATURE_DESCRIPTIONS,
        "importances":     gb_importances,
    }
