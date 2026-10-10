"""
crop_model.py — Train, save, load and evaluate the crop classification model.

Primary model: Decision Tree (beginner-friendly, interpretable).
A Random Forest is also trained to show probability estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import (
    CROP_FEATURE_COLS,
    DataLoadError,
    get_crop_features,
    get_crop_labels,
    load_crop_data,
)

MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"
DT_MODEL_PATH  = MODELS_DIR / "crop_decision_tree.joblib"
RF_MODEL_PATH  = MODELS_DIR / "crop_random_forest.joblib"

RANDOM_SEED = 42
TEST_SIZE   = 0.20


def _ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── training ───────────────────────────────────────────────────────────────────

def train_models(force: bool = False) -> tuple[Pipeline, Pipeline]:
    """
    Train Decision Tree and Random Forest classification pipelines.

    Each pipeline: StandardScaler → Classifier.
    Fits on training split only (no data leakage).

    Parameters
    ----------
    force : retrain even if saved models already exist.

    Returns
    -------
    (dt_pipeline, rf_pipeline)
    """
    _ensure_models_dir()

    if not force and DT_MODEL_PATH.exists() and RF_MODEL_PATH.exists():
        return load_models()

    df = load_crop_data()
    X  = get_crop_features(df)
    y  = get_crop_labels(df)

    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    dt_pipe = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
        )),
    ])
    dt_pipe.fit(X_train, y_train)

    rf_pipe = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
        )),
    ])
    rf_pipe.fit(X_train, y_train)

    joblib.dump(dt_pipe, DT_MODEL_PATH)
    joblib.dump(rf_pipe, RF_MODEL_PATH)

    return dt_pipe, rf_pipe


def load_models() -> tuple[Pipeline, Pipeline]:
    """Load pre-trained DT and RF pipelines from disk."""
    dt_pipe = joblib.load(DT_MODEL_PATH)
    rf_pipe = joblib.load(RF_MODEL_PATH)
    return dt_pipe, rf_pipe


def get_or_train_models() -> tuple[Pipeline, Pipeline]:
    """Return models from disk if available, otherwise train them."""
    if DT_MODEL_PATH.exists() and RF_MODEL_PATH.exists():
        return load_models()
    return train_models()


# ── prediction ─────────────────────────────────────────────────────────────────

def predict_crop(pipeline: Pipeline, input_dict: dict) -> tuple[str, dict[str, float] | None]:
    """
    Predict a crop given a dictionary of input features.

    Parameters
    ----------
    pipeline   : trained sklearn Pipeline
    input_dict : {feature_name: value, ...}

    Returns
    -------
    (crop_name, probability_dict | None)
    """
    X = pd.DataFrame([input_dict], columns=CROP_FEATURE_COLS)
    crop_name = pipeline.predict(X)[0]

    prob_dict: dict[str, float] | None = None
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        probs     = pipeline.predict_proba(X)[0]
        classes   = pipeline.classes_
        prob_dict = dict(zip(classes, probs.round(4)))

    return crop_name, prob_dict


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(pipeline: Pipeline) -> dict[str, Any]:
    """
    Evaluate a classification pipeline on the held-out test split.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                    class_report, test_size, train_size, classes,
                    y_test, y_pred
    """
    df = load_crop_data()
    X  = get_crop_features(df)
    y  = get_crop_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    y_pred = pipeline.predict(X_test)

    return {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "class_report":     classification_report(y_test, y_pred, zero_division=0),
        "test_size":        len(X_test),
        "train_size":       len(X_train),
        "classes":          list(pipeline.classes_),
        "y_test":           y_test.tolist(),
        "y_pred":           y_pred.tolist(),
    }
