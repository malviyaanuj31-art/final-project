"""
deep_learning_demo.py — Optional deep learning demonstration for AgriSmart AI.

Uses TensorFlow/Keras if available; degrades gracefully if not installed.
Trains a small neural network on the crop classification dataset.

OPTIONAL DEPENDENCY: pip install tensorflow
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data_loader import load_crop_data, get_crop_features, get_crop_labels

RANDOM_SEED = 42


def is_tf_available() -> bool:
    """Return True if TensorFlow can be imported."""
    try:
        importlib.import_module("tensorflow")
        return True
    except ImportError:
        return False


def train_neural_network(epochs: int = 30) -> dict[str, Any]:
    """
    Train a small neural network on the crop dataset.

    Returns
    -------
    dict with training history, test accuracy, and model summary text.
    Raises ImportError if TensorFlow is not available.
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = load_crop_data()
    X  = get_crop_features(df).values
    y_raw = get_crop_labels(df)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Small network: 3 dense layers
    model = keras.Sequential([
        keras.layers.Dense(64,  activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32,  activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.15,
        verbose=0,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Capture summary as string
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))

    return {
        "history_train_acc":  history.history["accuracy"],
        "history_val_acc":    history.history["val_accuracy"],
        "history_train_loss": history.history["loss"],
        "history_val_loss":   history.history["val_loss"],
        "test_accuracy":      round(float(test_acc), 4),
        "test_loss":          round(float(test_loss), 4),
        "n_classes":          n_classes,
        "epochs":             epochs,
        "model_summary":      "\n".join(summary_lines),
        "classes":            list(le.classes_),
    }
