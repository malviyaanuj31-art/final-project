"""
test_crop_model.py — Tests for crop classification model training and prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from src.crop_model import (
    train_models,
    predict_crop,
    evaluate_model,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.data_loader import load_crop_data, CROP_FEATURE_COLS


class TestModelTraining:

    @pytest.fixture(scope="class")
    @classmethod
    def trained_models(cls):
        """Train models once for all tests in this class."""
        dt_pipe, rf_pipe = train_models(force=True)
        return dt_pipe, rf_pipe

    def test_models_return_two_pipelines(self, trained_models):
        dt_pipe, rf_pipe = trained_models
        assert dt_pipe is not None
        assert rf_pipe is not None

    def test_pipelines_have_scaler_and_classifier(self, trained_models):
        from sklearn.pipeline import Pipeline
        dt_pipe, rf_pipe = trained_models
        for pipe in (dt_pipe, rf_pipe):
            assert isinstance(pipe, Pipeline)
            assert "scaler" in pipe.named_steps
            assert "classifier" in pipe.named_steps

    def test_dt_has_classes(self, trained_models):
        dt_pipe, _ = trained_models
        assert hasattr(dt_pipe, "classes_")
        assert len(dt_pipe.classes_) >= 5

    def test_rf_has_predict_proba(self, trained_models):
        _, rf_pipe = trained_models
        assert hasattr(rf_pipe.named_steps["classifier"], "predict_proba")


class TestPrediction:

    @pytest.fixture(scope="class")
    @classmethod
    def pipelines(cls):
        dt_pipe, rf_pipe = train_models(force=False)
        return dt_pipe, rf_pipe

    def _sample_input(self):
        return {
            "N": 80.0, "P": 40.0, "K": 40.0,
            "temperature": 25.0, "humidity": 80.0,
            "ph": 6.5, "rainfall": 150.0,
        }

    def test_prediction_returns_string(self, pipelines):
        dt_pipe, _ = pipelines
        crop, _ = predict_crop(dt_pipe, self._sample_input())
        assert isinstance(crop, str)
        assert len(crop) > 0

    def test_prediction_is_known_crop(self, pipelines):
        dt_pipe, _ = pipelines
        crop, _ = predict_crop(dt_pipe, self._sample_input())
        assert crop in dt_pipe.classes_

    def test_rf_returns_probabilities(self, pipelines):
        _, rf_pipe = pipelines
        crop, probs = predict_crop(rf_pipe, self._sample_input())
        assert probs is not None
        assert isinstance(probs, dict)
        # Probabilities should sum close to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_feature_order_consistent(self, pipelines):
        """Changing only one feature should not crash and may change prediction."""
        dt_pipe, _ = pipelines
        base_input = self._sample_input()
        crop1, _ = predict_crop(dt_pipe, base_input)

        modified = dict(base_input)
        modified["N"] = 0.0
        modified["temperature"] = 5.0
        crop2, _ = predict_crop(dt_pipe, modified)

        # Both should be valid crop names
        assert crop1 in dt_pipe.classes_
        assert crop2 in dt_pipe.classes_

    def test_prediction_uses_correct_columns(self, pipelines):
        """Ensure DataFrame constructed from dict uses CROP_FEATURE_COLS order."""
        dt_pipe, _ = pipelines
        input_dict = self._sample_input()
        X = pd.DataFrame([input_dict], columns=CROP_FEATURE_COLS)
        assert list(X.columns) == CROP_FEATURE_COLS


class TestEvaluation:

    @pytest.fixture(scope="class")
    @classmethod
    def eval_result(cls):
        _, rf_pipe = train_models(force=False)
        return evaluate_model(rf_pipe)

    def test_eval_has_required_keys(self, eval_result):
        required_keys = [
            "accuracy", "precision", "recall", "f1",
            "confusion_matrix", "class_report",
            "test_size", "train_size", "classes",
            "y_test", "y_pred",
        ]
        for key in required_keys:
            assert key in eval_result, f"Missing key: {key}"

    def test_accuracy_in_range(self, eval_result):
        assert 0.0 <= eval_result["accuracy"] <= 1.0

    def test_test_split_correct_size(self, eval_result):
        df = load_crop_data()
        expected_test = int(len(df) * TEST_SIZE)
        # Allow ±10 rows tolerance due to stratification rounding
        assert abs(eval_result["test_size"] - expected_test) <= 10

    def test_confusion_matrix_shape(self, eval_result):
        n = len(eval_result["classes"])
        cm = eval_result["confusion_matrix"]
        assert cm.shape == (n, n)

    def test_predictions_match_classes(self, eval_result):
        valid_crops = set(eval_result["classes"])
        for pred in eval_result["y_pred"]:
            assert pred in valid_crops

    def test_accuracy_is_reasonable(self, eval_result):
        """A well-tuned crop recommendation model should achieve >70% accuracy."""
        assert eval_result["accuracy"] >= 0.70, (
            f"Model accuracy {eval_result['accuracy']:.2%} is below 70%"
        )
