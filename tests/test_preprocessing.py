"""
test_preprocessing.py — Tests for input validation and preprocessing utilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import validate_input_row, build_scaler, apply_scaler


class TestValidateInputRow:

    def _valid_input(self):
        return {
            "N": 50.0, "P": 40.0, "K": 40.0,
            "temperature": 25.0, "humidity": 70.0,
            "ph": 6.5, "rainfall": 100.0,
        }

    def test_valid_input_returns_empty_errors(self):
        errors = validate_input_row(self._valid_input())
        assert errors == []

    def test_missing_field_returns_error(self):
        inp = self._valid_input()
        del inp["N"]
        errors = validate_input_row(inp)
        assert any("N" in e for e in errors)

    def test_negative_n_returns_error(self):
        inp = self._valid_input()
        inp["N"] = -10.0
        errors = validate_input_row(inp)
        assert any("N" in e for e in errors)

    def test_ph_above_14_returns_error(self):
        inp = self._valid_input()
        inp["ph"] = 15.0
        errors = validate_input_row(inp)
        assert any("ph" in e for e in errors)

    def test_humidity_above_100_returns_error(self):
        inp = self._valid_input()
        inp["humidity"] = 110.0
        errors = validate_input_row(inp)
        assert any("humidity" in e for e in errors)

    def test_temperature_zero_is_valid(self):
        inp = self._valid_input()
        inp["temperature"] = 0.0
        errors = validate_input_row(inp)
        assert errors == []

    def test_all_zeros_has_no_range_errors_for_valid_zeros(self):
        # N=0, P=0, K=0, ph=0 are all at boundary — should be valid
        inp = {
            "N": 0.0, "P": 0.0, "K": 0.0,
            "temperature": 0.0, "humidity": 0.0,
            "ph": 0.0, "rainfall": 0.0,
        }
        errors = validate_input_row(inp)
        assert errors == []

    def test_non_numeric_value_returns_error(self):
        inp = self._valid_input()
        inp["temperature"] = "hot"
        errors = validate_input_row(inp)
        assert any("temperature" in e for e in errors)

    def test_multiple_invalid_fields_returns_multiple_errors(self):
        inp = self._valid_input()
        inp["N"] = -1
        inp["ph"] = 20
        errors = validate_input_row(inp)
        assert len(errors) >= 2


class TestScaler:

    def test_scaler_transforms_correctly(self):
        X_train = pd.DataFrame({
            "N": [10.0, 50.0, 90.0],
            "P": [5.0, 25.0, 45.0],
        })
        scaler = build_scaler(X_train)
        X_scaled = apply_scaler(scaler, X_train)
        # After scaling, mean should be near 0
        assert np.abs(X_scaled.mean(axis=0)).max() < 1e-10

    def test_scaler_shape_preserved(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = build_scaler(X)
        X_scaled = apply_scaler(scaler, X)
        assert X_scaled.shape == X.shape
