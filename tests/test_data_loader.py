"""
test_data_loader.py — Tests for dataset loading and validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd

from src.data_loader import (
    load_crop_data,
    load_regression_data,
    get_crop_features,
    get_crop_labels,
    CROP_FEATURE_COLS,
    CROP_TARGET_COL,
    DataLoadError,
)


class TestCropDataLoader:

    def test_load_returns_dataframe(self):
        df = load_crop_data()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        df = load_crop_data()
        for col in CROP_FEATURE_COLS + [CROP_TARGET_COL]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_missing_values(self):
        df = load_crop_data()
        assert df.isnull().sum().sum() == 0, "Dataset contains missing values"

    def test_minimum_row_count(self):
        df = load_crop_data()
        assert len(df) >= 100, "Dataset has fewer than 100 rows"

    def test_multiple_crop_categories(self):
        df = load_crop_data()
        n_crops = df[CROP_TARGET_COL].nunique()
        assert n_crops >= 5, f"Expected at least 5 crop categories, got {n_crops}"

    def test_get_crop_features_excludes_target(self):
        df = load_crop_data()
        X = get_crop_features(df)
        assert CROP_TARGET_COL not in X.columns
        for col in CROP_FEATURE_COLS:
            assert col in X.columns

    def test_get_crop_labels_is_series(self):
        df = load_crop_data()
        y = get_crop_labels(df)
        assert isinstance(y, pd.Series)
        assert y.name == CROP_TARGET_COL

    def test_feature_columns_are_numeric(self):
        df = load_crop_data()
        for col in CROP_FEATURE_COLS:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_numeric_ranges_sensible(self):
        df = load_crop_data()
        assert df["N"].min() >= 0 and df["N"].max() <= 500
        assert df["ph"].min() >= 0 and df["ph"].max() <= 14
        assert df["humidity"].min() >= 0 and df["humidity"].max() <= 100
        assert df["temperature"].min() >= 0 and df["temperature"].max() <= 60


class TestRegressionDataLoader:

    def test_load_regression_returns_dataframe(self):
        df = load_regression_data()
        assert isinstance(df, pd.DataFrame)

    def test_regression_has_target_column(self):
        df = load_regression_data()
        assert "yield_tph" in df.columns

    def test_regression_minimum_rows(self):
        df = load_regression_data()
        assert len(df) >= 100
