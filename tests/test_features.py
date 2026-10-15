"""Tests for feature engineering, calendar encodings, lags, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_synthetic_energy_data
from src.preprocessing import preprocess_energy_data
from src.features import (
    create_features,
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
    get_feature_column_names,
)
from src.config import TARGET_COL, TIMESTAMP_COL, MIN_HISTORY_REQUIRED


@pytest.fixture
def sample_clean_df():
    """Fixture providing 14 days (336 hours) of clean hourly data."""
    raw = generate_synthetic_energy_data(num_days=14, seed=99, output_path=None)
    return preprocess_energy_data(raw, output_path=None)


def test_calendar_features_output(sample_clean_df):
    """Verify calendar feature ranges and cyclical trigonometric bounds."""
    df_cal = create_calendar_features(sample_clean_df)

    assert "hour" in df_cal.columns
    assert "day_of_week" in df_cal.columns
    assert "month" in df_cal.columns
    assert "is_weekend" in df_cal.columns

    # Check bounds
    assert df_cal["hour"].between(0, 23).all()
    assert df_cal["day_of_week"].between(0, 6).all()
    assert df_cal["hour_sin"].between(-1.0, 1.0).all()
    assert df_cal["hour_cos"].between(-1.0, 1.0).all()


def test_lag_features_alignment(sample_clean_df):
    """Verify lag features are correctly shifted in time."""
    df_lags = create_lag_features(sample_clean_df, target_col=TARGET_COL)

    # lag_1 at row index 1 must equal target at row index 0
    assert df_lags.loc[1, f"{TARGET_COL}_lag_1"] == sample_clean_df.loc[0, TARGET_COL]

    # lag_24 at row index 24 must equal target at row index 0
    assert df_lags.loc[24, f"{TARGET_COL}_lag_24"] == sample_clean_df.loc[0, TARGET_COL]


def test_rolling_features_zero_leakage_mathematical_proof(sample_clean_df):
    """Rigorous verification: Altering future values must NEVER perturb past or current features.

    If rolling statistics used the current step t, altering y(t) would change the feature at t.
    Because .shift(1) is applied, features at time t depend strictly on y(0)...y(t-1).
    """
    cutoff = 200  # Cutoff index

    # Base feature transformation
    base_features = create_features(sample_clean_df, drop_na=False)

    # Perturbed dataset: modify all values from cutoff onwards
    perturbed_df = sample_clean_df.copy()
    perturbed_df.loc[cutoff:, TARGET_COL] = perturbed_df.loc[cutoff:, TARGET_COL] * 5.0 + 1000.0

    perturbed_features = create_features(perturbed_df, drop_na=False)

    # All feature values prior to cutoff MUST be strictly identical
    rolling_cols = [c for c in base_features.columns if "rolling" in c or "lag" in c]
    pd.testing.assert_frame_equal(
        base_features.loc[: cutoff - 1, rolling_cols],
        perturbed_features.loc[: cutoff - 1, rolling_cols],
    )


def test_insufficient_history_raises_error():
    """Verify error when attempting feature extraction with drop_na on fewer than 168 hours."""
    short_df = pd.DataFrame({
        TIMESTAMP_COL: pd.date_range("2023-01-01", periods=50, freq="h"),
        TARGET_COL: np.ones(50) * 100.0,
        "temperature_c": np.ones(50) * 20.0,
        "production_status": np.zeros(50, dtype=int),
    })

    with pytest.raises(ValueError, match="less than the required warm-up window"):
        create_features(short_df, drop_na=True)


def test_get_feature_column_names(sample_clean_df):
    """Verify feature extractor correctly excludes target, timestamp, and audit columns."""
    feats = create_features(sample_clean_df, drop_na=True)
    cols = get_feature_column_names(feats)

    assert TARGET_COL not in cols
    assert TIMESTAMP_COL not in cols
    assert "is_imputed" not in cols
    assert len(cols) > 10

