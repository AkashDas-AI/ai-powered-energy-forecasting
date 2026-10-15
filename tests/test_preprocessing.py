"""Tests for data generation, schema validation, and preprocessing pipelines."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_synthetic_energy_data
from src.preprocessing import validate_raw_dataframe, preprocess_energy_data
from src.config import TIMESTAMP_COL, TARGET_COL, TEMPERATURE_COL, PRODUCTION_COL


def test_synthetic_data_generation_columns_and_frequency():
    """Verify generated synthetic data contains expected columns and regular hourly index."""
    df = generate_synthetic_energy_data(num_days=7, seed=42, output_path=None)

    # Check columns
    expected_cols = {TIMESTAMP_COL, TARGET_COL, TEMPERATURE_COL, PRODUCTION_COL}
    assert expected_cols.issubset(set(df.columns))

    # Check row count for 7 days
    assert len(df) == 7 * 24

    # Check hourly step continuity
    timestamps = pd.to_datetime(df[TIMESTAMP_COL])
    diffs = timestamps.diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()


def test_synthetic_data_non_negative_and_reproducible():
    """Verify synthetic load values are non-negative and reproducible with fixed seed."""
    df1 = generate_synthetic_energy_data(num_days=5, seed=123, output_path=None)
    df2 = generate_synthetic_energy_data(num_days=5, seed=123, output_path=None)

    # Non-negative check
    assert (df1[TARGET_COL] >= 0).all()

    # Exact numerical reproducibility
    pd.testing.assert_frame_equal(df1, df2)


def test_validate_raw_dataframe_missing_column():
    """Verify validation raises error when required columns are missing."""
    df = pd.DataFrame({"timestamp": ["2023-01-01 00:00:00"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_dataframe(df)


def test_preprocessing_duplicate_timestamp_rejection():
    """Verify duplicate timestamps trigger a descriptive ValueError."""
    df = pd.DataFrame({
        TIMESTAMP_COL: ["2023-01-01 00:00:00", "2023-01-01 00:00:00"],
        TARGET_COL: [100.0, 105.0],
        TEMPERATURE_COL: [15.0, 15.0],
        PRODUCTION_COL: [0, 0],
    })
    with pytest.raises(ValueError, match="duplicate timestamp"):
        preprocess_energy_data(input_source=df, output_path=None)


def test_preprocessing_negative_load_rejection():
    """Verify negative electricity loads trigger a descriptive ValueError."""
    df = pd.DataFrame({
        TIMESTAMP_COL: ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        TARGET_COL: [100.0, -25.0],
        TEMPERATURE_COL: [15.0, 14.0],
        PRODUCTION_COL: [0, 0],
    })
    with pytest.raises(ValueError, match="negative load values"):
        preprocess_energy_data(input_source=df, output_path=None)


def test_preprocessing_missing_hour_imputation():
    """Verify missing hourly interval is correctly re-indexed and flagged as imputed."""
    # Skips 01:00:00
    df = pd.DataFrame({
        TIMESTAMP_COL: ["2023-01-01 00:00:00", "2023-01-01 02:00:00"],
        TARGET_COL: [100.0, 120.0],
        TEMPERATURE_COL: [10.0, 14.0],
        PRODUCTION_COL: [0, 0],
    })

    cleaned = preprocess_energy_data(input_source=df, output_path=None, allow_imputation=True)

    # Must now have 3 rows: 00:00, 01:00, 02:00
    assert len(cleaned) == 3
    # Check linear interpolation for the missing middle step
    assert cleaned.loc[1, TARGET_COL] == 110.0
    # Audit flag must be True for imputed row
    assert cleaned.loc[1, "is_imputed"] == True
    assert cleaned.loc[0, "is_imputed"] == False


def test_preprocessing_missing_hour_without_imputation_raises():
    """Verify missing hourly interval raises error when allow_imputation is False."""
    df = pd.DataFrame({
        TIMESTAMP_COL: ["2023-01-01 00:00:00", "2023-01-01 02:00:00"],
        TARGET_COL: [100.0, 120.0],
        TEMPERATURE_COL: [10.0, 14.0],
        PRODUCTION_COL: [0, 0],
    })
    with pytest.raises(ValueError, match="Data continuity error"):
        preprocess_energy_data(input_source=df, output_path=None, allow_imputation=False)

