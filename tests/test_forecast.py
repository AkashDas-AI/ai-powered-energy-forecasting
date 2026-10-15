"""Tests for inference, model serialization, and multi-step recursive forecasting."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_synthetic_energy_data
from src.preprocessing import preprocess_energy_data
from src.train import train_and_evaluate
from src.forecast import generate_forecast, load_model_bundle
from src.config import TARGET_COL, TIMESTAMP_COL, FORECAST_HORIZON, MIN_HISTORY_REQUIRED


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """Module-level fixture to train a fast model for test assertions."""
    temp_dir = tmp_path_factory.mktemp("forecasting_test_artifacts")
    raw_path = temp_dir / "raw.csv"
    clean_path = temp_dir / "clean.csv"
    model_path = temp_dir / "model.joblib"
    metrics_path = temp_dir / "metrics.json"
    preds_path = temp_dir / "preds.csv"

    # Generate 60 days of data (sufficient for 168h warm-up and 720h test)
    raw = generate_synthetic_energy_data(num_days=60, seed=42, output_path=raw_path)
    clean = preprocess_energy_data(raw, output_path=clean_path)

    train_and_evaluate(
        data_path=clean_path,
        model_output_path=model_path,
        metrics_output_path=metrics_path,
        predictions_output_path=preds_path,
        test_size=168,  # Fast 1-week test partition for testing
    )

    return {
        "clean_data": clean,
        "model_path": model_path,
        "metrics_path": metrics_path,
    }


def test_model_bundle_preserves_metadata(trained_artifacts):
    """Verify saved joblib bundle includes model estimator and exact feature metadata."""
    bundle = load_model_bundle(trained_artifacts["model_path"])

    assert "model" in bundle
    assert "feature_names" in bundle
    assert isinstance(bundle["feature_names"], list)
    assert len(bundle["feature_names"]) > 0
    assert bundle["target_col"] == TARGET_COL


def test_forecast_generates_24_valid_hourly_steps(trained_artifacts):
    """Verify forecast produces exactly 24 future hourly steps with finite positive load."""
    clean_df = trained_artifacts["clean_data"]
    recent_history = clean_df.iloc[-200:].copy()  # At least 168 hours
    last_hist_ts = recent_history[TIMESTAMP_COL].iloc[-1]

    fc_df = generate_forecast(
        history_df=recent_history,
        model_path=trained_artifacts["model_path"],
        horizon=24,
        output_path=None,
    )

    # 1. Exactly 24 rows
    assert len(fc_df) == FORECAST_HORIZON

    # 2. Columns present
    assert "timestamp" in fc_df.columns
    assert "predicted_load_kw" in fc_df.columns
    assert "predicted_energy_kwh" in fc_df.columns

    # 3. Finite, non-negative predictions
    assert np.isfinite(fc_df["predicted_load_kw"]).all()
    assert (fc_df["predicted_load_kw"] >= 0.0).all()

    # 4. Strictly future timestamps incrementing hourly
    expected_start = pd.to_datetime(last_hist_ts) + pd.Timedelta(hours=1)
    assert fc_df["timestamp"].iloc[0] == expected_start
    ts_diffs = fc_df["timestamp"].diff().dropna()
    assert (ts_diffs == pd.Timedelta(hours=1)).all()


def test_forecast_insufficient_history_rejected(trained_artifacts):
    """Verify forecast raises descriptive ValueError when given < 168 hours of history."""
    clean_df = trained_artifacts["clean_data"]
    short_history = clean_df.iloc[-50:].copy()  # Only 50 hours

    with pytest.raises(ValueError, match="Insufficient historical data"):
        generate_forecast(
            history_df=short_history,
            model_path=trained_artifacts["model_path"],
            horizon=24,
            output_path=None,
        )


def test_forecast_with_custom_exogenous(trained_artifacts):
    """Verify forecast accepts explicit future exogenous inputs."""
    clean_df = trained_artifacts["clean_data"]
    recent_history = clean_df.iloc[-200:].copy()
    last_hist_ts = pd.to_datetime(recent_history[TIMESTAMP_COL].iloc[-1])

    future_ts = pd.date_range(start=last_hist_ts + pd.Timedelta(hours=1), periods=24, freq="h")
    custom_exog = pd.DataFrame({
        TIMESTAMP_COL: future_ts,
        "temperature_c": np.ones(24) * 35.0,  # Extreme heatwave scenario
        "production_status": np.ones(24, dtype=int),  # 24-hour full production
    })

    fc_df = generate_forecast(
        history_df=recent_history,
        model_path=trained_artifacts["model_path"],
        horizon=24,
        future_exogenous_df=custom_exog,
        output_path=None,
    )

    assert len(fc_df) == 24
    assert (fc_df["temperature_c"] == 35.0).all()
    assert (fc_df["production_status"] == 1).all()


def test_forecast_historical_continuity_contract(trained_artifacts):
    """Verify that forecast horizon begins exactly one hour after the final historical timestamp,
    preventing multi-year or discontinuous axis gaps in downstream visualizations."""
    clean_df = trained_artifacts["clean_data"]
    recent_history = clean_df.iloc[-200:].copy()
    last_hist_ts = pd.to_datetime(recent_history[TIMESTAMP_COL].iloc[-1])

    fc_df = generate_forecast(
        history_df=recent_history,
        model_path=trained_artifacts["model_path"],
        horizon=24,
        output_path=None,
    )

    forecast_start = pd.to_datetime(fc_df[TIMESTAMP_COL].iloc[0])
    temporal_gap = forecast_start - last_hist_ts
    assert temporal_gap == pd.Timedelta(hours=1), (
        f"Temporal discontinuity: forecast starts at {forecast_start}, but history ends at {last_hist_ts}. "
        f"Expected exact 1-hour increment."
    )

