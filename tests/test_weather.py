"""Unit tests for the weather data ingestion and weather-aware forecasting experiment module.

Validates:
1. Weather data parsing, schema integrity, and absence of NaNs.
2. Exact temporal alignment between UCI electricity load and Open-Meteo ERA5 weather proxy.
3. Feature engineering verification: exactly 26 features, zero target or weather leakage.
4. Mathematical verification of diurnal persistence fallback in operational multi-step forecasting.
5. Invariance and preservation of Baseline v1 artifacts (no overwriting or corruption).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import (
    WEATHER_RAW_CACHE_PATH,
    UCI_PROCESSED_DATA_PATH,
    UCI_WEATHER_PROCESSED_DATA_PATH,
    UCI_MODEL_PATH,
    UCI_METRICS_PATH,
    UCI_WEATHER_MODEL_PATH,
    UCI_WEATHER_METRICS_PATH,
    UCI_PREDICTIONS_PATH,
    UCI_WEATHER_PREDICTIONS_PATH,
    UCI_WEATHER_FORECAST_PATH,
    TIMESTAMP_COL,
    TARGET_COL,
    TEMPERATURE_COL,
    MIN_HISTORY_REQUIRED,
    FORECAST_HORIZON,
)
from src.features import create_features, get_feature_column_names
from src.forecast import generate_forecast, load_model_bundle
from src.weather import (
    load_weather_dataframe,
    align_weather_with_electricity,
    evaluate_weather_recursive_rolling_origins,
)


def test_load_weather_dataframe_structure():
    """Verify that cached weather data parses cleanly into canonical schema without NaNs."""
    if not WEATHER_RAW_CACHE_PATH.exists():
        pytest.skip("Weather raw cache file not present on disk.")

    df_weather = load_weather_dataframe(WEATHER_RAW_CACHE_PATH)
    assert isinstance(df_weather, pd.DataFrame)
    assert TIMESTAMP_COL in df_weather.columns
    assert TEMPERATURE_COL in df_weather.columns
    assert len(df_weather) >= 26304
    assert not df_weather[TEMPERATURE_COL].isna().any()

    # Monotonic hourly continuity
    diffs = df_weather[TIMESTAMP_COL].diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()


def test_weather_electricity_alignment():
    """Verify that joining electricity with weather preserves all 26,304 rows with zero nulls."""
    if not UCI_PROCESSED_DATA_PATH.exists() or not WEATHER_RAW_CACHE_PATH.exists():
        pytest.skip("Required processed electricity or weather cache file not present.")

    aligned_df = align_weather_with_electricity(
        electricity_path=UCI_PROCESSED_DATA_PATH,
        weather_path=WEATHER_RAW_CACHE_PATH,
        output_path=None,
    )
    assert len(aligned_df) == 26304
    assert not aligned_df[TEMPERATURE_COL].isna().any()
    assert not aligned_df[TARGET_COL].isna().any()
    assert aligned_df[TEMPERATURE_COL].min() >= -10.0
    assert aligned_df[TEMPERATURE_COL].max() <= 50.0


def test_weather_feature_generation_and_leakage_invariance():
    """Verify feature matrix generation and prove that future temperatures do not leak into history."""
    # Synthesize clean 200 hours of data with temperature
    dates = pd.date_range("2014-01-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame({
        TIMESTAMP_COL: dates,
        TARGET_COL: [50.0 + i * 0.1 for i in range(200)],
        TEMPERATURE_COL: [15.0 + (i % 24) * 0.5 for i in range(200)],
    })

    feats_orig = create_features(df, drop_na=False)
    feature_cols = get_feature_column_names(feats_orig)

    # Must contain 26 features (24 baseline + 2 weather)
    assert len(feature_cols) == 26
    assert TEMPERATURE_COL in feature_cols
    assert f"{TEMPERATURE_COL}_lag_1" in feature_cols

    # Perturb future temperature at t=199
    df_perturbed = df.copy()
    df_perturbed.loc[199, TEMPERATURE_COL] = 999.0

    feats_perturbed = create_features(df_perturbed, drop_na=False)

    # Row t=198 feature values MUST be strictly identical (zero lookahead leakage)
    row_orig_198 = feats_orig.iloc[198][feature_cols].values
    row_perturbed_198 = feats_perturbed.iloc[198][feature_cols].values
    np.testing.assert_array_equal(row_orig_198, row_perturbed_198)


def test_weather_forecast_operational_fallback():
    """Verify that generate_forecast uses diurnal persistence fallback when future weather is omitted."""
    dates = pd.date_range("2014-01-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame({
        TIMESTAMP_COL: dates,
        TARGET_COL: [60.0] * 200,
        TEMPERATURE_COL: [10.0 + (i % 24) for i in range(200)],
    })

    # Mock bundle requiring weather features
    class MockEstimator:
        def predict(self, X):
            return np.full(len(X), 55.0)

    bundle = {
        "model": MockEstimator(),
        "feature_names": get_feature_column_names(create_features(df, drop_na=True)),
        "target_col": TARGET_COL,
        "timestamp_col": TIMESTAMP_COL,
        "lag_hours": [1, 2, 24, 48, 168],
        "rolling_windows": [6, 12, 24, 168],
    }

    # Forecast without supplying future_exogenous_df
    fc = generate_forecast(df, model_bundle=bundle, horizon=24, output_path=None)
    assert len(fc) == 24
    assert TEMPERATURE_COL in fc.columns

    # Verify diurnal persistence: future step h temperature must equal history at t - 24 + h
    expected_temps = df[TEMPERATURE_COL].iloc[-24:].values
    actual_fc_temps = fc[TEMPERATURE_COL].values
    np.testing.assert_array_almost_equal(actual_fc_temps, expected_temps)


def test_baseline_preservation_contract():
    """Verify that Baseline v1 artifacts are preserved and distinct from Model B artifacts."""
    if not UCI_METRICS_PATH.exists():
        pytest.skip("Baseline v1 metrics not present.")

    with open(UCI_METRICS_PATH, "r", encoding="utf-8") as f:
        base_metrics = json.load(f)

    # Baseline v1 must have exactly 24 features
    assert base_metrics["training_metadata"]["feature_count"] == 24
    assert TEMPERATURE_COL not in base_metrics["training_metadata"]["feature_columns"]

    # Model B metrics must exist separately and have 26 features
    if UCI_WEATHER_METRICS_PATH.exists():
        with open(UCI_WEATHER_METRICS_PATH, "r", encoding="utf-8") as f:
            weather_metrics = json.load(f)
        assert weather_metrics["training_metadata"]["feature_count"] == 26
        assert TEMPERATURE_COL in weather_metrics["training_metadata"]["feature_columns"]
        assert f"{TEMPERATURE_COL}_lag_1" in weather_metrics["training_metadata"]["feature_columns"]

        # Confirm comparative metrics report exists
        assert "baseline_v1_vs_model_b_comparison" in weather_metrics
        assert "weather_features_total_importance_pct" in weather_metrics


def test_weather_next_24h_forecast_integrity():
    """Verify the 24h forecast artifact for 2015-01-01 generated by Model B."""
    if not UCI_WEATHER_FORECAST_PATH.exists():
        pytest.skip("Weather forecast artifact not present.")

    fc_df = pd.read_csv(UCI_WEATHER_FORECAST_PATH)
    assert len(fc_df) == 24
    assert "timestamp" in fc_df.columns
    assert "predicted_load_kw" in fc_df.columns
    assert "predicted_energy_kwh" in fc_df.columns
    assert "temperature_c" in fc_df.columns

    # Non-negative and finite
    assert (fc_df["predicted_load_kw"] >= 0.0).all()
    assert not fc_df["predicted_load_kw"].isna().any()

    # Monotonic hourly steps
    ts = pd.to_datetime(fc_df["timestamp"])
    assert ts.min() == pd.to_datetime("2015-01-01 00:00:00")
    assert ts.max() == pd.to_datetime("2015-01-01 23:00:00")


def test_operational_weather_leakage_rejection():
    """Verify that future realized temperatures never leak into operational predictions."""
    dates = pd.date_range("2014-06-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame({
        TIMESTAMP_COL: dates,
        TARGET_COL: [50.0 + (i % 24) for i in range(200)],
        TEMPERATURE_COL: [15.0 + (i % 24) * 0.4 for i in range(200)],
    })

    class DummyEstimator:
        def predict(self, X):
            return np.array([X["load_kw_lag_1"].iloc[0] + 0.1])

    bundle = {
        "model": DummyEstimator(),
        "feature_names": get_feature_column_names(create_features(df, drop_na=True)),
        "target_col": TARGET_COL,
        "timestamp_col": TIMESTAMP_COL,
        "lag_hours": [1, 2, 24, 48, 168],
        "rolling_windows": [6, 12, 24, 168],
    }

    # Operational forecast (future_exogenous_df is None)
    fc_standard = generate_forecast(df, model_bundle=bundle, horizon=24, future_exogenous_df=None, output_path=None)

    # In operational mode, future temperatures MUST come from 24h persistence
    expected_diurnal_temp = df[TEMPERATURE_COL].iloc[-24:].values
    np.testing.assert_array_equal(fc_standard[TEMPERATURE_COL].values, expected_diurnal_temp)


def test_recursive_future_load_leakage_rejection():
    """Verify that future actual loads never enter recursive feature computation."""
    dates = pd.date_range("2014-06-01 00:00:00", periods=250, freq="h")
    df = pd.DataFrame({
        TIMESTAMP_COL: dates,
        TARGET_COL: [50.0 + (i % 24) for i in range(250)],
        TEMPERATURE_COL: [20.0 for _ in range(250)],
    })

    origin_idx = 200
    history_slice = df.iloc[: origin_idx + 1].copy()

    # Create dummy forecaster that predicts a constant
    class ConstantEstimator:
        def predict(self, X):
            return np.array([77.0] * len(X))

    bundle = {
        "model": ConstantEstimator(),
        "feature_names": get_feature_column_names(create_features(df, drop_na=True)),
        "target_col": TARGET_COL,
        "timestamp_col": TIMESTAMP_COL,
        "lag_hours": [1, 2, 24, 48, 168],
        "rolling_windows": [6, 12, 24, 168],
    }

    # Forecast with true history
    fc1 = generate_forecast(history_slice, model_bundle=bundle, horizon=24, output_path=None)

    # Modify future actual loads (rows 201 to 224)
    df_modified_future = df.copy()
    df_modified_future.loc[201:224, TARGET_COL] = 9999.0

    history_slice_2 = df_modified_future.iloc[: origin_idx + 1].copy()
    fc2 = generate_forecast(history_slice_2, model_bundle=bundle, horizon=24, output_path=None)

    # Forecast MUST be strictly identical because future actuals cannot enter history_slice
    np.testing.assert_array_equal(fc1["predicted_load_kw"].values, fc2["predicted_load_kw"].values)


def test_weather_load_timestamp_misalignment_rejection():
    """Verify that align_weather_with_electricity rejects misaligned timestamps or missing rows."""
    dates = pd.date_range("2014-01-01 00:00:00", periods=50, freq="h")
    df_elec = pd.DataFrame({
        TIMESTAMP_COL: dates,
        TARGET_COL: [50.0] * 50,
        "entity_id": ["MT_320"] * 50,
        "is_imputed": [False] * 50,
        "source_dataset": ["UCI"] * 50,
    })

    # Weather with shifted timestamps (missing 5 hours)
    weather_dates = pd.date_range("2014-01-01 05:00:00", periods=50, freq="h")
    df_weather_bad = pd.DataFrame({
        TIMESTAMP_COL: weather_dates,
        TEMPERATURE_COL: [15.0] * 50,
    })

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        elec_path = Path(tmpdir) / "elec.csv"
        weather_path = Path(tmpdir) / "weather.json"
        df_elec.to_csv(elec_path, index=False)

        # Write bad weather json
        bad_json = {
            "hourly": {
                "time": [ts.strftime("%Y-%m-%dT%H:%M") for ts in weather_dates],
                "temperature_2m": [15.0] * 50,
            }
        }
        with open(weather_path, "w") as f:
            json.dump(bad_json, f)

        with pytest.raises(ValueError, match="unmatched timestamp rows"):
            align_weather_with_electricity(elec_path, weather_path, output_path=None)


def test_recursive_predictions_artifact_and_metric_reproducibility():
    """Verify that recursive predictions CSV exists, has 2,880 rows, and exactly reproduces reported metrics."""
    from src.config import UCI_WEATHER_RECURSIVE_PREDICTIONS_PATH
    from src.evaluation import evaluate_predictions

    if not UCI_WEATHER_RECURSIVE_PREDICTIONS_PATH.exists() or not UCI_WEATHER_METRICS_PATH.exists():
        pytest.skip("Recursive predictions CSV or weather metrics JSON not present.")

    df_rec = pd.read_csv(UCI_WEATHER_RECURSIVE_PREDICTIONS_PATH)
    assert len(df_rec) == 2880
    expected_cols = [
        "origin_timestamp",
        "target_timestamp",
        "horizon_step",
        "actual_load_kw",
        "predicted_load_kw",
        "weather_input_mode",
    ]
    for col in expected_cols:
        assert col in df_rec.columns
        assert not df_rec[col].isna().any()

    # Split by mode
    df_retro = df_rec[df_rec["weather_input_mode"] == "retrospective_reanalysis"]
    df_oper = df_rec[df_rec["weather_input_mode"] == "operational_persistence_fallback"]
    assert len(df_retro) == 1440
    assert len(df_oper) == 1440

    # Recalculate metrics
    m_retro = evaluate_predictions(df_retro["actual_load_kw"].values, df_retro["predicted_load_kw"].values)
    m_oper = evaluate_predictions(df_oper["actual_load_kw"].values, df_oper["predicted_load_kw"].values)

    with open(UCI_WEATHER_METRICS_PATH, "r", encoding="utf-8") as f:
        json_metrics = json.load(f)

    json_retro = json_metrics["experiment_1_retrospective_weather_recursive_24h"]["aggregate_metrics"]
    json_oper = json_metrics["experiment_2_operational_persistence_weather_recursive_24h"]["aggregate_metrics"]

    assert round(m_retro["mae"], 3) == round(json_retro["mae"], 3)
    assert round(m_retro["rmse"], 3) == round(json_retro["rmse"], 3)
    assert round(m_retro["wape_pct"], 2) == round(json_retro["wape_pct"], 2)

    assert round(m_oper["mae"], 3) == round(json_oper["mae"], 3)
    assert round(m_oper["rmse"], 3) == round(json_oper["rmse"], 3)
    assert round(m_oper["wape_pct"], 2) == round(json_oper["wape_pct"], 2)


def test_operational_weather_persistence_horizon_mapping():
    """Verify exact source timestamp alignment for all 24 horizon steps under diurnal persistence.

    Mathematical contract:
    For any forecast origin T_origin and horizon step h in {1, ..., 24}:
    - Target timestamp: T_target(h) = T_origin + h hours.
    - Diurnal persistence source timestamp: T_source(h) = T_target(h) - 24 hours = T_origin - (24 - h) hours.
    - Availability guarantee: T_source(h) <= T_origin for all h in {1, ..., 24}.
    - Step h=1 maps to T_origin - 23 hours (observed 23h prior to origin).
    - Step h=23 maps to T_origin - 1 hour (observed 1h prior to origin).
    - Step h=24 maps to T_origin (the last completed interval at forecast issuance).
    - Exactly ZERO future observations beyond T_origin are consumed.
    """
    from src.forecast import _derive_default_exogenous

    origin = pd.Timestamp("2014-11-01 00:00:00")
    history_ts = pd.date_range(end=origin, periods=168, freq="h")
    df_hist = pd.DataFrame({
        TIMESTAMP_COL: history_ts,
        TARGET_COL: 100.0,
        TEMPERATURE_COL: np.arange(len(history_ts), dtype=float),
    })

    future_ts = pd.date_range(start=origin + pd.Timedelta(hours=1), periods=24, freq="h")
    exog = _derive_default_exogenous(future_ts, df_hist)

    assert len(exog) == 24
    for h in range(1, 25):
        target_ts = future_ts[h - 1]
        assert target_ts == origin + pd.Timedelta(hours=h)

        # Expected source timestamp under T_target - 24h
        expected_src_ts = target_ts - pd.Timedelta(hours=24)
        assert expected_src_ts == origin - pd.Timedelta(hours=24 - h)

        # Source index in history
        expected_src_val = df_hist.loc[df_hist[TIMESTAMP_COL] == expected_src_ts, TEMPERATURE_COL].values[0]
        actual_injected_val = exog.loc[exog[TIMESTAMP_COL] == target_ts, TEMPERATURE_COL].values[0]

        assert actual_injected_val == expected_src_val
        # Critical availability guarantee: source timestamp must never exceed origin
        assert expected_src_ts <= origin

    # Explicit boundary verification:
    # h=1: T_origin - 23h (historical)
    assert future_ts[0] - pd.Timedelta(hours=24) == origin - pd.Timedelta(hours=23)
    # h=24: exactly T_origin (last completed historical observation)
    assert future_ts[23] - pd.Timedelta(hours=24) == origin


def test_holdout_metric_reproducibility_and_csv_precision():
    """Verify that CSV-recalculated metrics match JSON values within floating-point rounding precision.

    Precision Policy:
    - Model metrics in JSON are calculated from raw unrounded float64 predictions at training time.
    - Saved holdout predictions CSV files apply np.round(..., 2) to minimize file size.
    - Recalculating metrics from 2-decimal rounded CSV text introduces a tiny delta of ~0.0001 kW,
      which is purely a representation artifact and not a discrepancy in data or logic.
    - Both Model A and Model B match JSON metrics within 0.0002 kW (0.005%).
    """
    from src.evaluation import calculate_mae, calculate_rmse, calculate_wape, calculate_r2

    if not UCI_PREDICTIONS_PATH.exists() or not UCI_METRICS_PATH.exists():
        pytest.skip("Baseline v1 predictions or metrics missing.")
    if not UCI_WEATHER_PREDICTIONS_PATH.exists() or not UCI_WEATHER_METRICS_PATH.exists():
        pytest.skip("Model B predictions or metrics missing.")

    # 1. Model A Holdout
    df_a = pd.read_csv(UCI_PREDICTIONS_PATH)
    assert len(df_a) == 1464
    with open(UCI_METRICS_PATH, "r", encoding="utf-8") as f:
        meta_a = json.load(f)["random_forest_one_step"]

    mae_a_csv = calculate_mae(df_a["actual_load_kw"], df_a["rf_one_step_pred_kw"])
    rmse_a_csv = calculate_rmse(df_a["actual_load_kw"], df_a["rf_one_step_pred_kw"])
    wape_a_csv = calculate_wape(df_a["actual_load_kw"], df_a["rf_one_step_pred_kw"])
    r2_a_csv = calculate_r2(df_a["actual_load_kw"], df_a["rf_one_step_pred_kw"])

    # Documented precision bounds due to 2-decimal rounded predictions in CSV
    assert abs(mae_a_csv - meta_a["mae"]) < 0.0002
    assert abs(rmse_a_csv - meta_a["rmse"]) < 0.0002
    assert abs(wape_a_csv - meta_a["wape"]) < 0.0001
    assert abs(r2_a_csv - meta_a["r2_score"]) < 0.0001

    # 2. Model B Holdout
    df_b = pd.read_csv(UCI_WEATHER_PREDICTIONS_PATH)
    assert len(df_b) == 1464
    with open(UCI_WEATHER_METRICS_PATH, "r", encoding="utf-8") as f:
        meta_b = json.load(f)["model_b_one_step_metrics"]

    mae_b_csv = calculate_mae(df_b["actual_load_kw"], df_b["model_b_pred_kw"])
    rmse_b_csv = calculate_rmse(df_b["actual_load_kw"], df_b["model_b_pred_kw"])
    wape_b_csv = calculate_wape(df_b["actual_load_kw"], df_b["model_b_pred_kw"])
    r2_b_csv = calculate_r2(df_b["actual_load_kw"], df_b["model_b_pred_kw"])

    assert abs(mae_b_csv - meta_b["mae"]) < 0.0002
    assert abs(rmse_b_csv - meta_b["rmse"]) < 0.0002
    assert abs(wape_b_csv - meta_b["wape"]) < 0.0001
    assert abs(r2_b_csv - meta_b["r2_score"]) < 0.0001



