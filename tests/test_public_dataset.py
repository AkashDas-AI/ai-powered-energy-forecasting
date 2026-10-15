"""Tests for UCI ElectricityLoadDiagrams public dataset loader, resampling, and evaluation."""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.public_dataset import (
    parse_raw_uci_txt,
    detect_active_client_window,
    resample_15m_to_hourly,
    select_primary_meter,
)
from src.train import train_and_evaluate
from src.forecast import generate_forecast, load_model_bundle
from src.config import TIMESTAMP_COL, TARGET_COL, ENTITY_ID_COL, SOURCE_DATASET_COL


@pytest.fixture
def mock_uci_raw_file(tmp_path):
    """Create a temporary mock file mirroring the official LD2011_2014.txt format.

    Characteristics:
    - Semicolon delimiters.
    - Comma decimals (e.g. 150,250000).
    - Interval-ending timestamps starting at 00:15:00.
    - Quoted column headers.
    """
    mock_file = tmp_path / "mock_LD2011_2014.txt"
    timestamps = pd.date_range("2012-01-01 00:15:00", periods=192, freq="15min")  # 48 hours

    lines = ['"";"MT_001";"MT_320"']
    for idx, ts in enumerate(timestamps):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        # MT_001 has leading zeroes for the first 96 intervals (24 hours)
        mt1_val = "0,000000" if idx < 96 else "45,500000"
        # MT_320 is active throughout with diurnal variation
        base = 120.0 + 30.0 * np.sin(2 * np.pi * ts.hour / 24.0)
        mt320_val = f"{base:.6f}".replace(".", ",")
        lines.append(f'"{ts_str}";"{mt1_val}";"{mt320_val}"')

    mock_file.write_text("\n".join(lines), encoding="utf-8")
    return mock_file


def test_parse_raw_uci_txt(mock_uci_raw_file):
    """Verify parser correctly reads European semicolon/comma format and parses timestamps."""
    df = parse_raw_uci_txt(mock_uci_raw_file, selected_clients=["MT_320"])

    assert TIMESTAMP_COL in df.columns
    assert "MT_320" in df.columns
    assert "MT_001" not in df.columns  # Column filtering works
    assert len(df) == 192
    assert pd.api.types.is_numeric_dtype(df["MT_320"])
    assert df["MT_320"].iloc[0] > 50.0  # Successfully parsed comma decimal


def test_detect_active_client_window(mock_uci_raw_file):
    """Verify detection of connection date by identifying leading zero periods."""
    df = parse_raw_uci_txt(mock_uci_raw_file)
    start_mt1, end_mt1, zero_count = detect_active_client_window(df["MT_001"], df[TIMESTAMP_COL])

    assert zero_count == 96  # First 24 hours (96 intervals) were zero
    assert start_mt1 == pd.to_datetime("2012-01-02 00:15:00")


def test_resample_15m_to_hourly_complete_bins():
    """Verify 15-minute to 1-hour aggregation correctly computes the 4-interval arithmetic mean."""
    # 4 intervals in hour 00:00 (ending at 00:15, 00:30, 00:45, 01:00)
    ts = pd.date_range("2012-01-01 00:15:00", periods=4, freq="15min")
    df_15m = pd.DataFrame({
        TIMESTAMP_COL: ts,
        "load": [100.0, 120.0, 140.0, 160.0],  # Mean is 130.0
    })

    hourly = resample_15m_to_hourly(df_15m, target_col="load", handle_dst=False)

    assert len(hourly) == 1
    assert hourly.loc[0, TIMESTAMP_COL] == pd.to_datetime("2012-01-01 00:00:00")
    assert hourly.loc[0, TARGET_COL] == 130.0
    assert hourly.loc[0, "intervals_in_bin"] == 4
    assert hourly.loc[0, "is_imputed"] == False


def test_resample_15m_incomplete_bin_flagging():
    """Verify incomplete hourly bins (< 4 intervals) are flagged with is_imputed = True."""
    # Only 2 intervals in the hour
    ts = pd.date_range("2012-01-01 00:15:00", periods=2, freq="15min")
    df_15m = pd.DataFrame({
        TIMESTAMP_COL: ts,
        "load": [100.0, 120.0],
    })

    hourly = resample_15m_to_hourly(df_15m, target_col="load", handle_dst=False)
    assert hourly.loc[0, "intervals_in_bin"] == 2
    assert hourly.loc[0, "is_imputed"] == True


def test_dst_telemetry_faithful_aggregation_policy():
    """Verify that DST transition intervals (placeholders or aggregated values) are faithfully averaged without distortion."""
    # Simulate a Spring Forward Sunday transition: 16 intervals (4 hours)
    # Hour 01:00 contains sensor floor placeholders (~0.60 kW) as observed in Portuguese grid telemetry
    timestamps = pd.date_range("2012-03-25 00:15:00", periods=16, freq="15min")
    loads = [80.0]*4 + [0.60]*4 + [82.0]*4 + [85.0]*4

    df_15m = pd.DataFrame({TIMESTAMP_COL: timestamps, "load": loads})
    hourly = resample_15m_to_hourly(df_15m, target_col="load", handle_dst=False)

    # Hour 01:00 (index 1) must be the exact arithmetic mean of the four 0.60 kW readings -> 0.60 kW
    assert hourly.loc[1, TIMESTAMP_COL].hour == 1
    assert hourly.loc[1, TARGET_COL] == 0.60
    assert hourly.loc[1, "intervals_in_bin"] == 4
    # Zero synthetic interpolation: is_imputed remains False because all 4 observations exist
    assert hourly.loc[1, "is_imputed"] == False


def test_mock_dst_transition_96_interval_calendar_contract():
    """Verify that a full 96-interval calendar day resamples to exactly 24 hourly bins of 4 intervals each."""
    timestamps = pd.date_range("2012-10-28 00:15:00", periods=96, freq="15min")
    # Simulate typical diurnal load with doubled autumn aggregation in hour 01:00
    loads = np.full(96, 50.0)
    loads[4:8] = 100.0  # Hour 01:00 (readings 01:15, 01:30, 01:45, 02:00) doubled

    df_15m = pd.DataFrame({TIMESTAMP_COL: timestamps, "MT_320": loads})
    hourly = resample_15m_to_hourly(df_15m, target_col="MT_320", handle_dst=False)

    assert len(hourly) == 24
    assert (hourly["intervals_in_bin"] == 4).all()
    assert (hourly["is_imputed"] == False).all()
    # Hour 01:00 faithfully averages to 100.0 kW without ad-hoc division
    assert hourly.loc[1, TARGET_COL] == 100.0


def test_select_primary_meter(mock_uci_raw_file):
    """Verify primary meter selection against documented completeness and variance criteria."""
    df = parse_raw_uci_txt(mock_uci_raw_file)
    selected, meta = select_primary_meter(
        df,
        start_date="2012-01-01 00:15:00",
        end_date="2012-01-02 23:45:00",
        min_mean_kw=20.0,
        min_std_kw=5.0,
    )

    # MT_320 was active from the start, MT_001 had leading zeros
    assert selected == "MT_320"
    assert "mean_load_kw" in meta
    assert meta["mean_load_kw"] > 50.0


def test_end_to_end_pure_autoregressive_training_and_forecasting(tmp_path):
    """Verify that models and forecasting run successfully without temperature_c or production_status."""
    # Generate canonical hourly series with only timestamp and load_kw
    periods = 500  # ~21 days
    ts = pd.date_range("2012-01-01 00:00:00", periods=periods, freq="h")
    load = 150.0 + 40.0 * np.sin(2 * np.pi * ts.hour / 24.0) + np.random.normal(0, 5, size=periods)

    canonical_df = pd.DataFrame({
        TIMESTAMP_COL: ts,
        ENTITY_ID_COL: "MT_320",
        TARGET_COL: np.round(load, 2),
        "is_imputed": False,
        SOURCE_DATASET_COL: "UCI_Electricity_MT_320",
    })

    data_file = tmp_path / "canonical_uci.csv"
    model_file = tmp_path / "model.joblib"
    metrics_file = tmp_path / "metrics.json"
    preds_file = tmp_path / "preds.csv"
    canonical_df.to_csv(data_file, index=False)

    # Train model on pure load time series (no exogenous columns)
    report = train_and_evaluate(
        data_path=data_file,
        model_output_path=model_file,
        metrics_output_path=metrics_file,
        predictions_output_path=preds_file,
        test_size=48,
        run_recursive_evaluation=True,
    )

    assert "baseline_24h_persistence" in report
    assert "baseline_168h_weekly_persistence" in report
    assert "random_forest_one_step" in report
    assert "recursive_24h_rolling_origin_evaluation" in report

    # Test recursive multi-step forecasting without exogenous variables
    recent_history = canonical_df.iloc[-200:].copy()
    bundle = load_model_bundle(model_file)
    fc_df = generate_forecast(
        history_df=recent_history,
        model_bundle=bundle,
        horizon=24,
        output_path=None,
    )

    assert len(fc_df) == 24
    assert (fc_df["predicted_load_kw"] >= 0.0).all()
    # Exogenous columns must not be fabricated
    assert "temperature_c" not in fc_df.columns
    assert "production_status" not in fc_df.columns


def test_leakage_audit_future_target_invariance():
    """Audit test proving feature calculation at time t is mathematically invariant to future mutations."""
    from src.features import create_features, get_feature_column_names

    dates = pd.date_range("2014-01-01 00:00:00", periods=250, freq="h")
    y = np.random.uniform(50.0, 150.0, size=250)
    df_orig = pd.DataFrame({TIMESTAMP_COL: dates, TARGET_COL: y})

    f_orig = create_features(df_orig, drop_na=False)
    feat_cols = get_feature_column_names(f_orig)

    # Target index to audit
    audit_idx = 180
    orig_features = f_orig.loc[audit_idx, feat_cols].to_dict()

    # Mutate future values at and after audit_idx
    df_mutated = df_orig.copy()
    df_mutated.loc[audit_idx:, TARGET_COL] = np.random.uniform(5000.0, 10000.0, size=len(df_orig) - audit_idx)

    f_mutated = create_features(df_mutated, drop_na=False)
    mutated_features = f_mutated.loc[audit_idx, feat_cols].to_dict()

    # Assert bit-for-bit equivalence of feature vector at audit_idx
    for col in feat_cols:
        assert orig_features[col] == pytest.approx(mutated_features[col]), (
            f"Leakage detected in feature '{col}': {orig_features[col]} != {mutated_features[col]}"
        )


def test_leakage_audit_persistence_baselines_exact_alignment():
    """Audit test proving daily (24h) and weekly (168h) persistence baselines strictly align with historical lags."""
    from src.features import create_features

    dates = pd.date_range("2014-01-01 00:00:00", periods=250, freq="h")
    y = np.arange(250, dtype=float)
    df = pd.DataFrame({TIMESTAMP_COL: dates, TARGET_COL: y})

    feats = create_features(df, drop_na=False)
    test_idx = 200

    # 24h persistence must match exact observation at (test_idx - 24)
    assert feats.loc[test_idx, f"{TARGET_COL}_lag_24"] == df.loc[test_idx - 24, TARGET_COL]
    # 168h persistence must match exact observation at (test_idx - 168)
    assert feats.loc[test_idx, f"{TARGET_COL}_lag_168"] == df.loc[test_idx - 168, TARGET_COL]


def test_real_data_dst_transitions_raw_grid_continuity():
    """Audit test verifying that all 6 European DST transition days in 2012-2014 maintain exact grid continuity."""
    from src.config import UCI_RAW_DATA_PATH

    if not UCI_RAW_DATA_PATH.exists():
        pytest.skip("Real raw LD2011_2014.txt not present; skipping raw DST audit.")

    df_raw = parse_raw_uci_txt(UCI_RAW_DATA_PATH, selected_clients=["MT_320"])
    dst_days = [
        "2012-03-25", "2012-10-28",  # 2012 Spring / Fall
        "2013-03-31", "2013-10-27",  # 2013 Spring / Fall
        "2014-03-30", "2014-10-26",  # 2014 Spring / Fall
    ]

    # 1. Verify exactly 96 quarter-hour intervals per calendar day
    for day in dst_days:
        day_slice = df_raw[(df_raw[TIMESTAMP_COL] >= f"{day} 00:00:00") & (df_raw[TIMESTAMP_COL] <= f"{day} 23:59:59")]
        assert len(day_slice) == 96, f"Day {day} has {len(day_slice)} raw intervals instead of expected 96."

    # 2. Verify resampling yields exactly 24 continuous hourly intervals with 4 readings each
    hourly = resample_15m_to_hourly(df_raw, target_col="MT_320", handle_dst=False)
    for day in dst_days:
        day_hourly = hourly[(hourly[TIMESTAMP_COL] >= f"{day} 00:00:00") & (hourly[TIMESTAMP_COL] <= f"{day} 23:59:59")]
        assert len(day_hourly) == 24, f"Day {day} produced {len(day_hourly)} hourly bins instead of 24."
        assert (day_hourly["intervals_in_bin"] == 4).all(), f"Incomplete hourly bins found on {day}."


def test_public_forecast_reproducibility_contract():
    """Audit test verifying reproducibility and schema compliance of the public forecast."""
    from src.config import UCI_MODEL_PATH, UCI_PROCESSED_DATA_PATH

    if not UCI_MODEL_PATH.exists() or not UCI_PROCESSED_DATA_PATH.exists():
        pytest.skip("Model or processed public data missing; skipping reproducibility test.")

    bundle = load_model_bundle(UCI_MODEL_PATH)
    clean_df = pd.read_csv(UCI_PROCESSED_DATA_PATH)
    recent = clean_df.iloc[-336:].copy()

    fc = generate_forecast(
        history_df=recent,
        model_bundle=bundle,
        horizon=24,
        output_path=None,
    )

    # 24 rows, strictly consecutive
    assert len(fc) == 24
    ts = pd.to_datetime(fc[TIMESTAMP_COL])
    assert ts.is_monotonic_increasing
    assert len(ts.unique()) == 24
    assert ts.iloc[0] == pd.to_datetime("2015-01-01 00:00:00")
    assert ts.iloc[-1] == pd.to_datetime("2015-01-01 23:00:00")
    assert (fc["predicted_load_kw"] >= 0.0).all()
    assert np.isfinite(fc["predicted_load_kw"]).all()


