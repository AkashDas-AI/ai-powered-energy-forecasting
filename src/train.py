"""Model training and evaluation module.

Trains a Seasonal Persistence Baseline (24h and 168h) and a RandomForestRegressor on
chronologically partitioned time-series data. Saves the model artifact alongside complete
feature metadata, metrics, and out-of-sample test set predictions.

EVALUATION PROTOCOLS:
1. One-Step-Ahead Out-of-Sample Evaluation:
   Evaluates one-step predictive capacity across the test partition using features derived
   strictly from historical data preceding each step.
2. Recursive 24-Hour Rolling-Origin Multi-Step Evaluation:
   Simulates operational dispatch by generating 24 future hourly steps recursively from
   rolling forecast origins without observing any actual future load values. Computes both
   aggregate multi-step metrics and step-by-step horizon errors (h=1..24).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    TARGET_COL,
    TIMESTAMP_COL,
    TEST_SIZE_HOURS,
    RF_PARAMS,
    LAG_HOURS,
    ROLLING_WINDOWS,
    ensure_directories,
)
from src.features import create_features, get_feature_column_names
from src.evaluation import evaluate_predictions, calculate_mae, calculate_rmse, calculate_wape
from src.forecast import generate_forecast


def evaluate_recursive_rolling_origins(
    df_clean: pd.DataFrame,
    model_bundle: Dict[str, Any],
    test_start: Union[str, pd.Timestamp],
    test_end: Union[str, pd.Timestamp],
    horizon: int = 24,
    origin_stride_hours: int = 24,
) -> Dict[str, Any]:
    """Execute a rigorous recursive rolling-origin multi-step evaluation on the holdout partition.

    At each origin T_origin:
    - Slices history strictly up to T_origin.
    - Generates 24 hourly steps recursively using generate_forecast().
    - Compares predicted trajectory with actuals for T_origin + 1h .. T_origin + 24h.
    - Compiles overall multi-step error and per-horizon breakdown (h=1..24).
    """
    df = df_clean.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)

    t_start = pd.to_datetime(test_start)
    t_end = pd.to_datetime(test_end)

    # Candidate origins inside the test window where a full 24h future exists
    origin_indices = df[(df[TIMESTAMP_COL] >= t_start) & (df[TIMESTAMP_COL] <= t_end - pd.Timedelta(hours=horizon))].index.tolist()
    sampled_origins = origin_indices[::origin_stride_hours]

    if not sampled_origins:
        return {
            "evaluated_origins_count": 0,
            "message": "Insufficient test duration for requested horizon.",
        }

    all_actuals = []
    all_predictions = []
    horizon_errors: Dict[int, List[float]] = {h: [] for h in range(1, horizon + 1)}

    for origin_idx in sampled_origins:
        origin_ts = df.loc[origin_idx, TIMESTAMP_COL]
        # History strictly up to origin (including origin)
        history_slice = df.iloc[: origin_idx + 1].copy()

        # Actual future values for the 24 steps
        actual_future = df.iloc[origin_idx + 1 : origin_idx + 1 + horizon][TARGET_COL].values
        if len(actual_future) < horizon:
            continue

        # Generate recursive forecast without future target consumption
        fc_df = generate_forecast(
            history_df=history_slice,
            model_bundle=model_bundle,
            horizon=horizon,
            output_path=None,
        )
        pred_future = fc_df["predicted_load_kw"].values

        all_actuals.extend(actual_future)
        all_predictions.extend(pred_future)

        for step_i in range(horizon):
            h_step = step_i + 1
            abs_err = abs(actual_future[step_i] - pred_future[step_i])
            horizon_errors[h_step].append(abs_err)

    actuals_arr = np.array(all_actuals)
    preds_arr = np.array(all_predictions)

    aggregate_metrics = evaluate_predictions(actuals_arr, preds_arr)
    horizon_mae = {f"h_{h:02d}": round(float(np.mean(horizon_errors[h])), 3) for h in range(1, horizon + 1)}

    return {
        "evaluated_origins_count": len(sampled_origins),
        "total_evaluated_forecast_points": len(actuals_arr),
        "aggregate_metrics": aggregate_metrics,
        "mae_by_horizon_step": horizon_mae,
    }


def train_and_evaluate(
    data_path: Optional[Union[str, Path]] = PROCESSED_DATA_PATH,
    model_output_path: Optional[Union[str, Path]] = MODEL_PATH,
    metrics_output_path: Optional[Union[str, Path]] = METRICS_PATH,
    predictions_output_path: Optional[Union[str, Path]] = TEST_PREDICTIONS_PATH,
    test_size: Optional[int] = None,
    split_dates: Optional[Dict[str, str]] = None,
    run_recursive_evaluation: bool = True,
    origin_stride_hours: int = 24,
) -> Dict[str, Any]:
    """Train forecasting models on chronological data and execute dual evaluation.

    Args:
        data_path: Path to preprocessed energy data CSV.
        model_output_path: Where to save the joblib model artifact.
        metrics_output_path: Where to save JSON metrics.
        predictions_output_path: Where to save test set prediction CSV.
        test_size: Number of chronological hours for test set (if date split not supplied).
        split_dates: Optional dictionary with {'train_end': ..., 'test_start': ..., 'test_end': ...}.
        run_recursive_evaluation: If True, executes rolling-origin 24h recursive evaluation.
        origin_stride_hours: Frequency in hours between rolling evaluation origins.

    Returns:
        Dictionary containing trained model summary, baselines, and evaluation reports.
    """
    ensure_directories()

    csv_file = Path(data_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {csv_file}. "
            "Please run preprocessing prior to training."
        )

    df_clean = pd.read_csv(csv_file)
    df_clean[TIMESTAMP_COL] = pd.to_datetime(df_clean[TIMESTAMP_COL])

    # 1. Feature Engineering with Warm-Up Dropna
    df_features = create_features(df_clean, target_col=TARGET_COL, timestamp_col=TIMESTAMP_COL, drop_na=True)
    feature_cols = get_feature_column_names(df_features, target_col=TARGET_COL, timestamp_col=TIMESTAMP_COL)

    # 2. Chronological Split (Strict chronological order)
    if split_dates is not None and "test_start" in split_dates:
        t_split = pd.to_datetime(split_dates["test_start"])
        train_df = df_features[df_features[TIMESTAMP_COL] < t_split].copy()
        test_df = df_features[df_features[TIMESTAMP_COL] >= t_split].copy()
        if "test_end" in split_dates:
            t_end = pd.to_datetime(split_dates["test_end"])
            test_df = test_df[test_df[TIMESTAMP_COL] <= t_end].copy()
    else:
        effective_test_size = test_size if test_size is not None else TEST_SIZE_HOURS
        if len(df_features) <= effective_test_size:
            raise ValueError(
                f"Dataset length after warm-up ({len(df_features)} records) is insufficient "
                f"for requested test size of {effective_test_size} records."
            )
        train_df = df_features.iloc[:-effective_test_size].copy()
        test_df = df_features.iloc[-effective_test_size:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    # 3. Stage 1 Diagnostic Development Run (Train strictly on pre-validation data if train_end provided)
    stage_1_info = None
    if split_dates is not None and "train_end" in split_dates and "test_start" in split_dates:
        t_train_end = pd.to_datetime(split_dates["train_end"])
        t_test_start = pd.to_datetime(split_dates["test_start"])
        train_dev_df = df_features[df_features[TIMESTAMP_COL] <= t_train_end].copy()
        val_df = df_features[
            (df_features[TIMESTAMP_COL] > t_train_end) & (df_features[TIMESTAMP_COL] < t_test_start)
        ].copy()

        if len(val_df) > 0 and len(train_dev_df) > 0:
            model_dev = RandomForestRegressor(**RF_PARAMS)
            model_dev.fit(train_dev_df[feature_cols], train_dev_df[TARGET_COL].values)
            val_pred = model_dev.predict(val_df[feature_cols])
            val_metrics = evaluate_predictions(val_df[TARGET_COL].values, val_pred)

            diag_test_pred = model_dev.predict(X_test)
            diag_test_metrics = evaluate_predictions(y_test, diag_test_pred)

            stage_1_info = {
                "description": (
                    "Stage 1 development evaluation: Model fitted on training period "
                    "and evaluated on validation period for hyperparameter confirmation."
                ),
                "training_period": {
                    "start": str(train_dev_df[TIMESTAMP_COL].min()),
                    "end": str(train_dev_df[TIMESTAMP_COL].max()),
                    "samples": len(train_dev_df),
                },
                "validation_period": {
                    "start": str(val_df[TIMESTAMP_COL].min()),
                    "end": str(val_df[TIMESTAMP_COL].max()),
                    "samples": len(val_df),
                },
                "validation_metrics": val_metrics,
                "diagnostic_unrefit_test_metrics": diag_test_metrics,
            }

    # 4. Dual Baselines: 24h Daily Persistence & 168h Weekly Persistence
    baseline_24h_pred = test_df[f"{TARGET_COL}_lag_24"].values
    baseline_24h_metrics = evaluate_predictions(y_test, baseline_24h_pred)

    has_168 = f"{TARGET_COL}_lag_168" in test_df.columns
    baseline_168h_pred = test_df[f"{TARGET_COL}_lag_168"].values if has_168 else baseline_24h_pred
    baseline_168h_metrics = evaluate_predictions(y_test, baseline_168h_pred) if has_168 else None

    # 5. Stage 2 Official Production Refit on Combined Pre-Test History
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    # 6. One-Step-Ahead Out-of-Sample Evaluation
    rf_one_step_pred = model.predict(X_test)
    rf_one_step_metrics = evaluate_predictions(y_test, rf_one_step_pred)

    # Improvement vs 24h baseline
    wape_reduction_24h = (
        round(((baseline_24h_metrics["wape"] - rf_one_step_metrics["wape"]) / baseline_24h_metrics["wape"]) * 100.0, 2)
        if baseline_24h_metrics["wape"] > 0 else 0.0
    )
    rmse_reduction_24h = (
        round(((baseline_24h_metrics["rmse"] - rf_one_step_metrics["rmse"]) / baseline_24h_metrics["rmse"]) * 100.0, 2)
        if baseline_24h_metrics["rmse"] > 0 else 0.0
    )

    model_bundle = {
        "model": model,
        "feature_names": feature_cols,
        "target_col": TARGET_COL,
        "timestamp_col": TIMESTAMP_COL,
        "lag_hours": LAG_HOURS,
        "rolling_windows": ROLLING_WINDOWS,
        "rf_params": RF_PARAMS,
        "metadata": {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
            "training_cutoff": str(train_df[TIMESTAMP_COL].max()),
            "trained_at": datetime.now().isoformat(),
        },
    }

    # 7. Recursive 24-Hour Rolling-Origin Multi-Step Evaluation
    recursive_report = None
    if run_recursive_evaluation and len(test_df) >= 48:
        test_start_ts = test_df[TIMESTAMP_COL].min()
        test_end_ts = test_df[TIMESTAMP_COL].max()
        recursive_report = evaluate_recursive_rolling_origins(
            df_clean=df_clean,
            model_bundle=model_bundle,
            test_start=test_start_ts,
            test_end=test_end_ts,
            horizon=24,
            origin_stride_hours=origin_stride_hours,
        )

    evaluation_report = {
        "evaluation_protocol": (
            "Dual-Tier Chronological Out-of-Sample Benchmark with Two-Stage Training/Validation Policy. "
            "Stage 1: Development & Validation (hyperparameter evaluation). "
            "Stage 2: Production Refit on combined pre-test history before evaluating untouched holdout."
        ),
        "two_stage_training_policy": {
            "policy_description": (
                "Stage 1 fits on pre-validation history to measure validation error. "
                "Stage 2 refits on all pre-test history (Train + Validation) prior to final evaluation "
                "on the untouched holdout test set."
            ),
            "final_training_cutoff": str(train_df[TIMESTAMP_COL].max()),
            "validation_included_in_final_fit": True,
            "validation_samples_count": stage_1_info["validation_period"]["samples"] if stage_1_info else 0,
        },
        "stage_1_development_validation": stage_1_info,
        "test_period": {
            "start": str(test_df[TIMESTAMP_COL].min()),
            "end": str(test_df[TIMESTAMP_COL].max()),
            "hours": len(test_df),
        },
        "baseline_24h_persistence": baseline_24h_metrics,
        "baseline_168h_weekly_persistence": baseline_168h_metrics,
        "random_forest_one_step": rf_one_step_metrics,
        "performance_improvement_one_step_vs_24h_base": {
            "wape_reduction_pct": wape_reduction_24h,
            "rmse_reduction_pct": rmse_reduction_24h,
        },
        "recursive_24h_rolling_origin_evaluation": recursive_report,
        "training_metadata": model_bundle["metadata"],
    }

    # 7. Persist Artifacts
    if model_output_path is not None:
        target_model = Path(model_output_path)
        target_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bundle, target_model)

    if metrics_output_path is not None:
        target_metrics = Path(metrics_output_path)
        target_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(target_metrics, "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, indent=4)

    if predictions_output_path is not None:
        target_preds = Path(predictions_output_path)
        # Note: Predictions are serialized to CSV rounded to 2 decimals for compactness and readability.
        # Evaluation metrics in JSON reports are computed from unrounded float64 arrays in memory;
        # recalculating metrics from 2-decimal rounded CSV text yields a minor ~0.0001 kW delta.
        test_results_df = pd.DataFrame({
            TIMESTAMP_COL: test_df[TIMESTAMP_COL].values,
            "actual_load_kw": y_test,
            "baseline_24h_pred_kw": np.round(baseline_24h_pred, 2),
            "rf_one_step_pred_kw": np.round(rf_one_step_pred, 2),
            "error_rf_kw": np.round(rf_one_step_pred - y_test, 2),
        })
        if has_168:
            test_results_df["baseline_168h_pred_kw"] = np.round(baseline_168h_pred, 2)
        test_results_df.to_csv(target_preds, index=False)

    return evaluation_report


if __name__ == "__main__":
    res = train_and_evaluate()
    print("Train and evaluate execution completed.")
    print("Baseline 24h:", res["baseline_24h_persistence"])
    print("Random Forest One-Step:", res["random_forest_one_step"])
    if res.get("recursive_24h_rolling_origin_evaluation"):
        print("Recursive 24h Aggregate:", res["recursive_24h_rolling_origin_evaluation"]["aggregate_metrics"])
