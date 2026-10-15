"""Inference and multi-step forecasting module.

Loads the trained model bundle and performs recursive multi-step forecasting
for the upcoming 24-hour horizon.

TARGET LEAKAGE PREVENTION:
At step h, features are computed exclusively using historical actuals and
previously predicted values from steps 1 to (h - 1). Future targets are never
observed.

EXOGENOUS ASSUMPTIONS:
If future weather and production schedules are not explicitly provided:
1. Temperature: Assumes 24-hour persistence from the previous day's diurnal cycle.
2. Production: Derived deterministically from the facility's standard operating schedule.
These assumptions are recorded explicitly in the output metadata.

UNITS & ENERGY CONVERSION:
Because each time interval is exactly one hour (1H), the average power in kilowatts (kW)
over that hour directly equals the energy consumed in kilowatt-hours (kWh):
Energy (kWh) = Power (kW) * 1.0 hour.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_PATH,
    FORECAST_PATH,
    TARGET_COL,
    TIMESTAMP_COL,
    TEMPERATURE_COL,
    PRODUCTION_COL,
    FORECAST_HORIZON,
    MIN_HISTORY_REQUIRED,
    ensure_directories,
)
from src.features import create_calendar_features, create_lag_features, create_rolling_features


def load_model_bundle(model_path: Union[str, Path] = MODEL_PATH) -> Dict[str, Any]:
    """Safely load trained model estimator and its associated feature metadata."""
    file_path = Path(model_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at: {file_path}. "
            "Please train the model first by executing 'python main.py --train'."
        )
    bundle = joblib.load(file_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_names" not in bundle:
        raise ValueError("Invalid model bundle format: Missing 'model' or 'feature_names'.")
    return bundle


def _derive_default_exogenous(
    future_timestamps: pd.DatetimeIndex,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive future exogenous inputs under documented fallback assumptions."""
    exog_df = pd.DataFrame({TIMESTAMP_COL: future_timestamps})

    # Temperature fallback: 24-hour diurnal cycle persistence from the last 24 historical hours
    if TEMPERATURE_COL in history_df.columns and len(history_df) >= 24:
        last_24h_temp = history_df[TEMPERATURE_COL].iloc[-24:].values
        repeated_temp = np.resize(last_24h_temp, len(future_timestamps))
        exog_df[TEMPERATURE_COL] = repeated_temp
    else:
        exog_df[TEMPERATURE_COL] = 20.0  # Default mild temperature

    # Production schedule fallback: Standard facility calendar rule
    # Weekday 07:00-19:00 = 1, Saturday 08:00-14:00 = 1, otherwise 0
    day_of_week = future_timestamps.dayofweek
    hour_of_day = future_timestamps.hour
    prod_status = np.zeros(len(future_timestamps), dtype=int)
    weekday_active = (day_of_week < 5) & (hour_of_day >= 7) & (hour_of_day < 19)
    saturday_active = (day_of_week == 5) & (hour_of_day >= 8) & (hour_of_day < 14)
    prod_status[weekday_active | saturday_active] = 1
    exog_df[PRODUCTION_COL] = prod_status

    return exog_df


def generate_forecast(
    history_df: pd.DataFrame,
    model_bundle: Optional[Dict[str, Any]] = None,
    model_path: Union[str, Path] = MODEL_PATH,
    horizon: int = FORECAST_HORIZON,
    future_exogenous_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = FORECAST_PATH,
) -> pd.DataFrame:
    """Generate recursive multi-step forecasts for the upcoming horizon.

    Args:
        history_df: DataFrame of recent historical observations.
            Must contain at least MIN_HISTORY_REQUIRED records (168 hours).
        model_bundle: Optional pre-loaded model bundle dictionary.
        model_path: Path to model artifact if bundle not supplied.
        horizon: Number of future hours to forecast (default 24).
        future_exogenous_df: Optional future temperature/production schedule.
        output_path: File path to save output forecast CSV.

    Returns:
        DataFrame with columns:
        ['timestamp', 'predicted_load_kw', 'predicted_energy_kwh', 'temperature_c', 'production_status']
    """
    # 1. Validate inputs
    if history_df.empty:
        raise ValueError("Cannot forecast: Provided historical data is empty.")

    if len(history_df) < MIN_HISTORY_REQUIRED:
        raise ValueError(
            f"Insufficient historical data: Provided {len(history_df)} records, but at least "
            f"{MIN_HISTORY_REQUIRED} hours (1 week) are required to initialize 168-hour lag features."
        )

    for col in [TIMESTAMP_COL, TARGET_COL]:
        if col not in history_df.columns:
            raise ValueError(f"Historical data missing mandatory column: '{col}'")

    # 2. Ensure historical timestamps are ordered
    history = history_df.copy()
    history[TIMESTAMP_COL] = pd.to_datetime(history[TIMESTAMP_COL])
    history = history.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)

    # Retain strictly the required historical tail to compute 168h lags and rolling stats
    min_needed = MIN_HISTORY_REQUIRED + horizon + 48
    if len(history) > min_needed:
        history = history.iloc[-min_needed:].copy().reset_index(drop=True)

    # 3. Load model artifact
    bundle = model_bundle or load_model_bundle(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # 4. Generate future hourly timestamp index
    last_timestamp = history[TIMESTAMP_COL].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
        name=TIMESTAMP_COL,
    )

    # 5. Check which exogenous features the model actually uses
    uses_temp = TEMPERATURE_COL in feature_names
    uses_prod = PRODUCTION_COL in feature_names

    future_exog = None
    if uses_temp or uses_prod:
        if future_exogenous_df is not None:
            future_exog = future_exogenous_df.copy()
            future_exog[TIMESTAMP_COL] = pd.to_datetime(future_exog[TIMESTAMP_COL])
            if len(future_exog) < horizon:
                raise ValueError(
                    f"future_exogenous_df contains {len(future_exog)} rows, but horizon requires {horizon}."
                )
        else:
            future_exog = _derive_default_exogenous(future_timestamps, history)

    # 6. Recursive multi-step forecasting loop
    keep_cols = [TIMESTAMP_COL, TARGET_COL]
    if TEMPERATURE_COL in history.columns:
        keep_cols.append(TEMPERATURE_COL)
    if PRODUCTION_COL in history.columns:
        keep_cols.append(PRODUCTION_COL)

    extended_df = history[keep_cols].copy()
    predictions = []

    for step_idx, step_ts in enumerate(future_timestamps):
        # Build placeholder row for current step
        temp_dict = {
            TIMESTAMP_COL: [step_ts],
            TARGET_COL: [np.nan],  # Unknown target at time t
        }
        if uses_temp and future_exog is not None:
            step_temp = float(future_exog.loc[future_exog[TIMESTAMP_COL] == step_ts, TEMPERATURE_COL].values[0])
            temp_dict[TEMPERATURE_COL] = [step_temp]
        elif TEMPERATURE_COL in extended_df.columns:
            temp_dict[TEMPERATURE_COL] = [extended_df[TEMPERATURE_COL].iloc[-1]]

        if uses_prod and future_exog is not None:
            step_prod = int(future_exog.loc[future_exog[TIMESTAMP_COL] == step_ts, PRODUCTION_COL].values[0])
            temp_dict[PRODUCTION_COL] = [step_prod]
        elif PRODUCTION_COL in extended_df.columns:
            temp_dict[PRODUCTION_COL] = [extended_df[PRODUCTION_COL].iloc[-1]]

        temp_row = pd.DataFrame(temp_dict)
        eval_df = pd.concat([extended_df, temp_row], ignore_index=True)

        # Compute calendar, lag, and rolling features
        # Note: lags and rolling stats shift(1) strictly utilize extended_df past values!
        feats_df = create_calendar_features(eval_df, timestamp_col=TIMESTAMP_COL)
        feats_df = create_lag_features(feats_df, target_col=TARGET_COL)
        feats_df = create_rolling_features(feats_df, target_col=TARGET_COL)

        # Extract feature vector for the step row (last row)
        current_features = feats_df.iloc[[-1]][feature_names]

        # Check for unpopulated features
        if current_features.isna().any().any():
            nan_cols = current_features.columns[current_features.isna().any()].tolist()
            raise ValueError(f"Feature computation error at step {step_ts}: NaNs in columns: {nan_cols}")

        # Model prediction
        pred_kw = float(model.predict(current_features)[0])
        # Ensure non-negative physical constraint
        pred_kw = max(0.0, round(pred_kw, 2))

        # Store prediction
        pred_entry = {
            TIMESTAMP_COL: step_ts,
            "predicted_load_kw": pred_kw,
            "predicted_energy_kwh": pred_kw,  # 1 hour interval => 1 kWh per 1 kW average load
        }
        if uses_temp and future_exog is not None:
            pred_entry[TEMPERATURE_COL] = step_temp
        if uses_prod and future_exog is not None:
            pred_entry[PRODUCTION_COL] = step_prod

        predictions.append(pred_entry)

        # Feed prediction back into extended history for subsequent autoregressive steps
        feed_dict = {
            TIMESTAMP_COL: [step_ts],
            TARGET_COL: [pred_kw],
        }
        if TEMPERATURE_COL in temp_dict:
            feed_dict[TEMPERATURE_COL] = temp_dict[TEMPERATURE_COL]
        if PRODUCTION_COL in temp_dict:
            feed_dict[PRODUCTION_COL] = temp_dict[PRODUCTION_COL]

        extended_df = pd.concat([extended_df, pd.DataFrame(feed_dict)], ignore_index=True)

    forecast_df = pd.DataFrame(predictions)

    # 7. Persist forecast CSV if output_path is provided
    if output_path is not None:
        target_file = Path(output_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(target_file, index=False)

    return forecast_df


if __name__ == "__main__":
    from src.preprocessing import preprocess_energy_data
    ensure_directories()
    clean_data = preprocess_energy_data()
    recent_history = clean_data.iloc[-300:].copy()
    fc = generate_forecast(recent_history)
    print("Forecast generated successfully:")
    print(fc.head())

