"""Feature engineering module for energy consumption forecasting.

Generates temporal calendar features, autoregressive lag features, and rolling
window statistics.

CRITICAL GUARANTEE:
All autoregressive and rolling features strictly use historical data up to time (t - 1)
by applying explicit `.shift(1)` shifts. This guarantees ZERO target leakage into the
current observation at time t.

WARM-UP PERIOD:
The maximum lag is 168 hours (1 full week) and the maximum rolling window is 168 hours.
Consequently, a historical warm-up window of at least 168 hours (7 days) is required.
Observations within the initial warm-up period will naturally have NaN feature values,
which are removed during training dataset preparation.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.config import (
    TIMESTAMP_COL,
    TARGET_COL,
    TEMPERATURE_COL,
    PRODUCTION_COL,
    LAG_HOURS,
    ROLLING_WINDOWS,
    MIN_HISTORY_REQUIRED,
)


def create_calendar_features(df: pd.DataFrame, timestamp_col: str = TIMESTAMP_COL) -> pd.DataFrame:
    """Generate calendar-derived features and smooth cyclical trigonometric encodings."""
    df_out = df.copy()
    ts = pd.to_datetime(df_out[timestamp_col])

    # Discrete temporal coordinates
    df_out["hour"] = ts.dt.hour
    df_out["day_of_week"] = ts.dt.dayofweek  # 0 = Monday, 6 = Sunday
    df_out["day_of_month"] = ts.dt.day
    df_out["month"] = ts.dt.month
    df_out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical representations to preserve circular distance (e.g. 23:00 to 00:00)
    df_out["hour_sin"] = np.sin(2 * np.pi * df_out["hour"] / 24.0)
    df_out["hour_cos"] = np.cos(2 * np.pi * df_out["hour"] / 24.0)
    df_out["day_sin"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7.0)
    df_out["day_cos"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7.0)
    df_out["month_sin"] = np.sin(2 * np.pi * (df_out["month"] - 1) / 12.0)
    df_out["month_cos"] = np.cos(2 * np.pi * (df_out["month"] - 1) / 12.0)

    return df_out


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Generate autoregressive lag features for the target load."""
    df_out = df.copy()
    active_lags = lags or LAG_HOURS

    for lag in active_lags:
        # lag k means y_{t-k}
        df_out[f"{target_col}_lag_{lag}"] = df_out[target_col].shift(lag)

    # If temperature is present, create a 1-hour lag for temperature response
    if TEMPERATURE_COL in df_out.columns:
        df_out[f"{TEMPERATURE_COL}_lag_1"] = df_out[TEMPERATURE_COL].shift(1)

    return df_out


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Generate rolling window statistics over past observations strictly.

    CRITICAL: Applies .shift(1) before rolling to guarantee no lookahead leakage.
    At time t, the rolling statistics reflect observations from (t - w) to (t - 1).
    """
    df_out = df.copy()
    active_windows = windows or ROLLING_WINDOWS

    # Shifted series to prevent current observation leakage
    shifted_target = df_out[target_col].shift(1)

    for w in active_windows:
        df_out[f"{target_col}_rolling_mean_{w}h"] = shifted_target.rolling(window=w).mean()
        df_out[f"{target_col}_rolling_std_{w}h"] = shifted_target.rolling(window=w).std()

    return df_out


def create_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    timestamp_col: str = TIMESTAMP_COL,
    drop_na: bool = True,
) -> pd.DataFrame:
    """End-to-end feature pipeline applying calendar, lag, and rolling operations.

    Args:
        df: Input cleaned DataFrame.
        target_col: Target variable name (load_kw).
        timestamp_col: Timestamp column name.
        drop_na: If True, drops initial warm-up rows where lag/rolling features are NaN.

    Returns:
        DataFrame enriched with all engineered feature columns.
    """
    if len(df) < MIN_HISTORY_REQUIRED and drop_na:
        raise ValueError(
            f"Input DataFrame contains {len(df)} records, which is less than the required "
            f"warm-up window of {MIN_HISTORY_REQUIRED} hours for 168h lags and rolling features."
        )

    df_feats = create_calendar_features(df, timestamp_col=timestamp_col)
    df_feats = create_lag_features(df_feats, target_col=target_col)
    df_feats = create_rolling_features(df_feats, target_col=target_col)

    if drop_na:
        # Drop warm-up rows (e.g. first 168 hours)
        df_feats = df_feats.dropna().reset_index(drop=True)

    return df_feats


def get_feature_column_names(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    timestamp_col: str = TIMESTAMP_COL,
) -> List[str]:
    """Extract model feature column names excluding target, timestamp, and audit/metadata columns."""
    non_feature_cols = {
        target_col,
        timestamp_col,
        "is_imputed",
        "entity_id",
        "source_dataset",
        "intervals_in_bin",
    }
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    return feature_cols


if __name__ == "__main__":
    from src.data_generation import generate_synthetic_energy_data
    from src.preprocessing import preprocess_energy_data

    raw = generate_synthetic_energy_data(num_days=30, output_path=None)
    clean = preprocess_energy_data(raw, output_path=None)
    features = create_features(clean, drop_na=True)
    cols = get_feature_column_names(features)
    print(f"Generated {len(cols)} features: {cols}")
    print(f"Feature matrix shape after warm-up dropna: {features.shape}")

