"""Data validation and preprocessing module for energy consumption time series.

Provides reusable functions to:
- Verify schema integrity and required columns.
- Parse ISO/standard timestamps and enforce chronological ordering.
- Detect and handle duplicate timestamp entries.
- Validate numeric physical boundaries (e.g. non-negative load values).
- Detect and resolve missing hourly steps explicitly with audit logging.
"""

from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    REQUIRED_RAW_COLUMNS,
    TIMESTAMP_COL,
    TARGET_COL,
)


def validate_raw_dataframe(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    timestamp_col: str = TIMESTAMP_COL,
) -> None:
    """Validate that raw data contains the expected columns and non-empty records.

    Raises:
        ValueError: If columns are missing, DataFrame is empty, or datatypes are invalid.
    """
    if df.empty:
        raise ValueError("Data validation failed: The input DataFrame is completely empty.")

    required = required_cols or REQUIRED_RAW_COLUMNS
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Data validation failed: Missing required columns: {missing_cols}. "
            f"Expected columns: {required}."
        )

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(
            f"Data validation failed: Target column '{target_col}' must be numeric, "
            f"got dtype '{df[target_col].dtype}'."
        )


def preprocess_energy_data(
    input_source: Union[str, Path, pd.DataFrame] = RAW_DATA_PATH,
    output_path: Optional[Union[str, Path]] = PROCESSED_DATA_PATH,
    target_col: str = TARGET_COL,
    timestamp_col: str = TIMESTAMP_COL,
    allow_imputation: bool = True,
) -> pd.DataFrame:
    """Preprocess, validate, and clean facility energy time-series data.

    Args:
        input_source: File path to raw CSV or an existing DataFrame.
        output_path: Destination path to persist processed CSV (optional).
        target_col: Column name containing electrical load (kW).
        timestamp_col: Column name containing datetime stamps.
        allow_imputation: If True, missing hourly gaps are linearly interpolated
            and flagged in an audit column 'is_imputed'. If False, missing gaps raise ValueError.

    Returns:
        Cleaned and validated pd.DataFrame indexed chronologically.

    Raises:
        ValueError: On schema violations, duplicate timestamps, or negative loads.
    """
    # 1. Ingestion
    if isinstance(input_source, (str, Path)):
        file_path = Path(input_source)
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found at: {file_path}")
        df = pd.read_csv(file_path)
    elif isinstance(input_source, pd.DataFrame):
        df = input_source.copy()
    else:
        raise TypeError("input_source must be a file path (str, Path) or a pd.DataFrame.")

    # 2. Schema check
    validate_raw_dataframe(df, target_col=target_col, timestamp_col=timestamp_col)

    # 3. Timestamp parsing & sorting
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as exc:
        raise ValueError(f"Could not parse timestamps in column '{timestamp_col}': {exc}") from exc

    # 4. Duplicate timestamp detection
    duplicate_count = df.duplicated(subset=[timestamp_col]).sum()
    if duplicate_count > 0:
        dup_timestamps = df[df.duplicated(subset=[timestamp_col])][timestamp_col].tolist()
        raise ValueError(
            f"Data validation error: Detected {duplicate_count} duplicate timestamp(s). "
            f"Example duplicates: {dup_timestamps[:3]}"
        )

    # 5. Chronological sorting
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)

    # 6. Physical validation of target variable
    negative_loads = (df[target_col] < 0).sum()
    if negative_loads > 0:
        raise ValueError(
            f"Data validation error: Found {negative_loads} negative load values in '{target_col}'. "
            "Electricity consumption cannot be negative."
        )

    # 7. Check hourly regularity and handle missing gaps explicitly
    start_ts = df[timestamp_col].min()
    end_ts = df[timestamp_col].max()
    expected_index = pd.date_range(start=start_ts, end=end_ts, freq="h", name=timestamp_col)

    if len(df) != len(expected_index):
        missing_count = len(expected_index) - len(df)
        if not allow_imputation:
            raise ValueError(
                f"Data continuity error: Missing {missing_count} expected hourly interval(s) "
                f"between {start_ts} and {end_ts}. Imputation is disabled."
            )

        # Re-index to complete hourly timeline
        df_indexed = df.set_index(timestamp_col)
        reindexed_df = df_indexed.reindex(expected_index)

        # Explicit audit flag: True for imputed observations, False for original measurements
        reindexed_df["is_imputed"] = reindexed_df[target_col].isna()

        # Interpolate numeric target and exogenous features
        reindexed_df[target_col] = reindexed_df[target_col].interpolate(method="linear")
        if "temperature_c" in reindexed_df.columns:
            reindexed_df["temperature_c"] = reindexed_df["temperature_c"].interpolate(method="linear")
        if "production_status" in reindexed_df.columns:
            reindexed_df["production_status"] = reindexed_df["production_status"].ffill().bfill().astype(int)

        df = reindexed_df.reset_index()
    else:
        df["is_imputed"] = False

    # 8. Save processed data if destination specified
    if output_path is not None:
        target_file = Path(output_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_file, index=False)

    return df


if __name__ == "__main__":
    from src.config import ensure_directories
    ensure_directories()
    clean_df = preprocess_energy_data()
    print(f"Preprocessed dataset successfully. Records: {len(clean_df)}")
    print(f"Processed file saved to: {PROCESSED_DATA_PATH}")

