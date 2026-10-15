"""Evaluation metrics module for time-series energy consumption forecasting.

Implements standard regression metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Weighted Absolute Percentage Error (WAPE)
- Coefficient of Determination (R²)

DENOMINATOR & ZERO-HANDLING POLICY:
For WAPE = sum(|y - y_hat|) / sum(|y|):
If sum(|y|) is zero or less than epsilon (1e-8):
- If the numerator sum(|y - y_hat|) is also zero, WAPE is defined as 0.0 (perfect zero forecast).
- If the numerator is non-zero, WAPE returns float('inf') to accurately indicate undefined division.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def _validate_and_align_inputs(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate that actual and predicted arrays are non-empty, matching length, and finite."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Cannot calculate metrics on empty input arrays.")

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Dimension mismatch: actual values count ({len(y_t)}) does not match "
            f"predicted values count ({len(y_p)})."
        )

    if not (np.isfinite(y_t).all() and np.isfinite(y_p).all()):
        raise ValueError("Inputs contain NaN or Infinite values. Please clean data before evaluation.")

    return y_t, y_p


def calculate_mae(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Calculate Mean Absolute Error (MAE) in physical target units (kW)."""
    y_t, y_p = _validate_and_align_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def calculate_rmse(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Calculate Root Mean Squared Error (RMSE) in physical target units (kW)."""
    y_t, y_p = _validate_and_align_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def calculate_wape(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    epsilon: float = 1e-8,
) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE).

    Formula: sum(|y_true - y_pred|) / sum(|y_true|)

    Zero-handling policy:
    If sum(|y_true|) < epsilon:
        - If sum(|errors|) < epsilon, returns 0.0
        - Otherwise returns float('inf')
    """
    y_t, y_p = _validate_and_align_inputs(y_true, y_pred)
    total_abs_error = np.sum(np.abs(y_t - y_p))
    total_actual = np.sum(np.abs(y_t))

    if total_actual < epsilon:
        return 0.0 if total_abs_error < epsilon else float("inf")

    return float(total_abs_error / total_actual)


def calculate_r2(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    epsilon: float = 1e-8,
) -> float:
    """Calculate Coefficient of Determination (R²)."""
    y_t, y_p = _validate_and_align_inputs(y_true, y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)

    if ss_tot < epsilon:
        return 1.0 if ss_res < epsilon else 0.0

    return float(1.0 - (ss_res / ss_tot))


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> Dict[str, float]:
    """Calculate comprehensive dictionary of machine-readable forecast evaluation metrics."""
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    wape = calculate_wape(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
        "wape_pct": round(wape * 100.0, 2) if np.isfinite(wape) else float("inf"),
        "r2_score": round(r2, 4),
        "sample_count": len(y_true),
    }


if __name__ == "__main__":
    actual = np.array([100.0, 150.0, 200.0, 180.0])
    pred = np.array([105.0, 140.0, 210.0, 175.0])
    metrics = evaluate_predictions(actual, pred)
    print("Example Evaluation Metrics:", metrics)

