"""Tests for regression and time-series evaluation metrics."""

import numpy as np
import pytest

from src.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    calculate_r2,
    evaluate_predictions,
)


def test_mae_known_values():
    """Verify MAE calculation against hand-computed scalar value."""
    actual = [10.0, 20.0, 30.0]
    predicted = [12.0, 18.0, 33.0]
    # Errors: |10-12|=2, |20-18|=2, |30-33|=3 -> Mean = 7/3 = 2.333333
    expected_mae = 7.0 / 3.0
    assert pytest.approx(calculate_mae(actual, predicted), 1e-5) == expected_mae


def test_rmse_known_values():
    """Verify RMSE calculation against hand-computed scalar value."""
    actual = [2.0, 4.0]
    predicted = [4.0, 8.0]
    # Squared errors: (2-4)^2 = 4, (4-8)^2 = 16 -> Mean = 10 -> sqrt(10) = 3.162277
    expected_rmse = np.sqrt(10.0)
    assert pytest.approx(calculate_rmse(actual, predicted), 1e-5) == expected_rmse


def test_wape_known_values():
    """Verify WAPE calculation against hand-computed scalar value."""
    actual = [100.0, 200.0]
    predicted = [90.0, 220.0]
    # Sum abs errors: |100-90| + |200-220| = 10 + 20 = 30
    # Sum actual: 100 + 200 = 300
    # WAPE = 30 / 300 = 0.10 (10%)
    assert pytest.approx(calculate_wape(actual, predicted), 1e-5) == 0.10


def test_wape_zero_denominator_handling():
    """Verify documented zero-denominator handling for WAPE."""
    zero_actuals = [0.0, 0.0, 0.0]
    zero_predictions = [0.0, 0.0, 0.0]
    non_zero_predictions = [1.0, 2.0, 3.0]

    # Perfect forecast on zero demand -> WAPE is 0.0
    assert calculate_wape(zero_actuals, zero_predictions) == 0.0

    # Non-zero forecast on zero demand -> WAPE is float('inf')
    assert calculate_wape(zero_actuals, non_zero_predictions) == float("inf")


def test_dimension_mismatch_raises_error():
    """Verify mismatched input lengths raise ValueError."""
    with pytest.raises(ValueError, match="Dimension mismatch"):
        calculate_mae([1.0, 2.0], [1.0])


def test_nan_input_raises_error():
    """Verify NaN values in actuals or predictions trigger ValueError."""
    with pytest.raises(ValueError, match="contain NaN or Infinite"):
        calculate_rmse([1.0, np.nan], [1.0, 2.0])


def test_evaluate_predictions_dictionary_keys():
    """Verify evaluate_predictions returns complete machine-readable dictionary."""
    y_true = np.array([50.0, 100.0, 150.0])
    y_pred = np.array([52.0, 98.0, 153.0])
    metrics = evaluate_predictions(y_true, y_pred)

    expected_keys = {"mae", "rmse", "wape", "wape_pct", "r2_score", "sample_count"}
    assert expected_keys.issubset(set(metrics.keys()))
    assert metrics["sample_count"] == 3

