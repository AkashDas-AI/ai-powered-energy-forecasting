# Validation & Verification Guide

This document defines the strict verification procedures, quality gates, and automated test protocols used to validate each implementation phase of the **AI-Powered Energy Consumption Forecasting** project.

---

## 1. Quality Gates & Acceptance Protocol

For every module or functional block implemented in subsequent responses, the following checklist must be satisfied before marking the step as complete:

| Phase | Verification Item | Target Standard |
| :--- | :--- | :--- |
| **Data Ingestion** | Schema & Frequency Verification | Correct datatypes; strictly monotonic increasing timestamp index; no unexpected missing steps. |
| **Data Cleaning** | Imputation & Range Integrity | Zero unhandled `NaN`s in training columns; $0 \le \text{consumption} \le \text{physical\_limit}$. |
| **Feature Engineering** | Zero Lookahead Leakage | Lag features at time $t$ use only records $\le t$; rolling statistics use `closed='left'`. |
| **Data Splitting** | Strict Chronological Order | $\max(t_{\text{train}}) < \min(t_{\text{val}}) < \min(t_{\text{test}})$; no shuffle splits. |
| **Model Training** | Convergence & Gradient Health | Training/validation loss decreases and stabilizes; no `NaN` or exploding gradients. |
| **Benchmark Superiority** | Baseline Beating Criterion | Learned model must achieve strictly lower RMSE/MAE than the Seasonal Naive baseline. |
| **Code Reliability** | Automated Test Coverage | All corresponding unit/integration tests in `tests/` pass with zero failures. |

---

## 2. Automated Test Execution Commands

When testing the project locally, run the following commands:

```bash
# Run the entire test suite with verbose output
pytest tests/ -v

# Run targeted unit tests for specific modules
pytest tests/test_data.py -v
pytest tests/test_features.py -v
pytest tests/test_models.py -v

# Run tests with code coverage metrics
pytest --cov=src tests/ --cov-report=term-missing
```

---

## 3. Data Leakage Verification Pattern

To mathematically verify that feature engineering contains no future leakage:
```python
def verify_no_lookahead_leakage(df_raw, df_features, target_col="consumption"):
    """
    Asserts that features generated at index t are invariant
    to perturbations in future records (t + k for k >= 1).
    """
    perturbed_raw = df_raw.copy()
    cutoff_idx = len(df_raw) // 2
    # Alter future data completely
    perturbed_raw.iloc[cutoff_idx:, perturbed_raw.columns.get_loc(target_col)] *= 100.0

    perturbed_features = generate_features(perturbed_raw)

    # Assert features prior to cutoff are identical
    pd.testing.assert_frame_equal(
        df_features.iloc[:cutoff_idx],
        perturbed_features.iloc[:cutoff_idx]
    )
```

---

## 4. Implementation & Validation Log Template

For every upcoming implementation milestone, the progress report will supply:
1. **Summary of Changes**: Concise breakdown of modules created/modified.
2. **Technical Details**: Mathematical/architectural decisions implemented.
3. **Executable Verification Commands**: Exact commands or automated tests to validate the implementation.
4. **Validation Outcome**: Confirmation of test passes, loss curves, or metric thresholds.

