# Validation & Verification Guide

This document defines the automated verification procedures, quality gates, test execution protocols, and baseline integrity checks used across the **AI-Powered Energy Consumption Forecasting System**.

---

## 1. Quality Gates & Acceptance Protocol

For every module or functional modification, the following quality checklist must be satisfied:

| Phase | Verification Item | Target Standard |
| :--- | :--- | :--- |
| **Data Ingestion** | Schema & Frequency Verification | Correct datatypes; strictly monotonic increasing timestamp index; no unexpected missing steps. |
| **Data Cleaning** | Imputation & Range Integrity | Zero unhandled `NaN`s in training columns; $0 \le \text{consumption} \le \text{physical\_limit}$; `is_imputed` audit markers. |
| **Feature Engineering** | Zero Lookahead Leakage | Lag features at time $t$ use only records $\le t$; rolling statistics strictly use `.shift(1)`. |
| **Data Splitting** | Strict Chronological Order | Two-Stage Chronological Protocol: $\max(t_{\text{train}}) < \min(t_{\text{val}}) < \min(t_{\text{test}})$; no random shuffle splits. |
| **Model Training** | Convergence & Integrity | Model fits cleanly; feature names and training hyperparameters preserved in model bundle metadata. |
| **Benchmark Superiority** | Baseline Beating Criterion | Learned model must achieve strictly lower RMSE/MAE than both 24h Daily Naive and 168h Weekly Naive persistence baselines. |
| **Code Reliability** | Automated Test Coverage | All corresponding unit and integration tests in `tests/` pass with zero failures (48/48 passed). |

---

## 2. Automated Test Suite Structure & Execution

The test suite is organized into 6 focused test modules covering all system layers:

| Test Module | Coverage Area | Test Count |
| :--- | :--- | :---: |
| [`tests/test_preprocessing.py`](../tests/test_preprocessing.py) | Ingestion validation, monotonic frequency, duplicate timestamp rejection, negative load rejection, missing hour imputation. | 7 tests |
| [`tests/test_features.py`](../tests/test_features.py) | Calendar coordinate ranges, cyclical harmonics, lag feature alignment, mathematical zero-leakage proof, warm-up trimming. | 5 tests |
| [`tests/test_evaluation.py`](../tests/test_evaluation.py) | MAE, RMSE, WAPE calculation contracts, safe zero-denominator handling ($\sum \|y_i\| = 0$), dimension mismatch checks. | 7 tests |
| [`tests/test_forecast.py`](../tests/test_forecast.py) | Model bundle metadata preservation, 24-step forecast structure, minimum 168h warm-up validation, historical continuity contract. | 5 tests |
| [`tests/test_public_dataset.py`](../tests/test_public_dataset.py) | UCI 15-minute parsing, complete bin hourly resampling, client window detection, Portuguese DST grid preservation, meter selection. | 11 tests |
| [`tests/test_weather.py`](../tests/test_weather.py) | Open-Meteo ERA5 alignment, feature leakage invariance, operational diurnal persistence fallback, Baseline v1 preservation, CSV precision. | 13 tests |
| **Total Test Suite** | **Comprehensive End-to-End System Verification** | **48 tests** |

### Test Execution Commands
```bash
# 1. Run complete automated test suite (all 48 unit tests):
pytest -v

# 2. Run targeted test modules:
pytest tests/test_preprocessing.py -v
pytest tests/test_features.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_forecast.py -v
pytest tests/test_public_dataset.py -v
pytest tests/test_weather.py -v

# 3. Run bytecode compilation verification across all modules:
python -m compileall src tests main.py app.py
```

---

## 3. Data Leakage Verification Pattern

To mathematically prove that feature engineering contains zero future lookahead leakage, `tests/test_features.py` executes a **future perturbation invariance test**:

```python
def test_rolling_features_zero_leakage_mathematical_proof():
    """
    Asserts that features generated at index t are invariant
    to perturbations in future records (t + k for k >= 1).
    """
    clean_df = generate_synthetic_facility_load(days=30, seed=42)
    clean_df = clean_and_reindex_timeseries(clean_df)

    # 1. Compute unperturbed feature matrix
    base_features = build_feature_pipeline(clean_df, drop_warmup=False)

    # 2. Perturb future target data completely after midpoint cutoff
    perturbed_df = clean_df.copy()
    cutoff_idx = len(clean_df) // 2
    perturbed_df.iloc[cutoff_idx:, perturbed_df.columns.get_loc("load_kw")] *= 100.0

    # 3. Compute feature matrix on perturbed data
    perturbed_features = build_feature_pipeline(perturbed_df, drop_warmup=False)

    # 4. Assert that all feature values prior to cutoff are 100% bit-for-bit identical
    feature_cols = get_feature_column_names(base_features)
    pd.testing.assert_frame_equal(
        base_features.iloc[:cutoff_idx][feature_cols],
        perturbed_features.iloc[:cutoff_idx][feature_cols]
    )
```

---

## 4. Protected Baseline v1 Hash Verification

To guarantee that ongoing experimentation or weather pipeline executions never mutate the validated Baseline v1 artifacts, the four core baseline files are tracked with immutable SHA256 checksums:

```bash
python -c "
import hashlib
files = [
    'models/uci_forecaster_rf.joblib',
    'outputs/metrics/uci_model_metrics.json',
    'outputs/predictions/uci_test_set_predictions.csv',
    'outputs/predictions/uci_next_24h_forecast.csv'
]
expected = {
    'models/uci_forecaster_rf.joblib': '3eb2847dfaf581429d99859e35f837bcf79885f0421e0a5407ade7d5750ea49f',
    'outputs/metrics/uci_model_metrics.json': '586966739a343ebd87fb38a4e61312752e214d3e132c3154f5433ba5a80cc79b',
    'outputs/predictions/uci_test_set_predictions.csv': '7fad788030bd41fce501e60c597fb3b762dbb71090ce36e15887ba5872ee9a32',
    'outputs/predictions/uci_next_24h_forecast.csv': '4c39700e374951700efe109d86e731143a4d76f81c5090ed2ee0835df6079871'
}
for f in files:
    h = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    assert h == expected[f], f'Hash mismatch on {f}!'
    print(f'{f}: MATCHED')
"
```
