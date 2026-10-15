# Module Reference & API Specification

This document provides a comprehensive technical reference for all software modules, classes, functions, and interfaces in the **AI-Powered Energy Consumption Forecasting System**.

---

## 1. Package & Module Map

```
AI-Powered Energy Consumption Forecasting/
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Dynamic paths, hyperparameters, and project constants
│   ├── data_generation.py          # Physics-informed synthetic load generator
│   ├── preprocessing.py            # Validation, duplicate rejection, and imputation
│   ├── features.py                 # Calendar, lag, and shifted rolling feature pipeline
│   ├── train.py                    # Chronological splits, model training, and serialization
│   ├── evaluation.py               # MAE, RMSE, WAPE, and R2 evaluation metrics
│   ├── forecast.py                 # Autonomous 24-hour recursive multi-step forecaster
│   ├── public_dataset.py           # UCI smart meter ingestion, resampling, and benchmarking
│   └── weather.py                  # Open-Meteo ERA5 weather caching and controlled experiment
├── tests/
│   ├── __init__.py                 # Test package initializer
│   ├── test_preprocessing.py       # Ingestion, frequency, and imputation unit tests
│   ├── test_features.py            # Calendar, lag alignment, and zero-leakage tests
│   ├── test_evaluation.py          # Mathematical metric contracts and zero-handling tests
│   ├── test_forecast.py            # Recursive inference and historical continuity tests
│   ├── test_public_dataset.py      # Resampling, meter selection, and DST tests
│   └── test_weather.py             # Weather alignment, operational fallback, and precision tests
├── main.py                         # Unified command-line interface orchestrator
└── app.py                          # Streamlit interactive decision-support dashboard
```

---

## 2. Production Source Modules (`src/`)

### 2.1 Configuration Module (`src/config.py`)
Centralizes project filesystem paths, column naming standards, model hyperparameters, and global reproducibility settings.

- **Constants**:
  - `BASE_DIR`: Absolute path to project root, dynamically resolved via `pathlib.Path(__file__).resolve().parent.parent`.
  - `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`, `MODELS_DIR`, `OUTPUTS_DIR`, `FIGURES_DIR`: Standardized artifact directory paths.
  - `TIMESTAMP_COL`: `"timestamp"`.
  - `TARGET_COL`: `"load_kw"`.
  - `RANDOM_SEED`: `42`.
  - `DEFAULT_RF_PARAMS`: `{"n_estimators": 100, "max_depth": 14, "min_samples_split": 5, "random_state": 42, "n_jobs": -1}`.
  - `LAG_HOURS`: `[1, 2, 24, 48, 168]`.
  - `ROLLING_WINDOWS`: `[6, 12, 24, 168]`.

---

### 2.2 Data Generation Module (`src/data_generation.py`)
Simulates realistic commercial and industrial electricity telemetry based on physical principles.

- **Functions**:
  - `generate_synthetic_facility_load(days: int = 730, start_date: str = "2023-01-01", seed: int = 42) -> pd.DataFrame`:
    - *Logic*: Simulates continuous baseload (80 kW), weekday two-shift operations (07:00–19:00, ~220 kW), Saturday half-shifts, HVAC heating/cooling response to seasonal temperature swings ($7.5\text{ kW}/^\circ\text{C}$ for $T>21^\circ\text{C}$, $5.2\text{ kW}/^\circ\text{C}$ for $T<11^\circ\text{C}$), and Gaussian noise $\mathcal{N}(0, 7.5\text{ kW})$. Clamps total load at $\ge 20.0\text{ kW}$.
    - *Returns*: `pd.DataFrame` containing `timestamp`, `load_kw`, `temperature_c`, and `production_status`.
  - `save_synthetic_data(df: pd.DataFrame, output_path: Optional[str] = None) -> str`:
    - *Logic*: Saves the generated dataset to `data/raw/synthetic_facility_load.csv`.

---

### 2.3 Preprocessing Module (`src/preprocessing.py`)
Enforces strict schema validation, timestamp continuity, duplicate rejection, and audit-logged missing value imputation.

- **Functions**:
  - `validate_raw_dataframe(df: pd.DataFrame) -> None`:
    - *Logic*: Asserts that `timestamp` and `load_kw` exist. Checks that `timestamp` can be parsed to datetime and that `load_kw` is numeric.
    - *Exceptions*: Raises `ValueError` if required columns are missing or values cannot be converted.
  - `clean_and_reindex_timeseries(df: pd.DataFrame, allow_imputation: bool = True) -> pd.DataFrame`:
    - *Logic*:
      1. Converts `timestamp` to `pd.to_datetime` and sorts chronologically.
      2. Rejects duplicate timestamps: raises `ValueError` if duplicates exist.
      3. Rejects negative load values: raises `ValueError` if any `load_kw < 0`.
      4. Re-indexes series against an unbroken `pd.date_range(start, end, freq="1h")`.
      5. Fills missing intervals via linear interpolation if `allow_imputation=True`.
      6. Creates boolean column `is_imputed` (`True` for filled timestamps, `False` for observed).
    - *Exceptions*: Raises `ValueError` if missing timestamps exist and `allow_imputation=False`.

---

### 2.4 Feature Engineering Module (`src/features.py`)
Generates calendar coordinates, cyclical harmonics, autoregressive lags, and shifted rolling statistics with mathematical zero-leakage guarantees.

- **Functions**:
  - `create_calendar_features(df: pd.DataFrame) -> pd.DataFrame`:
    - *Calculates*: `hour`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos`.
  - `create_lag_features(df: pd.DataFrame, target_col: str = "load_kw", lags: List[int] = [1, 2, 24, 48, 168]) -> pd.DataFrame`:
    - *Logic*: Computes $y_{t-k}$ using `df[target_col].shift(k)` for each $k \in \text{lags}$.
  - `create_rolling_features(df: pd.DataFrame, target_col: str = "load_kw", windows: List[int] = [6, 12, 24, 168]) -> pd.DataFrame`:
    - *Zero-Leakage Rule*: Strictly computes `df[target_col].shift(1).rolling(w).mean()` and `std()` for each window $w$. Guarantees that at timestamp $t$, rolling features incorporate observations up to $t-1$ only.
  - `build_feature_pipeline(df: pd.DataFrame, target_col: str = "load_kw", drop_warmup: bool = True) -> pd.DataFrame`:
    - *Logic*: Sequentially executes calendar, lag, and rolling transforms. If `drop_warmup=True`, drops the initial 168 rows containing `NaN`s from the weekly lag.
  - `get_feature_column_names(df: pd.DataFrame) -> List[str]`:
    - *Logic*: Returns all engineered feature names, filtering out non-feature metadata (`timestamp`, `load_kw`, `is_imputed`).

---

### 2.5 Evaluation Module (`src/evaluation.py`)
Computes regression metrics with explicit zero-handling and contract verification.

- **Functions**:
  - `calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float`:
    - Mean Absolute Error: $\frac{1}{N}\sum |y_i - \hat{y}_i|$.
  - `calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float`:
    - Root Mean Squared Error: $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$.
  - `calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float`:
    - Weighted Absolute Percentage Error: $\frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$.
    - *Zero-Denominator Handling*: Returns `0.0` if both numerator and denominator equal 0; returns `float("inf")` if denominator is 0 with non-zero errors.
  - `calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float`:
    - Coefficient of Determination: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
  - `evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]`:
    - Returns dictionary with rounded keys: `{"mae", "rmse", "wape", "r2", "evaluated_points"}`.

---

### 2.6 Training Pipeline (`src/train.py`)
Orchestrates chronological data splitting, model fitting, metric evaluation, and bundle serialization.

- **Functions**:
  - `train_model(df_features: pd.DataFrame, test_hours: int = 720, rf_params: Optional[dict] = None) -> Tuple[RandomForestRegressor, dict, dict]`:
    - *Logic*: Chronologically splits data into training and out-of-sample test partitions (`iloc[:-test_hours]` vs `iloc[-test_hours:]`). Trains `RandomForestRegressor`. Evaluates both 24h Daily Naive persistence baseline and the fitted model.
  - `save_model_bundle(model: RandomForestRegressor, feature_cols: list, metrics: dict, output_path: str) -> None`:
    - *Bundle Contents*: Serializes a dictionary containing the fitted model object, feature column names, training timestamps, hyperparameters, and evaluation metrics using `joblib.dump()`.
  - `load_model_bundle(bundle_path: str) -> dict`:
    - *Logic*: Deserializes and returns the complete model bundle.

---

### 2.7 Recursive Forecasting Engine (`src/forecast.py`)
Implements autonomous multi-step 24-hour forecasting with dynamic lag feedback.

- **Functions**:
  - `generate_forecast(history_df: pd.DataFrame, model_path: str, horizon: int = 24, output_path: Optional[str] = None, future_temperature: Optional[float] = None, future_production: Optional[int] = None) -> pd.DataFrame`:
    - *Logic*:
      1. Validates that `history_df` contains at least 168 continuous hours.
      2. Loads the model bundle and extracts feature names.
      3. For $h = 1 \dots \text{horizon}$:
         - Generates the feature vector for timestamp $t+h$.
         - If exogenous variables are missing, applies 24-hour diurnal persistence fallback.
         - Predicts $\hat{y}_{t+h}$ using the model.
         - Appends predicted point to the running history state so that downstream lags ($y_{t+h-1}$, rolling windows) update dynamically.
      4. Exports prediction dataframe with columns: `timestamp`, `predicted_load_kw`, `predicted_energy_kwh`.

---

### 2.8 Public Dataset Pipeline (`src/public_dataset.py`)
Handles ingestion, resampling, and two-stage evaluation for the UCI Electricity benchmark.

- **Functions**:
  - `download_uci_dataset(data_dir: str) -> str`:
    - Downloads the official 249 MB ZIP archive from UCI and extracts `LD2011_2014.txt`.
  - `process_uci_meter(raw_file_path: str, meter_id: str = "MT_320", output_path: Optional[str] = None) -> pd.DataFrame`:
    - Parses 15-minute readings, aggregates four intervals into canonical 1-hour interval bins (mean), detects connection windows, and saves cleaned hourly series.
  - `train_public_baseline_model(data_path: str, output_model_path: str, output_metrics_path: str) -> dict`:
    - Implements the Two-Stage Chronological Protocol: Stage 1 dev/val (July–Oct 2014), Stage 2 production refit (Jan 2012–Oct 2014), holdout test (Nov–Dec 2014, 1,464 hours).
    - Evaluates dual-tier baselines (24h Daily Naive, 168h Weekly Naive) and Model A.
    - Evaluates 60-origin recursive 24h multi-step forecasting (1,440 predictions).
    - Serializes protected baseline artifacts and metrics JSON.
  - `forecast_public_baseline(model_path: str, history_path: str, output_forecast_path: str) -> pd.DataFrame`:
    - Generates official next-24h forecast for Portuguese meter `MT_320` starting `2015-01-01 00:00:00`.

---

### 2.9 Weather Experiment Module (`src/weather.py`)
Implements the controlled weather integration experiment using Open-Meteo ERA5 reanalysis.

- **Functions**:
  - `fetch_open_meteo_weather(start_date: str = "2012-01-01", end_date: str = "2015-01-02", cache_path: Optional[str] = None) -> pd.DataFrame`:
    - Fetches and caches hourly temperature reanalysis for Lisbon (`38.7223° N, -9.1393° W`).
  - `align_electricity_and_weather(load_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame`:
    - Inner joins load telemetry with temperature on hourly timestamp index.
  - `train_weather_model(data_path: str, output_model_path: str, output_metrics_path: str) -> dict`:
    - Fits Model B (26 features) over pre-test window (Jan 2012–Oct 2014).
    - Evaluates one-step holdout and 60-origin recursive multi-step holdout under both Retrospective ERA5 and Operational Diurnal Persistence.
    - Audits Random Forest Gini feature importances.
  - `forecast_weather_model(model_path: str, history_path: str, output_path: str) -> pd.DataFrame`:
    - Generates 24-hour day-ahead forecast for Model B.

---

### 2.10 Applications & Orchestration
- **`main.py`**:
  - Central CLI entrypoint supporting flags:
    - `--all`: Run full synthetic pipeline.
    - `--all-public`: Run full UCI pipeline (download, process, train, forecast).
    - `--all-weather`: Run full weather integration and controlled experiment pipeline.
- **`app.py`**:
  - Interactive Streamlit dashboard delivering operational analytics, KPI scorecards, historical time-series curves, and CSV export.

---

## 3. Test Suite Modules (`tests/`)

| Test File | Test Cases | Verification Focus |
| :--- | :---: | :--- |
| `tests/test_preprocessing.py` | 7 tests | Synthetic generation columns and frequency, non-negativity, missing column rejection, duplicate timestamp rejection, negative load rejection, missing hour imputation, and strict missing hour error. |
| `tests/test_features.py` | 5 tests | Calendar features output ranges, lag feature alignment, mathematical rolling feature zero-leakage proof, insufficient history error, and feature column names extractor. |
| `tests/test_evaluation.py` | 7 tests | MAE known values, RMSE known values, WAPE known values, WAPE zero-denominator handling, dimension mismatch error, NaN input error, and evaluation summary dictionary structure. |
| `tests/test_forecast.py` | 5 tests | Model bundle metadata preservation, 24-step forecast validity, insufficient history rejection, custom exogenous forecasting, and historical continuity contract. |
| `tests/test_public_dataset.py` | 11 tests | Raw text parsing, active client window detection, 15-min to hourly complete bin resampling, incomplete bin flagging, DST faithful aggregation, mock DST transition contract, meter selection, end-to-end training and forecasting, future target invariance leakage audit, persistence baselines exact alignment, real DST continuity, and forecast reproducibility contract. |
| `tests/test_weather.py` | 13 tests | Weather dataframe structure, electricity-weather alignment, weather feature generation and leakage invariance, weather forecast operational fallback, baseline preservation contract, next-24h weather forecast integrity, operational weather leakage rejection, recursive future load leakage rejection, load-weather timestamp misalignment rejection, recursive predictions artifact reproducibility, operational weather persistence horizon mapping, and holdout metric reproducibility and CSV precision. |
