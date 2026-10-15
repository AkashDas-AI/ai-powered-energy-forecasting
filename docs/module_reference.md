# Module Reference & Codebase Architecture

This document defines the modular structure, responsibilities, public interfaces, and interaction contracts of all software components in the **AI-Powered Energy Consumption Forecasting** project.

---

## 1. Directory & Package Structure

```
AI-Powered Energy Consumption Forecasting/
├── docs/                           # Central technical documentation repository
│   ├── README.md                   # Documentation index & lifecycle rules
│   ├── architecture.md             # System architecture & data flow diagrams
│   ├── dataset_spec.md             # Data schemas, validation & feature specs
│   ├── model_spec.md               # Model math, loss functions & training dynamics
│   ├── module_reference.md         # Module responsibilities & API contracts
│   └── validation_guide.md         # Testing protocols & verification checklists
├── src/                            # Production source code
│   ├── __init__.py
│   ├── config/                     # Configuration management & settings
│   │   ├── __init__.py
│   │   └── config.py               # Pydantic/dataclass-based validated configs
│   ├── data/                       # Ingestion, validation & cleaning
│   │   ├── __init__.py
│   │   ├── loader.py               # Raw file ingestion & format parsers
│   │   └── validator.py            # Timestamp continuity, range & schema checks
│   ├── features/                   # Feature extraction & scaling
│   │   ├── __init__.py
│   │   ├── calendar.py             # Temporal cyclical encodings & holiday tags
│   │   ├── lags.py                 # Historical autoregressive lag generator
│   │   └── rolling.py              # Rolling window aggregations & statistics
│   ├── models/                     # Forecasting model architectures
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract BaseForecaster interface
│   │   ├── baseline.py             # Persistence & Seasonal Naive models
│   │   ├── tree_models.py          # LightGBM / XGBoost multi-step forecasters
│   │   ├── deep_models.py          # PyTorch LSTM / GRU recurrent networks
│   │   └── trainer.py              # Training loops, early stopping & checkpointing
│   ├── evaluation/                 # Metrics, backtesting & diagnostics
│   │   ├── __init__.py
│   │   ├── metrics.py              # RMSE, MAE, MAPE, R2 calculations
│   │   └── backtester.py           # Expanding/rolling window time-series splits
│   └── pipeline/                   # Orchestration workflow
│       ├── __init__.py
│       └── run_pipeline.py         # End-to-end training & evaluation pipeline
├── tests/                          # Automated unit and integration tests
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_pipeline.py
├── requirements.txt                # Project dependencies
└── README.md                       # Workspace overview
```

---

## 2. Core Module Responsibilities & Contracts

### 2.1 Configuration Module (`src/config/`)
- **Responsibility**: Centralized configuration management using strongly typed schemas.
- **Contract**: Exposes `ProjectConfig`, `DataConfig`, `FeatureConfig`, and `ModelConfig`. Enforces valid paths, target column names, lookback windows, and forecast horizons at launch time.

### 2.2 Data Module (`src/data/`)
- **`DataLoader`**: Reads raw formats (CSV, Parquet), verifies file accessibility, and returns standardized `pd.DataFrame`.
- **`DataValidator`**: Verifies chronological ordering, detects duplicate timestamps, checks continuity at fixed frequency, handles imputation, and validates physical value bounds.

### 2.3 Feature Engineering Module (`src/features/`)
- **`CalendarFeatureExtractor`**: Transforms timestamp index into continuous sine/cosine harmonics and categorical calendar features.
- **`LagFeatureGenerator`**: Appends autoregressive lags ($t-k$) without looking into future records.
- **`RollingFeatureGenerator`**: Computes rolling statistics using closed="left" to strictly avoid data leakage.
- **`TimeSeriesScaler`**: Encapsulates `StandardScaler` / `MinMaxScaler`, strictly fitting on training slices and transforming validation/test slices.

### 2.4 Model Module (`src/models/`)
- **`BaseForecaster`**: Abstract class defining:
  - `fit(X_train, y_train, X_val=None, y_val=None) -> None`
  - `predict(X) -> np.ndarray`
  - `save(path) -> None`
  - `load(path) -> BaseForecaster`
- **`TreeForecaster`**: Multi-output or direct recursive wrappers for LightGBM/XGBoost.
- **`DeepSequenceForecaster`**: PyTorch-based neural forecaster managing dataset sequence collators (`SequenceDataset`), recurrent state transitions, and gradient backpropagation.

### 2.5 Evaluation Module (`src/evaluation/`)
- **`MetricsCalculator`**: Computes RMSE, MAE, MAPE, and $R^2$ across point forecasts.
- **`TimeSeriesBacktester`**: Generates chronological train/test folds and compiles cross-fold stability metrics.

