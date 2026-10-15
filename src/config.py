"""Configuration module for the Energy Consumption Forecasting System.

Defines reproducible file paths, data schema parameters, feature specifications,
and model hyperparameters derived dynamically from the project root.
Supports both the baseline synthetic facility dataset and public real-world datasets
(such as UCI ElectricityLoadDiagrams2011-2014).
"""

from pathlib import Path
from typing import Dict, List, Any

# Dynamically resolve project root (no hardcoded machine paths)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Core Directory Paths
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
OUTPUTS_METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
OUTPUTS_PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
OUTPUTS_FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
IMAGES_DIR: Path = PROJECT_ROOT / "images"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# Canonical File Paths — Synthetic Facility Dataset
RAW_DATA_PATH: Path = DATA_RAW_DIR / "synthetic_facility_load.csv"
PROCESSED_DATA_PATH: Path = DATA_PROCESSED_DIR / "cleaned_facility_load.csv"
MODEL_PATH: Path = MODELS_DIR / "random_forest_forecaster.joblib"
METRICS_PATH: Path = OUTPUTS_METRICS_DIR / "model_metrics.json"
TEST_PREDICTIONS_PATH: Path = OUTPUTS_PREDICTIONS_DIR / "test_set_predictions.csv"
FORECAST_PATH: Path = OUTPUTS_PREDICTIONS_DIR / "next_24h_forecast.csv"

# Canonical File Paths — UCI ElectricityLoadDiagrams Dataset (Public Benchmark)
UCI_DOWNLOAD_URL: str = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
UCI_RAW_ZIP_PATH: Path = DATA_RAW_DIR / "electricityloaddiagrams20112014.zip"
UCI_RAW_DATA_PATH: Path = DATA_RAW_DIR / "LD2011_2014.txt"
UCI_PROCESSED_DATA_PATH: Path = DATA_PROCESSED_DIR / "uci_electricity_hourly.csv"
UCI_MODEL_PATH: Path = MODELS_DIR / "uci_forecaster_rf.joblib"
UCI_METRICS_PATH: Path = OUTPUTS_METRICS_DIR / "uci_model_metrics.json"
UCI_PREDICTIONS_PATH: Path = OUTPUTS_PREDICTIONS_DIR / "uci_test_set_predictions.csv"
UCI_FORECAST_PATH: Path = OUTPUTS_PREDICTIONS_DIR / "uci_next_24h_forecast.csv"

# Canonical File Paths & Settings — Weather-Aware Experiment (Open-Meteo Historical / ERA5)
WEATHER_RAW_CACHE_PATH: Path = DATA_RAW_DIR / "weather_lisbon_2012_2015.json"
UCI_WEATHER_PROCESSED_DATA_PATH: Path = DATA_PROCESSED_DIR / "uci_electricity_weather_hourly.csv"
UCI_WEATHER_MODEL_PATH: Path = MODELS_DIR / "uci_forecaster_rf_weather.joblib"
UCI_WEATHER_METRICS_PATH: Path = OUTPUTS_METRICS_DIR / "uci_weather_model_metrics.json"
UCI_WEATHER_PREDICTIONS_PATH: Path = OUTPUTS_PREDICTIONS_DIR / "uci_weather_test_set_predictions.csv"
UCI_WEATHER_RECURSIVE_PREDICTIONS_PATH: Path = OUTPUTS_PREDICTIONS_DIR / "uci_weather_recursive_predictions.csv"
UCI_WEATHER_FORECAST_PATH: Path = OUTPUTS_PREDICTIONS_DIR / "uci_weather_next_24h_forecast.csv"

# Geographic Proxy Metadata (Lisbon, Portugal proxy for Portuguese client MT_320)
WEATHER_PROXY_LATITUDE: float = 38.7223
WEATHER_PROXY_LONGITUDE: float = -9.1393
WEATHER_PROXY_LOCATION: str = "Lisbon, Portugal"
WEATHER_PROXY_TIMEZONE: str = "Europe/Lisbon"

# Time Series & Canonical Schema Specifications
TIMESTAMP_COL: str = "timestamp"
TARGET_COL: str = "load_kw"
ENTITY_ID_COL: str = "entity_id"
SOURCE_DATASET_COL: str = "source_dataset"

# Optional Exogenous Variables (Present in synthetic data; optional in public data)
TEMPERATURE_COL: str = "temperature_c"
PRODUCTION_COL: str = "production_status"

# Baseline Mandatory Schema (Minimal required interface)
REQUIRED_RAW_COLUMNS: List[str] = [
    TIMESTAMP_COL,
    TARGET_COL,
]

# Forecast & Modeling Settings
FORECAST_HORIZON: int = 24  # Next 24 hourly steps
RANDOM_SEED: int = 42

# Time-Series Split Configuration — Synthetic
TEST_SIZE_HOURS: int = 24 * 30  # Hold out final 30 days (720 hours) for synthetic data

# Time-Series Split Configuration — UCI Public Dataset (2012-2014 Active Window)
UCI_DEFAULT_METER: str = "MT_320"
UCI_TRAIN_START: str = "2012-01-01 00:00:00"
UCI_TRAIN_END: str = "2014-06-30 23:00:00"
UCI_VAL_START: str = "2014-07-01 00:00:00"
UCI_VAL_END: str = "2014-10-31 23:00:00"
UCI_TEST_START: str = "2014-11-01 00:00:00"
UCI_TEST_END: str = "2014-12-31 23:00:00"

# Feature Engineering Settings
LAG_HOURS: List[int] = [1, 2, 24, 48, 168]  # 1h, 2h, 1d, 2d, 1w lags
ROLLING_WINDOWS: List[int] = [6, 12, 24, 168]  # 6h, 12h, 24h, 168h rolling stats
MIN_HISTORY_REQUIRED: int = 168  # 1 week of history needed for feature computation

# Model Hyperparameters
RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 14,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}


def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    directories = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        OUTPUTS_METRICS_DIR,
        OUTPUTS_PREDICTIONS_DIR,
        OUTPUTS_FIGURES_DIR,
        IMAGES_DIR,
        DOCS_DIR,
        NOTEBOOKS_DIR,
        TESTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project Root: {PROJECT_ROOT}")
    print("Project directories verified.")
