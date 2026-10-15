"""Command-line entrypoint for the Energy Consumption Forecasting pipeline.

Provides structured commands to execute the synthetic pipeline or the public UCI dataset workflow:
- Synthetic Facility Pipeline:
    python main.py --all
    python main.py --generate-data --days 730
    python main.py --preprocess
    python main.py --train
    python main.py --forecast

- Public Dataset (UCI ElectricityLoadDiagrams) Pipeline:
    python main.py --download-uci
    python main.py --process-public --meter MT_320
    python main.py --train-public
    python main.py --forecast-public
    python main.py --all-public
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    FORECAST_PATH,
    UCI_RAW_DATA_PATH,
    UCI_PROCESSED_DATA_PATH,
    UCI_MODEL_PATH,
    UCI_METRICS_PATH,
    UCI_PREDICTIONS_PATH,
    UCI_FORECAST_PATH,
    WEATHER_RAW_CACHE_PATH,
    UCI_WEATHER_PROCESSED_DATA_PATH,
    UCI_WEATHER_MODEL_PATH,
    UCI_WEATHER_METRICS_PATH,
    UCI_WEATHER_PREDICTIONS_PATH,
    UCI_WEATHER_FORECAST_PATH,
    WEATHER_PROXY_LOCATION,
    UCI_DEFAULT_METER,
    UCI_TRAIN_START,
    UCI_TEST_START,
    UCI_TEST_END,
    RANDOM_SEED,
    ensure_directories,
)
from src.data_generation import generate_synthetic_energy_data
from src.preprocessing import preprocess_energy_data
from src.train import train_and_evaluate
from src.forecast import generate_forecast, load_model_bundle
from src.public_dataset import download_uci_electricity, process_public_dataset
from src.weather import fetch_open_meteo_weather, align_weather_with_electricity, run_weather_experiment


def run_synthetic_pipeline(
    do_generate: bool = False,
    do_preprocess: bool = False,
    do_train: bool = False,
    do_forecast: bool = False,
    num_days: int = 730,
    seed: int = RANDOM_SEED,
) -> None:
    """Execute synthetic facility pipeline stages."""
    ensure_directories()

    print("==================================================================")
    print(" AI-POWERED ENERGY CONSUMPTION FORECASTING SYSTEM (SYNTHETIC)")
    print("==================================================================")

    # Stage 1: Data Generation
    if do_generate:
        print(f"\n[Stage 1/4] Generating synthetic energy dataset ({num_days} days, seed={seed})...")
        df_raw = generate_synthetic_energy_data(
            num_days=num_days,
            seed=seed,
            output_path=RAW_DATA_PATH,
        )
        print(f" -> Successfully generated {len(df_raw)} records at: {RAW_DATA_PATH}")

    # Stage 2: Preprocessing
    if do_preprocess:
        print("\n[Stage 2/4] Validating and preprocessing time series...")
        df_clean = preprocess_energy_data(
            input_source=RAW_DATA_PATH,
            output_path=PROCESSED_DATA_PATH,
            allow_imputation=True,
        )
        print(f" -> Preprocessed {len(df_clean)} records saved to: {PROCESSED_DATA_PATH}")

    # Stage 3: Training & Evaluation
    if do_train:
        print("\n[Stage 3/4] Training Baseline & RandomForestRegressor (Chronological Holdout)...")
        results = train_and_evaluate(
            data_path=PROCESSED_DATA_PATH,
            model_output_path=MODEL_PATH,
            metrics_output_path=METRICS_PATH,
            predictions_output_path=TEST_PREDICTIONS_PATH,
            run_recursive_evaluation=True,
        )
        baseline = results["baseline_24h_persistence"]
        rf = results["random_forest_one_step"]
        imp = results["performance_improvement_one_step_vs_24h_base"]

        print(f" -> Baseline (24h Persistence) : RMSE={baseline['rmse']:.2f} kW | MAE={baseline['mae']:.2f} kW | WAPE={baseline['wape_pct']:.2f}%")
        print(f" -> Random Forest Forecaster  : RMSE={rf['rmse']:.2f} kW | MAE={rf['mae']:.2f} kW | WAPE={rf['wape_pct']:.2f}%")
        print(f" -> Metric Improvement         : WAPE reduction: {imp['wape_reduction_pct']}% | RMSE reduction: {imp['rmse_reduction_pct']}%")
        if results.get("recursive_24h_rolling_origin_evaluation"):
            rec_agg = results["recursive_24h_rolling_origin_evaluation"]["aggregate_metrics"]
            print(f" -> Recursive 24h Multi-Step   : RMSE={rec_agg['rmse']:.2f} kW | MAE={rec_agg['mae']:.2f} kW | WAPE={rec_agg['wape_pct']:.2f}%")
        print(f" -> Model bundle saved to      : {MODEL_PATH}")
        print(f" -> Metrics saved to           : {METRICS_PATH}")
        print(f" -> Predictions saved to       : {TEST_PREDICTIONS_PATH}")

    # Stage 4: Forecasting
    if do_forecast:
        print("\n[Stage 4/4] Generating next 24-hour multi-step recursive forecast...")
        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Missing processed data at {PROCESSED_DATA_PATH}. Run preprocessing first."
            )
        df_clean = pd.read_csv(PROCESSED_DATA_PATH)
        recent_history = df_clean.iloc[-336:].copy()

        forecast_df = generate_forecast(
            history_df=recent_history,
            model_path=MODEL_PATH,
            horizon=24,
            output_path=FORECAST_PATH,
        )
        start_ts = forecast_df["timestamp"].min()
        end_ts = forecast_df["timestamp"].max()
        avg_load = forecast_df["predicted_load_kw"].mean()
        total_energy = forecast_df["predicted_energy_kwh"].sum()

        print(f" -> Forecast generated for {len(forecast_df)} hourly intervals:")
        print(f"    Horizon Window : {start_ts} to {end_ts}")
        print(f"    Average Load   : {avg_load:.2f} kW")
        print(f"    Total Energy   : {total_energy:.2f} kWh")
        print(f" -> Forecast CSV saved to: {FORECAST_PATH}")

    print("\n==================================================================")
    print(" Synthetic pipeline execution finished successfully.")
    print("==================================================================")


def run_public_pipeline(
    do_download: bool = False,
    do_process: bool = False,
    do_train: bool = False,
    do_forecast: bool = False,
    meter_id: str = UCI_DEFAULT_METER,
) -> None:
    """Execute public UCI Electricity dataset workflow."""
    ensure_directories()

    print("==================================================================")
    print(" AI-POWERED ENERGY CONSUMPTION FORECASTING (UCI PUBLIC DATASET)")
    print(f" Target Meter: {meter_id}")
    print("==================================================================")

    # 1. Download
    if do_download:
        print("\n[Public 1/4] Downloading official UCI ElectricityLoadDiagrams archive...")
        raw_path = download_uci_electricity()
        print(f" -> Raw file ready at: {raw_path}")

    # 2. Ingestion & Hourly Resampling
    if do_process:
        print(f"\n[Public 2/4] Parsing and resampling client '{meter_id}' to 1-hour intervals...")
        if not UCI_RAW_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Raw dataset file not found at: {UCI_RAW_DATA_PATH}.\n"
                "Please download the archive using 'python main.py --download-uci' or place LD2011_2014.txt in data/raw/."
            )
        canonical_df = process_public_dataset(
            raw_path=UCI_RAW_DATA_PATH,
            output_path=UCI_PROCESSED_DATA_PATH,
            meter_id=meter_id,
            start_date=UCI_TRAIN_START,
            end_date=UCI_TEST_END,
        )
        print(f" -> Resampled {len(canonical_df):,} hourly records saved to: {UCI_PROCESSED_DATA_PATH}")

    # 3. Model Training & Dual Evaluation
    if do_train:
        print(f"\n[Public 3/4] Training Baseline & RandomForest Forecaster on client '{meter_id}'...")
        if not UCI_PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Processed public dataset not found at: {UCI_PROCESSED_DATA_PATH}.\n"
                "Run 'python main.py --process-public' first."
            )
        split_dict = {
            "train_end": "2014-06-30 23:00:00",
            "test_start": UCI_TEST_START,  # 2014-11-01
            "test_end": UCI_TEST_END,      # 2014-12-31
        }
        report = train_and_evaluate(
            data_path=UCI_PROCESSED_DATA_PATH,
            model_output_path=UCI_MODEL_PATH,
            metrics_output_path=UCI_METRICS_PATH,
            predictions_output_path=UCI_PREDICTIONS_PATH,
            split_dates=split_dict,
            run_recursive_evaluation=True,
            origin_stride_hours=24,
        )
        b24 = report["baseline_24h_persistence"]
        b168 = report.get("baseline_168h_weekly_persistence")
        rf = report["random_forest_one_step"]

        print(f" -> Baseline (24h Persistence) : RMSE={b24['rmse']:.2f} kW | MAE={b24['mae']:.2f} kW | WAPE={b24['wape_pct']:.2f}%")
        if b168:
            print(f" -> Baseline (168h Persistence): RMSE={b168['rmse']:.2f} kW | MAE={b168['mae']:.2f} kW | WAPE={b168['wape_pct']:.2f}%")
        print(f" -> Random Forest One-Step     : RMSE={rf['rmse']:.2f} kW | MAE={rf['mae']:.2f} kW | WAPE={rf['wape_pct']:.2f}%")

        if report.get("recursive_24h_rolling_origin_evaluation"):
            rec_agg = report["recursive_24h_rolling_origin_evaluation"]["aggregate_metrics"]
            print(f" -> Recursive 24h Multi-Step   : RMSE={rec_agg['rmse']:.2f} kW | MAE={rec_agg['mae']:.2f} kW | WAPE={rec_agg['wape_pct']:.2f}%")

        print(f" -> Public model bundle saved to: {UCI_MODEL_PATH}")
        print(f" -> Public metrics saved to     : {UCI_METRICS_PATH}")
        print(f" -> Predictions saved to        : {UCI_PREDICTIONS_PATH}")

    # 4. 24-Hour Forecast
    if do_forecast:
        print(f"\n[Public 4/4] Generating next 24-hour forecast for client '{meter_id}'...")
        if not UCI_PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(f"Missing processed data at {UCI_PROCESSED_DATA_PATH}.")
        df_clean = pd.read_csv(UCI_PROCESSED_DATA_PATH)
        recent_history = df_clean.iloc[-336:].copy()

        forecast_df = generate_forecast(
            history_df=recent_history,
            model_path=UCI_MODEL_PATH,
            horizon=24,
            output_path=UCI_FORECAST_PATH,
        )
        start_ts = forecast_df["timestamp"].min()
        end_ts = forecast_df["timestamp"].max()
        avg_load = forecast_df["predicted_load_kw"].mean()
        total_energy = forecast_df["predicted_energy_kwh"].sum()

        print(f" -> Forecast generated for {len(forecast_df)} hourly intervals:")
        print(f"    Horizon Window : {start_ts} to {end_ts}")
        print(f"    Average Load   : {avg_load:.2f} kW")
        print(f"    Total Energy   : {total_energy:.2f} kWh")
        print(f" -> Forecast CSV saved to: {UCI_FORECAST_PATH}")

    print("\n==================================================================")
    print(" Public dataset pipeline finished.")
    print("==================================================================")


def run_weather_pipeline(
    do_process: bool = False,
    do_train: bool = False,
    do_forecast: bool = False,
) -> None:
    """Execute controlled weather-aware forecasting experiment pipeline."""
    ensure_directories()

    print("==================================================================")
    print(" WEATHER-AWARE ENERGY FORECASTING EXPERIMENT (MT_320 + LISBON ERA5)")
    print(f" Geographic Proxy: {WEATHER_PROXY_LOCATION}")
    print("==================================================================")

    if do_process:
        print("\n[Weather 1/2] Fetching/verifying historical reanalysis weather cache...")
        fetch_open_meteo_weather()
        print("\n[Weather 2/2] Aligning weather observations with UCI electricity dataset...")
        align_weather_with_electricity()

    if do_train or do_forecast:
        print("\n[Weather] Executing Model B training, feature evaluation, and forecasting...")
        results = run_weather_experiment()
        one_step = results["model_b_one_step_metrics"]
        comp = results.get("baseline_v1_vs_model_b_comparison", {}).get("one_step_comparison", {})
        print(f" -> Model B One-Step Holdout: RMSE={one_step['rmse']:.2f} kW | MAE={one_step['mae']:.2f} kW | WAPE={one_step['wape_pct']:.2f}%")
        print(f" -> Comparison vs Model A   : MAE delta: {comp.get('mae_delta_kw', 0.0):+.4f} kW | RMSE delta: {comp.get('rmse_delta_kw', 0.0):+.4f} kW")
        print(f" -> Total Weather Importance: {results['weather_features_total_importance_pct']:.4f}%")
        print(f" -> Artifacts saved to:")
        print(f"    - Model   : {UCI_WEATHER_MODEL_PATH}")
        print(f"    - Metrics : {UCI_WEATHER_METRICS_PATH}")
        print(f"    - Forecast: {UCI_WEATHER_FORECAST_PATH}")

    print("\n==================================================================")
    print(" Weather pipeline finished.")
    print("==================================================================")


def main() -> None:
    """Parse CLI arguments and route to appropriate pipeline."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Energy Consumption Forecasting System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Synthetic Workflow Flags
    parser.add_argument("--all", action="store_true", help="Run full synthetic pipeline.")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data.")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess synthetic data.")
    parser.add_argument("--train", action="store_true", help="Train synthetic model.")
    parser.add_argument("--forecast", action="store_true", help="Forecast next 24h for synthetic facility.")
    parser.add_argument("--days", type=int, default=730, help="Days to simulate for synthetic data.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")

    # Public Dataset Workflow Flags
    parser.add_argument("--all-public", action="store_true", help="Run full public dataset pipeline (process, train, forecast).")
    parser.add_argument("--download-uci", action="store_true", help="Download official UCI Electricity archive.")
    parser.add_argument("--process-public", action="store_true", help="Parse and resample public dataset for specified meter.")
    parser.add_argument("--train-public", action="store_true", help="Train model on public dataset.")
    parser.add_argument("--forecast-public", action="store_true", help="Generate 24h forecast for public meter.")
    parser.add_argument("--meter", type=str, default=UCI_DEFAULT_METER, help="Client meter ID in UCI dataset (e.g. MT_320).")

    # Weather-Aware Experiment Flags
    parser.add_argument("--all-weather", action="store_true", help="Run full weather experiment (process, train, forecast).")
    parser.add_argument("--process-weather", action="store_true", help="Fetch and align weather data with UCI electricity.")
    parser.add_argument("--train-weather", action="store_true", help="Train Model B (weather-aware) and evaluate against Baseline v1.")
    parser.add_argument("--forecast-weather", action="store_true", help="Generate 24h forecast using weather-aware Model B.")

    args = parser.parse_args()

    # Weather-aware experiment branch
    if args.all_weather:
        run_weather_pipeline(do_process=True, do_train=True, do_forecast=True)
    elif args.process_weather or args.train_weather or args.forecast_weather:
        run_weather_pipeline(
            do_process=args.process_weather,
            do_train=args.train_weather,
            do_forecast=args.forecast_weather,
        )
    # Public dataset branch
    elif args.all_public:
        run_public_pipeline(do_process=True, do_train=True, do_forecast=True, meter_id=args.meter)
    elif args.download_uci or args.process_public or args.train_public or args.forecast_public:
        run_public_pipeline(
            do_download=args.download_uci,
            do_process=args.process_public,
            do_train=args.train_public,
            do_forecast=args.forecast_public,
            meter_id=args.meter,
        )
    # Synthetic dataset branch (default)
    elif args.all or args.generate_data or args.preprocess or args.train or args.forecast:
        run_synthetic_pipeline(
            do_generate=args.all or args.generate_data,
            do_preprocess=args.all or args.preprocess,
            do_train=args.all or args.train,
            do_forecast=args.all or args.forecast,
            num_days=args.days,
            seed=args.seed,
        )
    else:
        print("No specific flag provided. Defaulting to full synthetic pipeline (--all)...")
        run_synthetic_pipeline(do_generate=True, do_preprocess=True, do_train=True, do_forecast=True)


if __name__ == "__main__":
    main()
