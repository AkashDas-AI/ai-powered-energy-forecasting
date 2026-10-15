"""Streamlit interactive web dashboard for the Energy Consumption Forecasting System.

Provides visual analytics for:
- Historical energy load profiles.
- Dual-tier baseline benchmarks (24h Persistence and 168h Weekly Persistence).
- Recursive 24-hour multi-step forecasting curves.
- Support for both Synthetic Facility Simulation and UCI Electricity Real Meter data.
- Direct CSV export of forecast predictions.

Launch with:
    streamlit run app.py
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="AI Energy Consumption Forecaster",
    page_icon="⚡",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent

# File Configurations by Dataset
DATASETS_CONFIG = {
    "Synthetic Facility Load (Simulated)": {
        "data_path": PROJECT_ROOT / "data" / "processed" / "cleaned_facility_load.csv",
        "model_path": PROJECT_ROOT / "models" / "random_forest_forecaster.joblib",
        "metrics_path": PROJECT_ROOT / "outputs" / "metrics" / "model_metrics.json",
        "forecast_path": PROJECT_ROOT / "outputs" / "predictions" / "next_24h_forecast.csv",
        "is_real": False,
        "default_name": "Synthetic Industrial Facility",
        "setup_cmd": "python main.py --all",
    },
    "UCI ElectricityLoadDiagrams (Official Operational Baseline - Model A)": {
        "data_path": PROJECT_ROOT / "data" / "processed" / "uci_electricity_hourly.csv",
        "model_path": PROJECT_ROOT / "models" / "uci_forecaster_rf.joblib",
        "metrics_path": PROJECT_ROOT / "outputs" / "metrics" / "uci_model_metrics.json",
        "forecast_path": PROJECT_ROOT / "outputs" / "predictions" / "uci_next_24h_forecast.csv",
        "is_real": True,
        "is_weather_exp": False,
        "default_name": "Portuguese Smart Meter (Client MT_320 - Model A Baseline)",
        "setup_cmd": "python main.py --process-public --train-public --forecast-public",
    },
    "UCI Electricity MT_320 + Weather (Research Experiment - Model B)": {
        "data_path": PROJECT_ROOT / "data" / "processed" / "uci_electricity_weather_hourly.csv",
        "model_path": PROJECT_ROOT / "models" / "uci_forecaster_rf_weather.joblib",
        "metrics_path": PROJECT_ROOT / "outputs" / "metrics" / "uci_weather_model_metrics.json",
        "forecast_path": PROJECT_ROOT / "outputs" / "predictions" / "uci_weather_next_24h_forecast.csv",
        "is_real": True,
        "is_weather_exp": True,
        "default_name": "Portuguese Meter (MT_320) + Lisbon ERA5 Weather Proxy (Model B)",
        "setup_cmd": "python main.py --all-weather",
    },
}

# Sidebar - Dataset Selection
st.sidebar.header("📂 Dataset & Workflow Selection")
dataset_key = st.sidebar.selectbox(
    "Select Forecasting Target:",
    options=list(DATASETS_CONFIG.keys()),
    index=0,
)

active_cfg = DATASETS_CONFIG[dataset_key]
PROCESSED_DATA_PATH = active_cfg["data_path"]
MODEL_PATH = active_cfg["model_path"]
METRICS_PATH = active_cfg["metrics_path"]
FORECAST_PATH = active_cfg["forecast_path"]


def check_prerequisites() -> tuple[bool, list[str]]:
    missing = []
    if not PROCESSED_DATA_PATH.exists():
        missing.append(f"Processed dataset: {PROCESSED_DATA_PATH.name}")
    if not MODEL_PATH.exists():
        missing.append(f"Trained model artifact: {MODEL_PATH.name}")
    if not METRICS_PATH.exists():
        missing.append(f"Model metrics JSON: {METRICS_PATH.name}")
    if not FORECAST_PATH.exists():
        missing.append(f"Forecast output CSV: {FORECAST_PATH.name}")
    return len(missing) == 0, missing


# Title and Header
st.title("⚡ AI-Powered Energy Consumption Forecasting System")
if active_cfg.get("is_weather_exp"):
    st.info(
        "🔬 **Research Experiment: Model B (Weather-Aware Random Forest)** — "
        "Investigates whether historical ambient temperature adds predictive value to meter MT_320 using a Lisbon, Portugal ERA5 proxy. "
        "Finding: Temperature adds <0.5% feature importance; Model A remains the preferred operational baseline."
    )
elif active_cfg["is_real"]:
    st.info(
        "⚡ **Official Operational Baseline: Model A (Pure Autoregressive Random Forest)** — "
        "UCI ElectricityLoadDiagrams2011–2014 (Meter MT_320). "
        "Dual-tier benchmark with two-stage chronological training protocol and zero lookahead leakage."
    )
else:
    st.info(
        "🏭 **Physics-Informed Simulation: Synthetic Industrial Facility** — "
        "Simulates baseload (80 kW), 2-shift operational production calendar, HVAC temperature thresholds, and stochastic noise."
    )

# Prerequisite Verification
is_ready, missing_files = check_prerequisites()

if not is_ready:
    st.warning(f"⚠️ **Artifacts Missing for {dataset_key}**")
    st.markdown(
        f"The system could not locate the required artifacts on disk. "
        f"Please run the pipeline command for this workflow:"
    )
    st.code(active_cfg["setup_cmd"], language="bash")
    st.markdown("Missing files:")
    for item in missing_files:
        st.markdown(f"- `{item}`")
    st.stop()


@st.cache_data
def load_dataset_artifacts(data_p, metrics_p, forecast_p):
    df_hist = pd.read_csv(data_p)
    df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

    with open(metrics_p, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    df_fc = pd.read_csv(forecast_p)
    df_fc["timestamp"] = pd.to_datetime(df_fc["timestamp"])

    return df_hist, metrics, df_fc


history_df, metrics_data, forecast_df = load_dataset_artifacts(
    PROCESSED_DATA_PATH, METRICS_PATH, FORECAST_PATH
)

# Sidebar Controls
st.sidebar.divider()
st.sidebar.markdown(f"**Target Entity:** {active_cfg['default_name']}")
st.sidebar.markdown(f"**Total History Available:** {len(history_df):,} hours")
st.sidebar.markdown("**Forecast Horizon:** 24 Hours (Hourly Steps)")

history_display_days = st.sidebar.slider(
    "History View Window (Days)",
    min_value=3,
    max_value=60,
    value=14,
    step=1,
)

st.sidebar.divider()
st.sidebar.markdown("### 📘 Units & Conventions")
st.sidebar.info(
    "**Power Unit:** Kilowatts (kW)\n\n"
    "**Energy Unit:** Kilowatt-Hours (kWh)\n\n"
    "*Note:* Because each time step is exactly 1 hour (1H), the 1-hour average load "
    "in kW numerically equals the energy consumed in kWh."
)

# KPI Metrics Extraction
rf_metrics = (
    metrics_data.get("random_forest_one_step")
    or metrics_data.get("model_b_one_step_metrics")
    or metrics_data.get("random_forest_model", {})
)
baseline_24h = metrics_data.get("baseline_24h_persistence", {})
baseline_168h = metrics_data.get("baseline_168h_weekly_persistence")
rec_eval = metrics_data.get("recursive_24h_rolling_origin_evaluation")
comp_data = metrics_data.get("baseline_v1_vs_model_b_comparison", {})

if active_cfg.get("is_weather_exp") and comp_data:
    one_step_comp = comp_data.get("one_step_comparison", {})
    rmse_delta_val = one_step_comp.get("rmse_delta_kw")
    wape_delta_val = one_step_comp.get("wape_delta_pct")
    rmse_delta = f"{rmse_delta_val:+.2f} kW vs Model A" if rmse_delta_val is not None else None
    wape_delta = f"{wape_delta_val:+.2f}% vs Model A" if wape_delta_val is not None else None
elif baseline_24h.get("rmse"):
    rmse_diff = round((rf_metrics.get("rmse", 0.0) - baseline_24h.get("rmse", 0.0)), 2)
    wape_diff = round((rf_metrics.get("wape_pct", 0.0) - baseline_24h.get("wape_pct", 0.0)), 2)
    rmse_delta = f"{rmse_diff:+.2f} kW vs Base"
    wape_delta = f"{wape_diff:+.2f}% vs Base"
else:
    rmse_delta = None
    wape_delta = None

total_forecast_energy_kwh = forecast_df["predicted_energy_kwh"].sum()
peak_forecast_kw = forecast_df["predicted_load_kw"].max()

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    label="Next 24h Projected Energy",
    value=f"{total_forecast_energy_kwh:,.1f} kWh",
    help="Sum of projected hourly energy consumption for the 24-hour forecast horizon.",
)
col2.metric(
    label="Projected Peak Demand",
    value=f"{peak_forecast_kw:.1f} kW",
    help="Maximum instantaneous hourly average load expected in the 24-hour forecast horizon.",
)
col3.metric(
    label="Model Holdout RMSE",
    value=f"{rf_metrics.get('rmse', 0.0):.2f} kW",
    delta=rmse_delta,
    delta_color="inverse",
    help="Root Mean Squared Error on out-of-sample holdout test partition.",
)
col4.metric(
    label="Model Holdout WAPE",
    value=f"{rf_metrics.get('wape_pct', 0.0):.2f}%",
    delta=wape_delta,
    delta_color="inverse",
    help="Weighted Absolute Percentage Error on out-of-sample holdout test partition.",
)

st.divider()

# Tab Navigation
tab1, tab2, tab3 = st.tabs(["📈 24-Hour Forecast", "🕒 Historical Profile", "📊 Benchmark & Validation"])

with tab1:
    start_ts = forecast_df["timestamp"].min()
    end_ts = forecast_df["timestamp"].max()
    st.subheader(f"24-Hour Forecast Curve (Out-of-Sample Horizon: {start_ts.strftime('%Y-%m-%d %H:%M')} to {end_ts.strftime('%Y-%m-%d %H:%M')})")
    st.caption(
        "ℹ️ Note: This visualizes an out-of-sample operational forecast evaluated at the conclusion of the historical window, "
        "not a live streaming IoT prediction. Power units are kilowatts (kW) and energy is kilowatt-hours (kWh)."
    )

    recent_history_hours = 48
    recent_actuals = history_df.iloc[-recent_history_hours:][["timestamp", "load_kw"]].copy()
    recent_actuals = recent_actuals.rename(columns={"load_kw": "Load (kW)"})
    recent_actuals["Series"] = "Historical Actual"

    future_series = forecast_df[["timestamp", "predicted_load_kw"]].copy()
    future_series = future_series.rename(columns={"predicted_load_kw": "Load (kW)"})
    future_series["Series"] = "Forecast (Next 24h)"

    combined_plot_df = pd.concat([recent_actuals, future_series], ignore_index=True)
    pivot_chart = combined_plot_df.pivot(index="timestamp", columns="Series", values="Load (kW)")
    st.line_chart(pivot_chart, use_container_width=True)

    col_table, col_dl = st.columns([3, 1])
    with col_table:
        st.markdown("#### Hourly Forecast Breakdown")
        display_forecast_df = forecast_df.copy()
        display_forecast_df["timestamp"] = display_forecast_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        rename_map = {
            "timestamp": "Timestamp",
            "predicted_load_kw": "Forecast Load (kW)",
            "predicted_energy_kwh": "Energy (kWh)",
            "temperature_c": "Ambient Temp (°C)",
            "production_status": "Production Shift",
        }
        display_cols = [c for c in display_forecast_df.columns if c in rename_map]
        st.dataframe(display_forecast_df[display_cols].rename(columns=rename_map), use_container_width=True, height=300)

    with col_dl:
        st.markdown("#### Export Forecast")
        st.write("Download the generated predictions as a standard CSV file:")
        forecast_csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast CSV",
            data=forecast_csv_bytes,
            file_name=f"energy_forecast_24h_{active_cfg['data_path'].stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tab2:
    st.subheader(f"Historical Energy Profile (Past {history_display_days} Days)")
    hours_to_show = history_display_days * 24
    history_slice = history_df.iloc[-hours_to_show:].copy()

    plot_cols = ["load_kw"]
    if "temperature_c" in history_slice.columns:
        plot_cols.append("temperature_c")

    history_plot_df = history_slice.set_index("timestamp")[plot_cols]
    st.line_chart(history_plot_df, use_container_width=True)

with tab3:
    st.subheader("Model Evaluation & Performance Comparison")
    st.markdown(f"**Protocol:** {metrics_data.get('evaluation_protocol', 'Holdout evaluation')}")

    eval_col1, eval_col2, eval_col3 = st.columns(3)
    if active_cfg.get("is_weather_exp") and comp_data:
        one_step_comp = comp_data.get("one_step_comparison", {})
        rec_exp2 = metrics_data.get("experiment_2_operational_persistence_weather_recursive_24h", {}).get("aggregate_metrics", {})
        with eval_col1:
            st.markdown("#### Candidate: Model B (One-Step)")
            st.write(f"- **RMSE:** {rf_metrics.get('rmse', 'N/A')} kW")
            st.write(f"- **MAE:** {rf_metrics.get('mae', 'N/A')} kW")
            st.write(f"- **WAPE:** {rf_metrics.get('wape_pct', 'N/A')}%")
            st.write(f"- **R² Score:** {rf_metrics.get('r2_score', 'N/A')}")
            st.write(f"- **Holdout Samples:** {rf_metrics.get('sample_count', 'N/A')}")

        with eval_col2:
            st.markdown("#### Baseline v1: Model A (One-Step)")
            st.write(f"- **RMSE:** {one_step_comp.get('model_a_rmse', 'N/A')} kW")
            st.write(f"- **MAE:** {one_step_comp.get('model_a_mae', 'N/A')} kW")
            st.write(f"- **WAPE:** {one_step_comp.get('model_a_wape_pct', 'N/A')}%")
            st.write(f"- **R² Score:** {one_step_comp.get('model_a_r2', 'N/A')}")
            st.write(f"- **Holdout Samples:** {rf_metrics.get('sample_count', 'N/A')}")

        with eval_col3:
            st.markdown("#### Model B Recursive 24h (Operational)")
            st.write(f"- **RMSE:** {rec_exp2.get('rmse', 'N/A')} kW")
            st.write(f"- **MAE:** {rec_exp2.get('mae', 'N/A')} kW")
            st.write(f"- **WAPE:** {rec_exp2.get('wape_pct', 'N/A')}%")
            st.write(f"- **R² Score:** {rec_exp2.get('r2_score', 'N/A')}")
    else:
        with eval_col1:
            label_col1 = "Model A (One-Step Holdout)" if active_cfg["is_real"] else "Random Forest (One-Step)"
            st.markdown(f"#### {label_col1}")
            st.write(f"- **RMSE:** {rf_metrics.get('rmse', 'N/A')} kW")
            st.write(f"- **MAE:** {rf_metrics.get('mae', 'N/A')} kW")
            st.write(f"- **WAPE:** {rf_metrics.get('wape_pct', 'N/A')}%")
            st.write(f"- **R² Score:** {rf_metrics.get('r2_score', 'N/A')}")
            st.write(f"- **Holdout Samples:** {rf_metrics.get('sample_count', 'N/A')}")

        with eval_col2:
            st.markdown("#### 24h Seasonal Naive Base")
            st.write(f"- **RMSE:** {baseline_24h.get('rmse', 'N/A')} kW")
            st.write(f"- **MAE:** {baseline_24h.get('mae', 'N/A')} kW")
            st.write(f"- **WAPE:** {baseline_24h.get('wape_pct', 'N/A')}%")
            st.write(f"- **R² Score:** {baseline_24h.get('r2_score', 'N/A')}")

        with eval_col3:
            if baseline_168h:
                st.markdown("#### 168h Weekly Persistence Base")
                st.write(f"- **RMSE:** {baseline_168h.get('rmse', 'N/A')} kW")
                st.write(f"- **MAE:** {baseline_168h.get('mae', 'N/A')} kW")
                st.write(f"- **WAPE:** {baseline_168h.get('wape_pct', 'N/A')}%")
                st.write(f"- **R² Score:** {baseline_168h.get('r2_score', 'N/A')}")
            elif rec_eval:
                rec_agg = rec_eval.get("aggregate_metrics", {})
                st.markdown("#### Recursive 24h Rolling-Origin")
                st.write(f"- **RMSE:** {rec_agg.get('rmse', 'N/A')} kW")
                st.write(f"- **MAE:** {rec_agg.get('mae', 'N/A')} kW")
                st.write(f"- **WAPE:** {rec_agg.get('wape_pct', 'N/A')}%")
                st.write(f"- **Evaluated Origins:** {rec_eval.get('evaluated_origins_count', 'N/A')}")
            else:
                st.markdown("#### Additional Metrics")
                st.write("No secondary baseline recorded.")

    # Multi-Step Recursive Summary for Model A
    if active_cfg["is_real"] and not active_cfg.get("is_weather_exp") and rec_eval:
        st.markdown("---")
        st.markdown("#### 🔄 Autonomous Multi-Step Benchmark: Recursive 24-Hour Rolling-Origin (Model A)")
        rec_agg = rec_eval.get("aggregate_metrics", {})
        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        r_c1.metric("Recursive 24h RMSE", f"{rec_agg.get('rmse', 'N/A')} kW")
        r_c2.metric("Recursive 24h MAE", f"{rec_agg.get('mae', 'N/A')} kW")
        r_c3.metric("Recursive 24h WAPE", f"{rec_agg.get('wape_pct', 'N/A')}%")
        r_c4.metric("Evaluated Origins", f"{rec_eval.get('evaluated_origins_count', 'N/A')} (1,440 pts)")

    # Weather-Aware Comparative Experiment Diagnostics
    comp_data = metrics_data.get("baseline_v1_vs_model_b_comparison")
    if comp_data:
        st.divider()
        st.markdown("### 🔬 Controlled Experiment: Model A vs Model B Comparison")
        st.markdown(
            "Comparing **Baseline v1 (Model A - Pure Autoregressive)** with "
            "**Candidate (Model B - Weather-Aware with Lisbon ERA5 Proxy)**:"
        )

        one_step_comp = comp_data.get("one_step_comparison", {})
        rec_comp = comp_data.get("recursive_24h_comparison", {})

        comp_df = pd.DataFrame([
            {
                "Evaluation Mode": "One-Step Ahead Holdout",
                "Metric": "MAE (kW)",
                "Model A (Baseline v1)": f"{one_step_comp.get('model_a_mae', 0.0):.2f}",
                "Model B (Weather)": f"{one_step_comp.get('model_b_mae', 0.0):.2f}",
                "Delta (B - A)": f"{one_step_comp.get('mae_delta_kw', 0.0):+.4f} kW",
            },
            {
                "Evaluation Mode": "One-Step Ahead Holdout",
                "Metric": "RMSE (kW)",
                "Model A (Baseline v1)": f"{one_step_comp.get('model_a_rmse', 0.0):.2f}",
                "Model B (Weather)": f"{one_step_comp.get('model_b_rmse', 0.0):.2f}",
                "Delta (B - A)": f"{one_step_comp.get('rmse_delta_kw', 0.0):+.4f} kW",
            },
            {
                "Evaluation Mode": "One-Step Ahead Holdout",
                "Metric": "WAPE (%)",
                "Model A (Baseline v1)": f"{one_step_comp.get('model_a_wape_pct', 0.0):.2f}%",
                "Model B (Weather)": f"{one_step_comp.get('model_b_wape_pct', 0.0):.2f}%",
                "Delta (B - A)": f"{one_step_comp.get('wape_delta_pct', 0.0):+.2f}%",
            },
            {
                "Evaluation Mode": "Recursive 24h (Retrospective Reanalysis)",
                "Metric": "WAPE (%)",
                "Model A (Baseline v1)": f"{rec_comp.get('model_a_recursive_wape_pct', 0.0):.2f}%",
                "Model B (Weather)": f"{rec_comp.get('model_b_exp1_retrospective_wape_pct', 0.0):.2f}%",
                "Delta (B - A)": f"{rec_comp.get('model_b_exp1_retrospective_wape_pct', 0.0) - rec_comp.get('model_a_recursive_wape_pct', 0.0):+.2f}%",
            },
            {
                "Evaluation Mode": "Recursive 24h (Operational Diurnal Fallback)",
                "Metric": "WAPE (%)",
                "Model A (Baseline v1)": f"{rec_comp.get('model_a_recursive_wape_pct', 0.0):.2f}%",
                "Model B (Weather)": f"{rec_comp.get('model_b_exp2_operational_wape_pct', 0.0):.2f}%",
                "Delta (B - A)": f"{rec_comp.get('model_b_exp2_operational_wape_pct', 0.0) - rec_comp.get('model_a_recursive_wape_pct', 0.0):+.2f}%",
            },
        ])
        st.dataframe(comp_df, use_container_width=True)

        weather_imp = metrics_data.get("weather_features_total_importance_pct", 0.0)
        st.info(
            f"**Feature Importance Finding:** Ambient temperature features contribute **{weather_imp:.4f}%** "
            f"total importance (out of 26 features). Client MT_320 is predominantly autoregressive and calendar-driven."
        )
        st.caption(f"**Scientific Conclusion:** {comp_data.get('scientific_conclusion', '')}")

