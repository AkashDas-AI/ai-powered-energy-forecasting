# Known Limitations & Engineering Assumptions

This document explicitly outlines the analytical boundaries, physical assumptions, empirical constraints, and known limitations of the **AI-Powered Energy Consumption Forecasting System**.

---

## 1. Synthetic Data Limitations

- **Simplified Operational Reality**: The synthetic dataset is generated through physical and heuristic equations. While it models diurnal curves, production shifts, and thermal HVAC demand, it does not reflect unanticipated real-world industrial anomalies such as sudden equipment failures, voltage sags, power outages, or major factory retooling.
- **Stationary Production Rules**: Real commercial facilities experience shifting order books, overtime shifts, and seasonal shutdowns that deviate from fixed calendar rules.

---

## 2. Exogenous Forecasting & Weather Assumptions

- **Future Temperature Availability**: Real-world operations require numerical weather prediction (NWP) feeds. In the absence of an external day-ahead forecast API, the current forecaster relies on a **24-hour diurnal persistence assumption** ($\hat{T}_{t+h} = T_{t+h-24}$). During rapid cold fronts or heatwaves, this assumption introduces forecast error.
- **Geographic Weather Proxy Noise**: Open-Meteo ERA5 data is anchored to Lisbon grid coordinates (`38.7223° N, -9.1393° W`) because Portuguese utility client addresses are anonymized. Microclimate differences between Lisbon and the true client facility introduce spatial proxy variance; Lisbon weather is strictly a geographic proxy, not a verified local facility microclimate.
- **Reanalysis vs. Operational Forecasts**: Retrospective ERA5 reanalysis represents post-event meteorological observation. Evaluating models on future reanalysis represents a diagnostic hindcast; true operational performance must be evaluated using operational weather persistence, which degrades recursive multi-step WAPE from 5.61% to 5.64%.

---

## 3. Cold Start & Historical Lookback Constraints

- **Minimum Warm-Up Requirement**: The model utilizes 168-hour (7-day) autoregressive lags and rolling statistics. Consequently, a facility must possess at least **168 hours of uninterrupted historical data** before the inference engine can generate its first forecast.
- **Data Gap Recovery**: Extended sensor outages exceeding several days cannot be reliably reconstructed through linear interpolation and will require specialized seasonal imputation models.

---

## 4. Error Compounding in Recursive Multi-Step Forecasting

- In recursive multi-step forecasting ($h=1, \dots, 24$), prediction $\hat{y}_{t+1}$ is fed back into the feature matrix as the lag for step $t+2$.
- Any prediction error in early forecast steps ($h=1, 2$) propagates and compounds into later horizons ($h=18, \dots, 24$):
  - $h=+1$: 2.14 kW MAE
  - $h=+6$: 3.99 kW MAE
  - $h=+12$: 6.16 kW MAE
  - $h=+18$: 6.68 kW MAE
  - $h=+24$: 5.95 kW MAE (re-anchors to 24-hour observed lag)

---

## 5. Point Forecasting vs. Probabilistic Intervals

- The current implementation delivers deterministic point predictions ($\mathbb{E}[y_{t+h}]$).
- It does not yet generate probabilistic prediction intervals (e.g. 10th and 90th quantile bounds), which are valuable for operational risk management and reserve spinning capacity decisions in utility markets.

---

## 6. Public Dataset Limitations (UCI ElectricityLoadDiagrams2011-2014)

- **Single-Meter Evaluation**: Evaluation is grounded on Portuguese utility client `MT_320`. While validating pipeline functionality, multi-step autoregressive dynamics, and baseline outperformance, results on a single meter cannot be assumed to generalize universally across all 370 consumers without cohort benchmarking.
- **Anonymized Consumer Classifications**: The 370 client columns (`MT_001` through `MT_370`) represent Portuguese smart meters. However, official UCI metadata does not distinguish between industrial plants, commercial buildings, agricultural loads, or residential apartments. While `MT_320` is selected for high active load and continuity, specific industrial facility claims cannot be made without external utility metadata.
- **Daylight Saving Time (DST) Logger Telemetry Semantics**: The raw Portuguese dataset records telemetry on a fixed 96-interval calendar grid every day. Spring transition contains placeholder readings (~0.60 kW) and Autumn transition aggregates two hours into one slot. Because the grid operator's internal dispatch clock cannot be independently verified from public metadata, observed data is preserved faithfully as recorded rather than artificially manipulated.
- **Heterogeneous Connection Dates**: Client installations occurred asynchronously throughout 2011–2014, requiring explicit connection window detection to prevent corrupting training sets with uninstalled zero sequences.

---

## 7. Future Architectural Roadmap

1. **Numerical Weather Prediction (NWP) Forecast Integration**: Ingest real-time day-ahead weather forecasts rather than relying on diurnal persistence fallback.
2. **Advanced Multi-Horizon Architectures**: Benchmark deep sequence models (e.g. Temporal Fusion Transformer, N-BEATS, PatchTST) against Baseline v1.
3. **Multi-Meter Cohort Clustering**: Group the remaining 369 UCI meters into industrial, commercial, and residential clusters to evaluate weather sensitivity across consumer types.
