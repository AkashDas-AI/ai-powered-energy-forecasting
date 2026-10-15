# Methodology & Modeling Approach

This document describes the analytical principles, engineering strategies, and modeling methodology implemented in the **AI-Powered Energy Consumption Forecasting System**, covering both the baseline synthetic engine and the migration plan for real-world public datasets.

---

## 1. Synthetic Data Generation Principles

To simulate the physical energy dynamics of an industrial/commercial facility without relying on proprietary or unverified external sensor dumps, [`src/data_generation.py`](../src/data_generation.py) synthesizes four distinct physical components:

1. **Continuous Baseload ($P_{\text{base}}$)**: Minimum power demand (80 kW) from servers, security, and continuous refrigeration.
2. **Operational Shift Dynamics ($P_{\text{prod}}$)**: Operating schedules derived from facility calendar rules:
   - Weekdays (Mon–Fri): 07:00 to 19:00 full production ($\sim 220$ kW with intra-day ramp curve).
   - Saturdays: 08:00 to 14:00 half-day shift.
   - Sundays / Nights: Standby mode.
3. **Thermal HVAC Dynamics ($P_{\text{hvac}}$)**:
   - Cooling activation: When $T > 21^\circ\text{C}$, load increases by $7.5\text{ kW}/^\circ\text{C}$.
   - Heating activation: When $T < 11^\circ\text{C}$, load increases by $5.2\text{ kW}/^\circ\text{C}$.
4. **Stochastic Variation ($\epsilon_t$)**: Zero-mean Gaussian noise $\mathcal{N}(0, 7.5\text{ kW})$.
5. **Physical Clamping**: The total load is clamped at $P_t = \max(P_{\text{base}} + P_{\text{prod}} + P_{\text{hvac}} + \epsilon_t, 20.0\text{ kW})$ to guarantee non-negative consumption.

---

## 2. Preprocessing & Temporal Continuity

Industrial sensors frequently suffer from communication dropouts, clock desynchronization, and transmission duplicates:
- **Duplicate Timestamp Resolution**: Any duplicate timestamp raises an immediate `ValueError` to prevent conflicting readings.
- **Hourly Grid Re-indexing**: The series is aligned to a complete `pd.date_range` with hourly frequency (`1H`).
- **Audit-Logged Imputation**: Missing intervals are linearly interpolated and flagged with an explicit boolean column `is_imputed = True`, ensuring auditability.

---

## 3. Feature Engineering & Leakage Prevention

Temporal dependencies and periodicities are encoded through three feature sets:
1. **Calendar Harmonics**: Both discrete indices (`hour`, `day_of_week`, `month`) and continuous trigonometric transformations ($\sin/\cos$) are computed to preserve circular distances (e.g. 23:00 is adjacent to 00:00).
2. **Autoregressive Lags**: $y_{t-1}, y_{t-2}, y_{t-24}, y_{t-48}, y_{t-168}$ capture short-term inertia, diurnal cycles, and weekly rhythms.
3. **Shifted Rolling Windows**: 6h, 12h, 24h, and 168h rolling means and standard deviations.
   - **Crucial Rule**: To eliminate target leakage, `.shift(1)` is applied before rolling window calculations:
     $$\mu_{t, w} = \frac{1}{w}\sum_{i=1}^{w} y_{t-i}$$
     At time $t$, no information from $y_t$ or subsequent timestamps enters the feature vector.

---

## 4. Modeling Strategy & Evaluation Protocol

### Baseline Model
- **24-Hour Seasonal Naive Persistence**: $\hat{y}_{t} = y_{t-24}$.
- Serves as the realistic hurdle that any viable machine learning model must beat.

### Machine Learning Model
- **Random Forest Regressor**: Ensembles 100 decorrelated decision trees (`max_depth=14`, `min_samples_split=5`).
- Captures complex non-linear interactions between temperature, production schedules, and lagged consumption without requiring feature stationarization.

### Chronological Splitting Protocol
- Traditional randomized K-fold cross-validation is strictly avoided because shuffling future records into the training partition causes catastrophic lookahead bias.
- The dataset is partitioned chronologically: the final 30 days (720 hours) are held out for out-of-sample evaluation.

---

## 5. Recursive Multi-Step Forecasting

For operational planning over the next 24 hours ($t+1, \dots, t+24$), actual future load readings $y_{t+h}$ do not exist.
The forecasting engine implements a **recursive autoregressive loop**:
1. At step $h=1$, predict $\hat{y}_{t+1}$ using observed history up to $t$.
2. Append $\hat{y}_{t+1}$ to the extended history.
3. At step $h=2$, recalculate lag 1 ($=\hat{y}_{t+1}$), lag 2 ($=y_t$), and rolling statistics, then predict $\hat{y}_{t+2}$.
4. Repeat through $h=24$.
5. Future weather and production inputs are supplied by operational schedules or fall back to 24-hour diurnal persistence.

---

## 6. Real Dataset Forecasting & Operational Decision Evaluation Plan

When integrating real-world smart meter datasets (such as UCI ElectricityLoadDiagrams2011–2014):

### 6.1 Sub-Hourly to Hourly Resampling
- Raw 15-minute load readings $P_{15\text{m}, i}$ (in $\text{kW}$) are resampled to 1-hour intervals using mean aggregation:
  $$\bar{P}_{\text{hour}} = \frac{1}{4} \sum_{i=1}^{4} P_{15\text{m}, i}$$
- Because 1 hour of average power $\bar{P}$ (in $\text{kW}$) equals $\bar{P}$ kilowatt-hours ($\text{kWh}$), this directly preserves energy equivalence.

### 6.2 Chronological Splitting & Two-Stage Refit Policy
To balance hyperparameter development against maximizing seasonal training volume without lookahead leakage, the system implements a strict **Two-Stage Chronological Protocol**:

1. **Stage 1 — Model Development & Validation (Pre-Test Split)**:
   - **Training Partition**: `2012-01-01 00:00:00` to `2014-06-30 23:00:00` (21,720 valid samples after 168-hour warm-up dropna).
   - **Validation Partition**: `2014-07-01 00:00:00` to `2014-10-31 23:00:00` (2,952 hourly records / 4 months).
   - **Objective**: Confirm feature stability, lag structure, and tree depth (`max_depth=14`, `n_estimators=100`) strictly on historical data without touching the final holdout test set.
   - *Diagnostic Validation Performance*: MAE = **4.84 kW**, RMSE = **6.54 kW**, WAPE = **4.37%**, $R^2$ = **0.9630**.
   - *Diagnostic Un-Refit Holdout Test*: MAE = **4.27 kW**, RMSE = **6.33 kW**, WAPE = **4.64%**, $R^2$ = **0.9453**.

2. **Stage 2 — Production Refit & Final Holdout Benchmark**:
   - **Combined Pre-Test Training Window**: `2012-01-01 00:00:00` to `2014-10-31 23:00:00` (24,672 valid training samples).
   - **Final Holdout Test Partition**: `2014-11-01 00:00:00` to `2014-12-31 23:00:00` (1,464 hourly records / 61 continuous days).
   - **Protocol**: The confirmed architecture is refit across all pre-test data to integrate the most recent seasonal transitions prior to evaluating the untouched November–December 2014 holdout.
   - **Leakage Audit Guarantee**: November–December 2014 is strictly excluded from all training iterations and hyperparameter decisions. All rolling statistics strictly apply `.shift(1)` to eliminate target leakage.

### 6.3 Empirical Daylight Saving Time (DST) Telemetry Analysis
Auditing the raw 15-minute observations in [`data/raw/LD2011_2014.txt`](../data/raw/LD2011_2014.txt) across all six Portuguese DST transition days (2012–2014) establishes the following empirical facts:

1. **Directly Verified Raw Data Observations**:
   - Every calendar day in the raw dataset contains **exactly 96 quarter-hourly records**, regardless of whether the physical day had 23, 24, or 25 astronomical hours.
   - **Spring Forward (March)**: Nominal 01:00–02:00 readings exist in the file but record the sensor baseline floor (`0.60 kW` on `MT_320`).
   - **Fall Back (October)**: Nominal 01:00–02:00 readings exhibit elevated consumption (~1.52–1.53x vs neighbor hours), indicating that the repeated hour was accumulated into the 01:00 slot.
2. **Calendar-Hour Bins vs. Physical Hours**:
   - The resulting hourly records represent **calendar-hour bins** (local clock hours) rather than uniform 60-minute physical elapsed durations.
   - Potential impact: 2 hours out of 8,760 annual hours (~0.02% of observations) reflect grid logger convention anomalies. Their effect on annual training convergence and holdout evaluation is negligible.
3. **Preservation Policy & Inferred Semantics Disclaimer**:
   - Because the Portuguese grid operator's internal dispatch clock and meter firmware timezone semantics cannot be independently verified from public metadata alone, the pipeline **faithfully preserves observed data as recorded** (4 readings averaged per calendar hour bin) rather than performing speculative interpolation or ad-hoc value division.

### 6.4 Dual-Tier Baselines
Real load profiles exhibit strong weekly shift cycles as well as daily diurnal cycles. Models must benchmark against two distinct baselines:
1. **24-Hour Persistence**: $\hat{y}_{t+h} = y_{t+h-24}$ (daily diurnal naive baseline).
2. **168-Hour Weekly Persistence**: $\hat{y}_{t+h} = y_{t+h-168}$ (predicts Monday 10:00 using the previous Monday 10:00).

### 6.5 One-Step vs. 24-Hour Recursive Separation
- **One-Step-Ahead Protocol ($t+1$)**: Evaluates immediate feature sensitivity assuming observed ground truth up to $t+h-1$. On `MT_320`, achieves **MAE = 4.30 kW**, **RMSE = 6.52 kW**, **WAPE = 4.67%**, and **$R^2 = 0.9419$** (35.2% WAPE reduction vs 24h baseline).
- **Recursive 24-Hour Horizon Protocol**: Evaluates autonomous multi-step dispatch readiness where errors in $\hat{y}_{t+1}$ feed into downstream autoregressive inputs without ground truth. Across 60 rolling origins, achieves aggregate **MAE = 5.17 kW**, **RMSE = 8.87 kW**, **WAPE = 5.61%**, and **$R^2 = 0.8917$**.
  - Error accumulates predictably across the horizon: $h=+1$ (2.14 kW) $\rightarrow$ $h=+6$ (3.99 kW) $\rightarrow$ $h=+12$ (6.16 kW) $\rightarrow$ $h=+18$ (6.68 kW), before stabilizing at $h=+24$ (5.95 kW) as the 24-hour lag re-anchors to observed history.

### 6.6 Operational Decision-Support Use Case (Peak-Demand Alerting)
In industrial utility contracts, electricity bills include both energy consumption charges ($\$/\text{kWh}$) and **peak demand charges** ($\$/\text{kW}$ for the maximum 15-minute or 1-hour demand peak in the billing cycle).
- The forecaster evaluates whether the projected next-24h load exceeds the 90th historical percentile threshold.
- If a peak is detected, the system generates a **Peak Alert Window** (e.g. "Predicted Peak Demand of $380\text{ kW}$ at 14:00–16:00") enabling facility operators to reschedule energy-intensive equipment.

---

## 7. Weather-Aware Energy Forecasting Controlled Experiment

To rigorously evaluate whether external atmospheric conditions provide predictive lift over pure autoregression, the system extends meter `MT_320` with historical ambient temperature data.

### 7.1 Objective & Research Hypothesis
- **Hypothesis**: Does ambient outdoor temperature improve load forecasting accuracy for client `MT_320`?
- **Scientific Protocol**: Treat as a controlled experiment. Do not presuppose that weather improves accuracy, and do not characterize anonymized meter `MT_320` as an industrial facility without empirical confirmation.

### 7.2 Defensible Weather Source & Attribution
- **API Source**: Open-Meteo Historical Weather API (`https://archive-api.open-meteo.com/v1/archive`).
- **Reanalysis Provider**: ECMWF ERA5 reanalysis (Copernicus Climate Change Service).
- **Variable**: Hourly 2-meter air temperature (`temperature_2m`, converted to canonical `temperature_c`).
- **Attribution & Terms**: Creative Commons Attribution 4.0 International (CC BY 4.0).

### 7.3 Geographic Proxy Justification & Limitations
1. **Meter Anonymity**: The official UCI dataset (Trindade et al., INESC Porto / EDP) contains zero client addresses or coordinates to preserve customer privacy.
2. **Selection of Lisbon, Portugal (`38.7223° N`, `-9.1393° W`, elevation ~66m)**:
   - Lisbon serves as the primary national demand center and Portuguese transmission grid anchor.
   - All 26,328 hourly observations (2012-01-01 to 2015-01-01) were ingested and cached to `data/raw/weather_lisbon_2012_2015.json` (0 nulls, mean 16.26 °C, min 1.9 °C, max 37.8 °C).
3. **Proxy Limitations & Uncertainties**:
   - Microclimate variance between Lisbon (coastal Atlantic) and the true unverified meter location (which could be in Porto, Minho, or Alentejo) introduces spatial proxy noise.
   - Lisbon proxy temperatures must **never** be cited as the true facility microclimate.

### 7.4 Experimental Modes: Retrospective vs. Operational
1. **Experiment 1 — Retrospective Weather-Assisted Evaluation (Diagnostic Hindcast)**:
   - Provides realized ERA5 reanalysis temperature for future horizon steps $t+1 \dots t+24$.
   - **Research Purpose**: Determines theoretical maximum correlation under perfect weather foresight.
   - **Operational Status**: Strictly diagnostic research; real operations cannot know realized future weather at forecast origin.
2. **Experiment 2 — Operationally Valid Multi-Step Forecast (Diurnal Persistence Fallback)**:
   - Derives future temperature exclusively from observations available at forecast origin $t$ via 24-hour diurnal persistence ($\hat{T}_{t+h} = T_{t+h-24}$).
   - **Research Purpose**: Simulates operational deployment when day-ahead Numerical Weather Prediction (NWP) archives are unavailable.

### 7.5 Model Architectures & Feature Allocation
- **Baseline v1 (Model A — Pure Autoregressive)**:
  - 24 features: 11 calendar features, 5 autoregressive lags (`lag_1, 2, 24, 48, 168`), 8 rolling statistics (`mean/std` over 6h, 12h, 24h, 168h).
- **Candidate (Model B — Weather-Aware)**:
  - 26 features: All 24 baseline features + `temperature_c` + `temperature_c_lag_1`.
  - Trained using the identical two-stage chronological protocol (`2012-01-08` to `2014-10-31` refit, `2014-11-01` to `2014-12-31` holdout).

### 7.6 Side-by-Side Experimental Findings

| Evaluation Metric | Baseline v1 (Model A: Pure Autoregressive) | Model B (Experiment 1: Retrospective Weather) | Model B (Experiment 2: Operational Persistence) | Delta (Candidate vs Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **One-Step Holdout MAE** | 4.30 kW | 4.29 kW | 4.29 kW | $-0.01\text{ kW}$ ($-0.19\%$) |
| **One-Step Holdout RMSE** | 6.52 kW | 6.58 kW | 6.58 kW | $+0.05\text{ kW}$ ($+0.79\%$) |
| **One-Step Holdout WAPE** | 4.67% | 4.67% | 4.67% | $0.00\%$ |
| **One-Step Holdout $R^2$** | 0.9419 | 0.9410 | 0.9410 | $-0.0009$ |
| **Recursive 24h WAPE** | **5.61%** | 5.56% | 5.64% | $+0.03\%$ (Operational) |
| **Recursive 24h RMSE** | 8.87 kW | 8.78 kW | 8.84 kW | $-0.03\text{ kW}$ |
| **Recursive 24h MAE** | 5.17 kW | 5.12 kW | 5.19 kW | $+0.02\text{ kW}$ (Operational) |

### 7.7 Feature Importance Analysis
Random Forest Gini importance across all 26 features reveals:
- `load_kw_lag_24`: **47.21%** (Primary daily rhythm)
- `load_kw_lag_168`: **37.21%** (Primary weekly cycle)
- `load_kw_lag_1`: **10.44%** (Short-term inertia)
- *Total Autoregressive Lags*: **94.86%**
- `temperature_c`: **0.2689%** (Rank 11 of 26)
- `temperature_c_lag_1`: **0.1693%** (Rank 16 of 26)
- *Total Weather Feature Importance*: **0.4382%**

### 7.8 Scientific Conclusion & Operational Policy Decision
1. **Negligible Predictive Gain**: Meter `MT_320` demand is overwhelmingly dictated by regular business hours, day-of-week routines, and recent consumption inertia ($>94.8\%$ importance). Ambient temperature accounts for less than $0.5\%$ importance and provides zero meaningful reduction in holdout WAPE.
2. **Operational Degradation**: Under realistic operational dispatch with persistence fallback weather, Model B slightly degrades recursive multi-step WAPE from **5.61%** to **5.64%**.
3. **Operational Baseline Retention**:
   - **Model A (Pure Autoregressive)** is officially retained as the production baseline forecaster.
   - **Model B (Weather-Aware)** is documented as an exploratory research experiment.
   - Realized ERA5 weather evaluations are classified as retrospective hindcasts, preventing operational leakage.

---

### 7.9 Audit of Holdout Metric Reproducibility & CSV Serialization Precision

An audit of the holdout test metrics revealed a minor discrepancy between metrics saved in JSON reports and metrics recalculated directly from prediction CSV files:
- **Model A (Baseline v1)**: JSON MAE = **4.2992 kW**, RMSE = **6.5247 kW** vs CSV-recalculated MAE = **4.2991 kW**, RMSE = **6.5246 kW** ($\Delta \approx 0.0001\text{ kW}$).
- **Model B (Weather-Aware)**: JSON MAE = **4.2910 kW**, RMSE = **6.5765 kW** vs CSV-recalculated MAE = **4.2911 kW**, RMSE = **6.5765 kW** ($\Delta \approx 0.0001\text{ kW}$).

#### Mathematical Provenance & Root Cause
1. **Raw In-Memory Array Evaluation (JSON Reports)**:
   During training in [`src/train.py`](../src/train.py) and [`src/weather.py`](../src/weather.py), `evaluate_predictions` receives the unrounded 64-bit floating point model predictions array (`float64`) directly from `model.predict(X_test)`:
   - Model A Full Precision: $\text{MAE} = 4.29919452\text{ kW} \rightarrow \mathbf{4.2992\text{ kW}}$, $\text{RMSE} = 6.52470901\text{ kW} \rightarrow \mathbf{6.5247\text{ kW}}$, $\text{WAPE} = 0.04674535 \rightarrow \mathbf{0.0467}$ ($4.67\%$), $R^2 = 0.94191238 \rightarrow \mathbf{0.9419}$.
   - Model B Full Precision: $\text{MAE} = 4.29103521\text{ kW} \rightarrow \mathbf{4.2910\text{ kW}}$, $\text{RMSE} = 6.57646974\text{ kW} \rightarrow \mathbf{6.5765\text{ kW}}$, $\text{WAPE} = 0.04665663 \rightarrow \mathbf{0.0467}$ ($4.67\%$), $R^2 = 0.94098711 \rightarrow \mathbf{0.9410}$.
   These round exactly to the 4-decimal JSON metrics.

2. **Prediction CSV Serialization**:
   When writing [`outputs/predictions/uci_test_set_predictions.csv`](../outputs/predictions/uci_test_set_predictions.csv) and [`outputs/predictions/uci_weather_test_set_predictions.csv`](../outputs/predictions/uci_weather_test_set_predictions.csv), predicted load values are rounded to 2 decimal places (`np.round(..., 2)`) to keep CSV artifacts compact and clean:
   - Model A CSV Recalculation: $\text{MAE} = 4.29906264\text{ kW} \rightarrow \mathbf{4.2991\text{ kW}}$, $\text{RMSE} = 6.52464738\text{ kW} \rightarrow \mathbf{6.5246\text{ kW}}$, $\text{WAPE} = 0.04674391 \rightarrow \mathbf{0.0467}$, $R^2 = 0.94191348 \rightarrow \mathbf{0.9419}$.
   - Model B CSV Recalculation: $\text{MAE} = 4.29111305\text{ kW} \rightarrow \mathbf{4.2911\text{ kW}}$, $\text{RMSE} = 6.57645748\text{ kW} \rightarrow \mathbf{6.5765\text{ kW}}$, $\text{WAPE} = 0.04665748 \rightarrow \mathbf{0.0467}$, $R^2 = 0.94098733 \rightarrow \mathbf{0.9410}$.

3. **Conclusion & Policy**:
   Both calculations evaluate the exact identical 1,464 holdout rows with identical actual target values. The $\sim 0.0001\text{ kW}$ difference is strictly a representation artifact of 2-decimal rounded prediction text in CSV format versus in-memory 64-bit IEEE 754 floating point precision. Both metrics match within $\pm 0.0002\text{ kW}$ ($< 0.003\%$).

---

### 7.10 Operational Diurnal Persistence Timestamp Semantics & Horizon Alignment

In operational multi-step forecasting without day-ahead Numerical Weather Prediction (NWP), future exogenous temperature is generated via 24-hour diurnal persistence:
$$\hat{T}_{\text{target}}(h) = T_{\text{source}}(h) = T_{\text{target}}(h) - 24\text{ hours}$$

#### Complete 24-Step Temporal Mapping Table
Let $T_{\text{origin}}$ be the forecast origin timestamp (e.g. `2014-11-01 00:00:00`). The 24 future forecast steps map as follows:

| Horizon Step ($h$) | Target Timestamp ($T_{\text{target}}$) | Source Temp Timestamp ($T_{\text{source}}$) | Lag from Target ($T_{\text{target}} - T_{\text{source}}$) | Availability Relative to Origin ($T_{\text{origin}} - T_{\text{source}}$) | Lookahead Leakage Audit |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$h=01$** | `2014-11-01 01:00:00` | `2014-10-31 01:00:00` | $+24\text{ hours}$ | 23 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=02$** | `2014-11-01 02:00:00` | `2014-10-31 02:00:00` | $+24\text{ hours}$ | 22 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=03$** | `2014-11-01 03:00:00` | `2014-10-31 03:00:00` | $+24\text{ hours}$ | 21 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=04$** | `2014-11-01 04:00:00` | `2014-10-31 04:00:00` | $+24\text{ hours}$ | 20 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=05$** | `2014-11-01 05:00:00` | `2014-10-31 05:00:00` | $+24\text{ hours}$ | 19 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=06$** | `2014-11-01 06:00:00` | `2014-10-31 06:00:00` | $+24\text{ hours}$ | 18 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=07$** | `2014-11-01 07:00:00` | `2014-10-31 07:00:00` | $+24\text{ hours}$ | 17 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=08$** | `2014-11-01 08:00:00` | `2014-10-31 08:00:00` | $+24\text{ hours}$ | 16 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=09$** | `2014-11-01 09:00:00` | `2014-10-31 09:00:00` | $+24\text{ hours}$ | 15 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=10$** | `2014-11-01 10:00:00` | `2014-10-31 10:00:00` | $+24\text{ hours}$ | 14 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=11$** | `2014-11-01 11:00:00` | `2014-10-31 11:00:00` | $+24\text{ hours}$ | 13 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=12$** | `2014-11-01 12:00:00` | `2014-10-31 12:00:00` | $+24\text{ hours}$ | 12 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=13$** | `2014-11-01 13:00:00` | `2014-10-31 13:00:00` | $+24\text{ hours}$ | 11 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=14$** | `2014-11-01 14:00:00` | `2014-10-31 14:00:00` | $+24\text{ hours}$ | 10 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=15$** | `2014-11-01 15:00:00` | `2014-10-31 15:00:00` | $+24\text{ hours}$ | 9 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=16$** | `2014-11-01 16:00:00` | `2014-10-31 16:00:00` | $+24\text{ hours}$ | 8 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=17$** | `2014-11-01 17:00:00` | `2014-10-31 17:00:00` | $+24\text{ hours}$ | 7 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=18$** | `2014-11-01 18:00:00` | `2014-10-31 18:00:00` | $+24\text{ hours}$ | 6 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=19$** | `2014-11-01 19:00:00` | `2014-10-31 19:00:00` | $+24\text{ hours}$ | 5 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=20$** | `2014-11-01 20:00:00` | `2014-10-31 20:00:00` | $+24\text{ hours}$ | 4 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=21$** | `2014-11-01 21:00:00` | `2014-10-31 21:00:00` | $+24\text{ hours}$ | 3 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=22$** | `2014-11-01 22:00:00` | `2014-10-31 22:00:00` | $+24\text{ hours}$ | 2 hours before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=23$** | `2014-11-01 23:00:00` | `2014-10-31 23:00:00` | $+24\text{ hours}$ | 1 hour before origin | Zero leakage ($T_{\text{source}} < T_{\text{origin}}$) |
| **$h=24$** | `2014-11-02 00:00:00` | `2014-11-01 00:00:00` | $+24\text{ hours}$ | At origin timestamp | Zero leakage ($T_{\text{source}} = T_{\text{origin}}$) |

#### Resolving Horizon $h=24$ and Interval-Label Semantics
1. **Verbal Slip Resolution**:
   Task 8A informally stated that $h=24$ temperature was drawn from `origin_timestamp - 24 hours`. As proved by the formula above, drawing from `origin - 24 hours` would be $T_{\text{target}} - 48\text{ hours}$ (a 48-hour lag). The actual code implementation in [`src/forecast.py`](../src/forecast.py) (`_derive_default_exogenous`) uses the last 24 historical hours `history_df[TEMPERATURE_COL].iloc[-24:]`, so the 24th step maps to `history_df[TEMPERATURE_COL].iloc[-1]`, which is **$T_{\text{origin}}$** (exactly $T_{\text{target}} - 24\text{ hours}$).

2. **Interval-Beginning Convention & Dispatch Availability**:
   In [`src/public_dataset.py`](../src/public_dataset.py), each hourly row timestamp $T$ represents the start of the hourly interval $[T, T+1\text{h})$, aggregated from four 15-minute readings ($T+15\text{m}, T+30\text{m}, T+45\text{m}, T+60\text{m}$).
   - The row labeled $T_{\text{origin}}$ (e.g. `2014-11-01 00:00:00`) is fully completed and recorded at physical wall-clock time `01:00:00`.
   - Dispatch forecasting takes place after interval $T_{\text{origin}}$ has concluded.
   - Both the realized consumption $y_{T_{\text{origin}}}$ and the hourly ambient temperature $T_{T_{\text{origin}}}$ are finalized and available in historical records at the forecast origin.
   - For step $h=1$ ($01:00:00$), $y_{T_{\text{origin}}}$ serves as lag 1, and the injected weather is from $T_{\text{origin}} - 23\text{h}$ (23 hours prior to origin).
   - For step $h=24$ (`2014-11-02 00:00:00`), the injected diurnal weather is from $T_{\text{origin}}$ (the last finalized interval at origin).
   - Because $T_{\text{source}}(h) \le T_{\text{origin}}$ across all 24 horizon steps, operational diurnal persistence preserves the diurnal cycle with zero lookahead leakage.


