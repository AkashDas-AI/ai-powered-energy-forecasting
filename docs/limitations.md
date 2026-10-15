# Known Limitations & Engineering Assumptions

This document explicitly outlines the analytical boundaries, physical assumptions, and known limitations of the **AI-Powered Energy Consumption Forecasting System**.

---

## 1. Synthetic Data Limitations

- **Simplified Operational Reality**: The dataset is generated through physical and heuristic equations. While it models diurnal curves, production shifts, and thermal HVAC demand, it does not reflect unanticipated real-world industrial anomalies such as sudden equipment failures, voltage sags, power outages, or major factory retooling.
- **Stationary Production Rules**: Real facilities experience shifting order books, overtime shifts, and seasonal shutdowns that deviate from fixed calendar rules.

---

## 2. Exogenous Forecasting Assumptions

- **Future Temperature Availability**: Real-world operations require numerical weather prediction (NWP) feeds. In the absence of an external weather API, the current forecaster relies on a **24-hour diurnal persistence assumption** ($T_{t+h} = T_{t+h-24}$). During rapid cold fronts or heatwaves, this assumption introduces forecast error.
- **Production Schedules**: Future production status is assumed to be known or deterministic. Deviations in actual shift operations directly impact the forecast accuracy.

---

## 3. Cold Start & Historical Lookback Constraints

- **Minimum Warm-Up Requirement**: The model utilizes 168-hour (7-day) autoregressive lags and rolling statistics. Consequently, a facility must possess at least **168 hours of uninterrupted historical data** before the inference engine can generate its first forecast.
- **Data Gap Recovery**: Extended sensor outages exceeding several days cannot be reliably reconstructed through linear interpolation and will require specialized seasonal imputation models.

---

## 4. Error Compounding in Recursive Multi-Step Forecasting

- In recursive multi-step forecasting ($h=1, \dots, 24$), prediction $\hat{y}_{t+1}$ is fed back into the feature matrix as the lag for step $t+2$.
- Any prediction error in early forecast steps ($h=1, 2$) can propagate and compound into later horizons ($h=20, \dots, 24$), leading to potential drift.

---

## 5. Point Forecasting vs. Probabilistic Intervals

- The current implementation delivers deterministic point predictions ($\mathbb{E}[y_{t+h}]$).
- It does not yet generate probabilistic prediction intervals (e.g. 10th and 90th quantile bounds), which are valuable for operational risk management and reserve spinning capacity decisions in utility markets.

---

## 6. Public Dataset Limitations (UCI ElectricityLoadDiagrams2011–2014)

- **Anonymized Consumer Classifications**: The 370 client columns (`MT_001` through `MT_370`) represent Portuguese smart meters. However, official UCI metadata does not distinguish between industrial plants, commercial buildings, agricultural loads, or residential apartments. While `MT_320` is selected for high active load and continuity, specific industrial facility claims cannot be made without external utility metadata.
- **Absence of Environmental Exogenous Variables**: The dataset contains no outdoor temperature, humidity, wind, or irradiance features. Temperature-dependent HVAC load swings cannot be directly attributed to ambient conditions from this file alone.
- **Daylight Saving Time Artifacts**: As documented by the data author, March 23-hour days record zeroes between 01:00 and 02:00, while October 25-hour days aggregate two hours into a single timestamp. While our pipeline applies interpolation and distribution adjustments, these represent heuristic approximations.
- **Heterogeneous Connection Dates**: Client installations occurred asynchronously throughout 2011–2014, requiring explicit connection window detection to prevent corrupting training sets with uninstalled zero sequences.

