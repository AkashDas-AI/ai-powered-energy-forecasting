# Data Dictionary

This document defines all schema attributes, column descriptions, datatypes, physical units, and allowable value ranges used across the **AI-Powered Energy Consumption Forecasting System**, covering both the baseline synthetic generator and the proposed canonical public dataset interface.

---

## 1. Raw Dataset Schema (`data/raw/synthetic_facility_load.csv`)

| Column Name | Data Type | Physical Unit | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns]` | ISO 8601 (Hourly) | YYYY-MM-DD HH:00 | Hourly timestamp marking the beginning of the observation interval. |
| `load_kw` | `float64` | Kilowatts (kW) | $\ge 20.0$ kW | Total facility electrical power demand averaged over the 1-hour interval. |
| `temperature_c` | `float64` | Celsius (°C) | $-10.0$ to $+45.0$ °C | Ambient outdoor dry-bulb temperature affecting HVAC heating/cooling loads. |
| `production_status` | `int64` | Binary Flag | `0` or `1` | Operational state of the industrial production line (0 = Standby/Off, 1 = Active Shift). |

---

## 2. Processed Dataset Schema (`data/processed/cleaned_facility_load.csv`)

Inherits all raw columns and appends data cleaning audit metadata:

| Column Name | Data Type | Physical Unit | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns]` | ISO 8601 (Hourly) | Strictly monotonic | Complete, uninterrupted hourly sequence. |
| `load_kw` | `float64` | Kilowatts (kW) | $\ge 20.0$ kW | Validated numeric consumption with missing values interpolated. |
| `temperature_c` | `float64` | Celsius (°C) | Real scalar | Cleaned and interpolated ambient temperature. |
| `production_status` | `int64` | Binary Flag | `0` or `1` | Cleaned production schedule indicator. |
| `is_imputed` | `bool` | Boolean | `True` / `False` | Audit marker indicating whether this observation was synthesized to fill a missing interval. |

---

## 3. Engineered Feature Matrix (Model Inputs)

Generated in [`src/features.py`](../src/features.py):

| Feature Category | Column Name | Transformation / Formula | Range / Units |
| :--- | :--- | :--- | :--- |
| **Calendar** | `hour` | `dt.hour` | 0 – 23 |
| **Calendar** | `day_of_week` | `dt.dayofweek` (0=Monday, 6=Sunday) | 0 – 6 |
| **Calendar** | `day_of_month` | `dt.day` | 1 – 31 |
| **Calendar** | `month` | `dt.month` | 1 – 12 |
| **Calendar** | `is_weekend` | `1` if day $\ge 5$ else `0` | 0 or 1 |
| **Cyclical** | `hour_sin`, `hour_cos` | $\sin / \cos(2\pi \cdot \text{hour} / 24)$ | $[-1.0, 1.0]$ |
| **Cyclical** | `day_sin`, `day_cos` | $\sin / \cos(2\pi \cdot \text{day} / 7)$ | $[-1.0, 1.0]$ |
| **Cyclical** | `month_sin`, `month_cos`| $\sin / \cos(2\pi \cdot (\text{month}-1) / 12)$ | $[-1.0, 1.0]$ |
| **Autoregressive Lags** | `load_kw_lag_1` | $y_{t-1}$ (prior hour load) | $\ge 0$ kW |
| **Autoregressive Lags** | `load_kw_lag_2` | $y_{t-2}$ (2 hours prior) | $\ge 0$ kW |
| **Autoregressive Lags** | `load_kw_lag_24` | $y_{t-24}$ (exact prior day same hour) | $\ge 0$ kW |
| **Autoregressive Lags** | `load_kw_lag_48` | $y_{t-48}$ (2 days prior same hour) | $\ge 0$ kW |
| **Autoregressive Lags** | `load_kw_lag_168` | $y_{t-168}$ (prior week same hour) | $\ge 0$ kW |
| **Exogenous Lags** | `temperature_c_lag_1` | $\text{temp}_{t-1}$ (prior hour temperature, if present) | °C |
| **Rolling Windows** | `load_kw_rolling_mean_{6,12,24,168}h` | Rolling mean shifted by 1 hour | $\ge 0$ kW |
| **Rolling Windows** | `load_kw_rolling_std_{6,12,24,168}h` | Rolling standard deviation shifted by 1 | $\ge 0$ kW |

---

## 4. Forecast Output Schema (`outputs/predictions/next_24h_forecast.csv`)

| Column Name | Data Type | Physical Unit | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns]` | ISO 8601 (Hourly) | Future hourly forecast timestamps (next 24 hours). |
| `predicted_load_kw` | `float64` | Kilowatts (kW) | Predicted average electrical power demand. |
| `predicted_energy_kwh` | `float64` | Kilowatt-Hours (kWh) | Predicted cumulative energy consumption ($1.0\text{ kW avg} \times 1\text{h} \equiv 1.0\text{ kWh}$). |
| `temperature_c` | `float64` | Celsius (°C) | Future ambient temperature input (optional/persistence fallback). |
| `production_status` | `int64` | Binary Flag | Future production schedule state (optional). |

---

## 5. Canonical Multi-Dataset Interface (Proposed Real Dataset Representation)

To ingest diverse public real-world datasets without breaking the existing pipeline, the following canonical representation is defined:

| Canonical Field | Type | Mandatory? | Physical Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `datetime64[ns]` | **Yes** | ISO 8601 (UTC / Fixed Local) | Uninterrupted regular hourly timestamp. |
| `entity_id` | `str` | **Yes** | Alphanumeric String | Meter, building, or customer identifier (e.g. `'MT_320'`, `'facility_01'`). |
| `load_kw` | `float64` | **Yes** | Kilowatts ($\text{kW}$) | Average active power demand across the hourly interval. |
| `temperature_c` | `float64` | *No (Optional)* | Celsius (°C) | Ambient outdoor dry-bulb temperature. |
| `production_status` | `int64` | *No (Optional)* | Binary (`0` or `1`) | Operational shift state. |
| `is_imputed` | `bool` | **Yes** | Boolean | Audit flag for missing value interpolation. |
| `source_dataset` | `str` | **Yes** | Name / Citation Tag | Provenance identifier (e.g., `'UCI_LD2011_2014'`, `'Synthetic_Facility'`). |

---

## 6. Source Schema Mappings to Canonical Format

### A. UCI ElectricityLoadDiagrams20112014 (`LD2011_2014.txt`)
- **Raw Delimiter / Format**: Semicolon (`;`), comma decimal (`,`), 15-minute intervals.
- **Timestamp Column**: First column (string `'yyyy-mm-dd hh:mm:ss'`) $\rightarrow$ parsed to `datetime64[ns]`.
- **Target Value Mapping**: Column `MT_xxx` (in 15-min $\text{kW}$) $\rightarrow$ 4 intervals per hour aggregated via arithmetic mean:
  $$\text{load\_kw}_{\text{hourly}} = \frac{1}{4}\sum_{i=1}^4 \text{MT\_xxx}_i$$
- **`entity_id`**: Set to `'MT_xxx'` (e.g., `'MT_320'`).
- **Anonymization Caveat**: `MT_320` represents an anonymized Portuguese grid electricity consumer with ~102 kW mean demand. It is not confirmed as an industrial facility in official metadata and must be documented as empirical smart-meter telemetry.
- **Exogenous Columns**: Omitted / handled in pure autoregressive mode without synthetic fabrication.
- **`source_dataset`**: `'UCI_Electricity_MT_320'`.

### B. UCI Power Consumption of Tetouan City (`Tetuan City power consumption.csv`)
- **Raw Delimiter / Format**: Comma (`,`), standard dot decimal, 10-minute intervals.
- **Timestamp Column**: `DateTime` $\rightarrow$ parsed to `datetime64[ns]`.
- **Target Value Mapping**: `Zone 1 Power Consumption` $\rightarrow$ 6 intervals per hour averaged:
  $$\text{load\_kw}_{\text{hourly}} = \frac{1}{6}\sum_{i=1}^6 \text{Zone\_1}_i$$
- **Exogenous Columns**: `Temperature` $\rightarrow$ `temperature_c` (mean of 6 readings per hour).
- **`entity_id`**: `'Tetouan_Zone_1'`.
- **`source_dataset`**: `'UCI_Tetouan_Zone1'`.

---

## 7. Weather Reanalysis & Enriched Dataset Schemas

### A. Raw Weather Cache (`data/raw/weather_lisbon_2012_2015.json`)
- **Source**: Open-Meteo Historical Weather API (ERA5 Reanalysis).
- **Location**: Lisbon, Portugal (`38.7223° N`, `-9.1393° W`, elevation 66m).
- **Timezone**: `Europe/Lisbon`.
- **Fields**:
  - `hourly.time`: ISO 8601 strings (`YYYY-MM-DDTHH:MM`), 26,328 observations.
  - `hourly.temperature_2m`: Ambient air temperature at 2 meters elevation in Celsius (°C).

### B. Processed Weather-Enriched Electricity Dataset (`data/processed/uci_electricity_weather_hourly.csv`)
- **Row Count**: Exactly 26,304 hourly records (`2012-01-01 00:00:00` to `2014-12-31 23:00:00`).
- **Columns**:
  - `timestamp` (`datetime64[ns]`): Monotonic hourly interval.
  - `entity_id` (`str`): Client identifier (`MT_320`).
  - `load_kw` (`float64`): Arithmetic mean power over the hour in kilowatts.
  - `is_imputed` (`bool`): Imputation flag.
  - `source_dataset` (`str`): `'UCI_Electricity_MT_320'`.
  - `temperature_c` (`float64`): Hourly ambient temperature from Lisbon ERA5 proxy (°C, range 1.9 to 37.8 °C, 0 nulls).

---

## 8. Weather Experiment Artifact Schemas

### A. Test Set Predictions (`outputs/predictions/uci_weather_test_set_predictions.csv`)
- **Row Count**: 1,464 records (`2014-11-01 00:00:00` to `2014-12-31 23:00:00`).
- **Columns**:
  - `timestamp`: Out-of-sample holdout timestamp.
  - `actual_load_kw`: True measured electrical load.
  - `model_b_pred_kw`: Model B (weather-aware) one-step prediction in kW.
  - `error_model_b_kw`: Prediction error ($\hat{y}_t - y_t$).
  - `temperature_c`: Observed proxy temperature at step $t$.

### B. Next-24h Forecast (`outputs/predictions/uci_weather_next_24h_forecast.csv`)
- **Row Count**: 24 records (`2015-01-01 00:00:00` to `2015-01-01 23:00:00`).
- **Columns**:
  - `timestamp`: Target forecast timestamp.
  - `predicted_load_kw`: Model B recursive multi-step load forecast in kW.
  - `predicted_energy_kwh`: Model B recursive multi-step energy forecast in kWh.
  - `temperature_c`: Ambient temperature (from 2015-01-01 reanalysis cache or diurnal persistence).

### C. Metrics JSON (`outputs/metrics/uci_weather_model_metrics.json`)
- Comprehensive diagnostic dictionary documenting:
  - Geographic proxy metadata (Lisbon coordinates, license, uncertainties).
  - Stage 1 validation metrics (July–October 2014).
  - Stage 2 holdout metrics (November–December 2014).
  - All 26 feature importances ranked (`load_kw_lag_24` 47.21%, `temperature_c` 0.27%).
  - Experiment 1 (Retrospective weather recursive 24h WAPE = 5.56%).
  - Experiment 2 (Operational persistence weather recursive 24h WAPE = 5.64%).
  - Side-by-side comparison table against Baseline v1 (Model A).

### D. Multi-Step Recursive Predictions (`outputs/predictions/uci_weather_recursive_predictions.csv`)
- **Row Count**: Exactly 2,880 records (60 rolling forecast origins $\times$ 24 hourly steps $\times$ 2 evaluation modes).
- **Columns**:
  - `origin_timestamp`: Historical forecast issuance origin (every 24h from `2014-11-01 00:00:00` to `2014-12-30 00:00:00`).
  - `target_timestamp`: Predicted hourly target timestamp ($t+1 \dots t+24$).
  - `horizon_step`: Integer step index $h \in [1, 24]$.
  - `actual_load_kw`: True measured electrical load in kW.
  - `predicted_load_kw`: Model B recursive prediction in kW.
  - `weather_input_mode`: Mode tag: `'retrospective_reanalysis'` (1,440 rows) or `'operational_persistence_fallback'` (1,440 rows).


