"""Synthetic data generation module for facility energy consumption.

Simulates hourly electricity consumption for a mid-sized industrial/commercial facility.
The simulation accounts for:
- Base load (continuous refrigeration, security, IT servers).
- Operating shifts and production schedules (weekday shifts vs. weekend idle).
- Ambient temperature-dependent HVAC loads (cooling and heating demands).
- Modest stochastic variations with guaranteed non-negative constraints.

NOTE: This dataset is entirely synthetic and created for educational, modeling,
and benchmarking purposes. It must not be presented as measured industrial sensor data.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    RANDOM_SEED,
    TIMESTAMP_COL,
    TARGET_COL,
    TEMPERATURE_COL,
    PRODUCTION_COL,
)


def generate_synthetic_energy_data(
    start_date: str = "2023-01-01 00:00:00",
    num_days: int = 365 * 2,  # 2 years of hourly data (17,520 hours)
    seed: int = RANDOM_SEED,
    output_path: Optional[Union[str, Path]] = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Generate realistic hourly electricity load and exogenous weather/production data.

    Args:
        start_date: Starting ISO timestamp string.
        num_days: Duration in calendar days to simulate.
        seed: Random seed for exact numerical reproducibility.
        output_path: Destination CSV path. If None, file will not be written.

    Returns:
        pd.DataFrame containing simulated hourly records with columns:
        ['timestamp', 'load_kw', 'temperature_c', 'production_status']
    """
    rng = np.random.default_rng(seed)
    total_hours = num_days * 24

    # 1. Continuous hourly timestamp range
    timestamps = pd.date_range(start=start_date, periods=total_hours, freq="h")
    df = pd.DataFrame({TIMESTAMP_COL: timestamps})

    # Temporal features for physics/schedule simulation
    day_of_year = df[TIMESTAMP_COL].dt.dayofyear.values
    hour_of_day = df[TIMESTAMP_COL].dt.hour.values
    day_of_week = df[TIMESTAMP_COL].dt.dayofweek.values  # 0=Monday, 6=Sunday

    # 2. Ambient Temperature Simulation (°C)
    # Seasonal wave: Mean 16°C, winter trough ~3°C, summer peak ~29°C
    seasonal_temp = 16.0 + 13.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    # Diurnal wave: Peak at 15:00, trough at 05:00
    diurnal_temp = 5.5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
    # Autoregressive weather noise
    weather_noise = rng.normal(0, 1.8, size=total_hours)
    temperature_c = np.round(seasonal_temp + diurnal_temp + weather_noise, 2)
    df[TEMPERATURE_COL] = temperature_c

    # 3. Production Schedule (0 = Off/Standby, 1 = Active Production)
    # Weekdays (0-4): Main shift 07:00 to 19:00
    # Saturday (5): Half-day morning shift 08:00 to 14:00
    # Sunday (6): Facility shut down / idle
    production_status = np.zeros(total_hours, dtype=int)
    weekday_active = (day_of_week < 5) & (hour_of_day >= 7) & (hour_of_day < 19)
    saturday_active = (day_of_week == 5) & (hour_of_day >= 8) & (hour_of_day < 14)
    production_status[weekday_active | saturday_active] = 1
    df[PRODUCTION_COL] = production_status

    # 4. Energy Load Synthesis (kW)
    # Component A: Continuous baseload (IT, lighting, security, continuous pumps)
    baseload_kw = 80.0

    # Component B: Production machinery load (active during production hours)
    # Varies slightly by hour with a ramp-up and ramp-down profile
    machinery_load_kw = np.zeros(total_hours)
    machinery_load_kw[production_status == 1] = 220.0 + 35.0 * np.sin(
        np.pi * (hour_of_day[production_status == 1] - 7) / 12.0
    )

    # Component C: HVAC cooling and heating demand
    # Cooling activates when temperature > 21°C
    cooling_load = np.maximum(0.0, temperature_c - 21.0) * 7.5
    # Heating activates when temperature < 11°C
    heating_load = np.maximum(0.0, 11.0 - temperature_c) * 5.2
    hvac_load_kw = cooling_load + heating_load

    # Component D: Modest stochastic operational fluctuations
    stochastic_noise = rng.normal(0, 7.5, size=total_hours)

    # Total aggregate load with strict non-negative clamping
    raw_load_kw = baseload_kw + machinery_load_kw + hvac_load_kw + stochastic_noise
    # Baseline physical lower bound of 20 kW for this facility
    load_kw = np.maximum(raw_load_kw, 20.0)
    df[TARGET_COL] = np.round(load_kw, 2)

    # 5. Persist to CSV if requested
    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)

    return df


if __name__ == "__main__":
    generated_df = generate_synthetic_energy_data()
    print(f"Generated synthetic energy dataset with {len(generated_df)} hourly records.")
    print(f"Target file: {RAW_DATA_PATH}")
    print(generated_df.head())

