# Data Storage & Management Guide

This directory manages datasets for the **AI-Powered Energy Consumption Forecasting System**.

---

## 1. Directory Structure

- **`data/raw/`**: Contains raw, uncleaned data files (e.g. `synthetic_facility_load.csv`). Data in this directory is treated as immutable source input.
- **`data/processed/`**: Contains cleaned, validated, and time-aligned datasets (e.g. `cleaned_facility_load.csv`) with imputed flags and normalized columns, ready for feature engineering and modeling.

---

## 2. Dataset Provenance & Synthetic Disclaimer

> [!IMPORTANT]
> The primary dataset used in this project is **synthetic**, generated to simulate the physical load characteristics of an industrial/commercial facility. It incorporates base electrical loads, weekday/weekend production schedules, ambient temperature-dependent HVAC demands, and stochastic fluctuations. It is **not** measured industrial sensor data.

---

## 3. How to Regenerate Data

You can regenerate the raw synthetic dataset at any time using the project CLI:

```bash
# Generate 2 years (730 days / 17,520 hours) of synthetic facility data
python main.py --generate-data --days 730 --seed 42

# Preprocess and validate the newly generated data
python main.py --preprocess
```

---

## 4. Public Dataset Ingestion & Third-Party Licensing

- **UCI ElectricityLoadDiagrams20112014**: Real smart meter load diagrams from the UCI Machine Learning Repository (Trindade, 2015, DOI: [10.24432/C58C86](https://doi.org/10.24432/C58C86)), licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Raw files (`data/raw/LD2011_2014.txt`, 678 MB) are excluded from Git tracking and can be fetched via `python main.py --download-uci`.
- **Open-Meteo Weather Reanalysis**: Historical ambient temperature data (ECMWF ERA5) licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Important**: Third-party datasets are subject to CC BY 4.0 terms and are NOT relicensed under the project's MIT license.

---

## 5. Git Commitment Policy

In accordance with best data science practices and the repository `.gitignore` configuration:
- Raw and processed datasets (`.csv`, `.txt`, `.zip`, `.json`) are excluded from Git tracking to avoid repository bloat and potential leakage.
- Only directory structure tracking files (`.gitkeep`) are committed.
- Anyone cloning this repository can cleanly recreate the synthetic data using the deterministic seed (`seed=42`), or fetch the public dataset via the documented CLI command.

