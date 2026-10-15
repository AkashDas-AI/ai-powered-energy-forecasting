# Technical Documentation Hub

Welcome to the central technical documentation repository for the **AI-Powered Energy Consumption Forecasting** system. This folder serves as the single source of truth for system architecture, component specifications, mathematical methodology, data dictionaries, verification protocols, and interview preparation.

---

## Documentation Index

| Document | Purpose & Scope |
| :--- | :--- |
| [**Interview & Technical Mastery Guide**](interview_guide.md) | **Primary study & review guide**: Elevator pitches, architecture deep dive, module breakdown, benchmark scorecards, weather experiment insights, and 25 comprehensive interview Q&As. |
| [**Architecture & Data Flow**](architecture.md) | High-level system architecture diagram, multi-target pipeline flow, end-to-end data lifecycle, and artifact storage topology. |
| [**Module Reference & API Specification**](module_reference.md) | Detailed technical reference of all production modules in `src/`, public functions, input/output contracts, and test modules in `tests/`. |
| [**Methodology & Approach**](methodology.md) | Simulation physics, zero-leakage mathematical proofs, two-stage chronological split protocol, recursive multi-step mechanics, and Portuguese DST empirical analysis. |
| [**Data Dictionary**](data_dictionary.md) | Detailed schema definition, physical units (kW, kWh, °C), datatypes, and valid numeric ranges for raw, processed, and forecast outputs. |
| [**Public Dataset Selection**](dataset_selection.md) | Candidate dataset comparative evaluation matrix, selection rationale (UCI ElectricityLoadDiagrams vs. Tetouan vs. PJM), and integration plan. |
| [**Known Limitations & Assumptions**](limitations.md) | Boundary conditions, single-meter scope, anonymized metadata caveats, cold-start warm-up constraints, and future roadmap. |
| [**Validation & Verification Guide**](validation_guide.md) | Automated testing protocols (48 unit tests across 6 modules), quality gates, mathematical anti-leakage tests, and Baseline v1 SHA256 checksum verification. |

---

## Project Documentation Protocol

1. **Module & Document Alignment**: All code changes, schema modifications, and modeling decisions are mirrored in this technical documentation folder.
2. **Mathematical Precision**: Power ($\text{kW}$) and energy ($\text{kWh}$) are mathematically distinguished throughout all documentation and pipelines.
3. **Reproducibility Guarantee**: Every benchmark metric is accompanied by exact replication commands, seed controls, and verified file checksums.
