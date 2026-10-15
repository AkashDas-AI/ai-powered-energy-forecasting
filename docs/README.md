# AI-Powered Energy Consumption Forecasting — Technical Documentation Hub

Welcome to the central technical documentation repository for the **AI-Powered Energy Consumption Forecasting** system. This folder serves as the single source of truth for technical specifications, architecture diagrams, data dictionaries, model dynamics, and verification procedures.

---

## 📁 Documentation Index

| Document | Purpose & Scope |
| :--- | :--- |
| [**Public Dataset Selection**](dataset_selection.md) | Official candidate dataset evaluation, comparative matrix, selection rationale, and integration plan. |
| [**Architecture & Data Flow**](architecture.md) | High-level system architecture, component breakdown, and end-to-end data lifecycle. |
| [**Data Dictionary**](data_dictionary.md) | Detailed schema definition, physical units (kW, kWh, °C), datatypes, and valid numeric ranges. |
| [**Methodology & Approach**](methodology.md) | Simulation dynamics, leakage prevention, chronological splitting, and recursive inference mechanics. |
| [**Known Limitations**](limitations.md) | Boundary conditions, cold-start warm-up constraints, and exogenous assumptions. |
| [**Validation & Testing Guide**](validation_guide.md) | Automated testing protocols (`pytest`), quality gates, and anti-leakage verification routines. |
| [**Dataset Specification**](dataset_spec.md) | Real-world benchmark specifications (PJM, UCI) and preprocessing protocols. |
| [**Model Specification & Learning**](model_spec.md) | Mathematical formulation of tree-based and recurrent sequence models, loss functions, and optimization. |
| [**Module Reference**](module_reference.md) | Codebase package structure, module responsibilities, class/function contracts, and dependencies. |

---

## 🔄 Project Documentation Protocol

Every implementation step:
1. **Module & Document Alignment**: Code changes and technical decisions are mirrored in this folder.
2. **Progress Reports**: Each milestone is accompanied by a concise implementation brief.
3. **Verification Artefacts**: Includes executable test commands and observed outputs before completion.
