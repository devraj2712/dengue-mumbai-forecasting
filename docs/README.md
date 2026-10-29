# Project Documentation Directory

Welcome to the documentation folder for the research project: **"Predicting Dengue Outbreaks in Mumbai: A Deep Learning Approach Integrating Climatic and Epidemiological Datasets"**.

This folder contains comprehensive, publication-grade documentation organized for code maintainability and academic presentation to mentors and collaborators.

---

## Documentation Index

1. **[`01_DATA_SOURCES_AND_METADATA.md`](file:///c:/Dengue_DE/dengue_mumbai_project/docs/01_DATA_SOURCES_AND_METADATA.md)**
   * Complete inventory of all datasets (Dengue, Malaria, Open-Meteo ERA5 Climate, UN Population).
   * Exact API parameters, spatial coordinates (`19.0760° N, 72.8777° E`), temporal bounds (2015–2023), and units.
   * Full feature dictionary for the 12 columns in `mumbai_dengue_master.csv`.

2. **[`02_DATA_ARCHITECTURE_AND_LINEAGE.md`](file:///c:/Dengue_DE/dengue_mumbai_project/docs/02_DATA_ARCHITECTURE_AND_LINEAGE.md)**
   * Architectural data flow diagrams (Mermaid) mapping `raw/` $\rightarrow$ `intermediate/` $\rightarrow$ `final/`.
   * Explanation of ISO week Monday-alignment (`W-MON`) for temporal merging.
   * Biological and vector-ecological rationale linking climate drivers (temperature, humidity, rain) to mosquito breeding cycles and disease outbreaks.

3. **[`03_MENTOR_SUMMARY_CHEAT_SHEET.md`](file:///c:/Dengue_DE/dengue_mumbai_project/docs/03_MENTOR_SUMMARY_CHEAT_SHEET.md)**
   * An executive 30-second summary tailored for showing to your research mentor/professor.
   * Quick fact table and anticipated Q&A for academic reviews.

4. **[`04_LAG_ANALYSIS_AND_EDA_FINDINGS.md`](file:///c:/Dengue_DE/dengue_mumbai_project/docs/04_LAG_ANALYSIS_AND_EDA_FINDINGS.md)**
   * Quantitative cross-correlation results across 1 to 8 weeks.
   * Biological justification for the 6–8 week rainfall lag and 5-week humidity lag.
   * Documentation of generated publication figures and the model-ready dataset.

5. **[`05_DEEP_LEARNING_MODEL_ARCHITECTURE.md`](file:///c:/Dengue_DE/dengue_mumbai_project/docs/05_DEEP_LEARNING_MODEL_ARCHITECTURE.md)**
   * Mathematical formulation of sliding window sequence-to-point forecasting.
   * Chronological train (2015–2020), validation (2021–2022), and out-of-sample test (2023) split methodology.
   * BiLSTM + Temporal Attention architecture, Huber loss function, and benchmark evaluation protocols.

6. **[`06_MODEL_EVALUATION_AND_BENCHMARK_RESULTS.md`](file:///c:/Dengue_DE/dengue_mumbai_project/docs/06_MODEL_EVALUATION_AND_BENCHMARK_RESULTS.md)**
   * Out-of-sample benchmark comparison table (Persistence, Ridge, Random Forest, BiLSTM-Attention).
   * Discussion of the 2023 extreme outbreak surge and regime adaptation.
   * Interpretability analysis of learned temporal attention weights across the 8-week lookback horizon.
