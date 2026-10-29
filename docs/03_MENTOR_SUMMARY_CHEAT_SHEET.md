# Research Mentor Presentation Cheat Sheet & Q&A

This document provides a concise, high-level summary designed to help you present the project's data architecture and data collection strategy to your research mentor or advisor.

---

## 1. Executive Summary (The 30-Second Pitch)

> "We have built an end-to-end data acquisition and integration pipeline for predicting dengue outbreaks in Mumbai over a 9-year study period (2015–2023, 469 weeks). We integrated four data streams: target Dengue surveillance, auxiliary Malaria co-infection data, 5 core ERA5 climate variables (temperature min/max/mean, cumulative rainfall, relative humidity) from Open-Meteo, and UN population projections. Daily weather is aggregated to weekly Monday-aligned blocks (`W-MON`) to match disease reporting cycles, and raw case counts are normalized into incidence rates per 100,000 population."

---

## 2. Quick Fact Sheet for Mentor Review

| Question | Answer |
| :--- | :--- |
| **Project Title** | Predicting Dengue Outbreaks in Mumbai: A Deep Learning Approach Integrating Climatic and Epidemiological Datasets |
| **Study Region** | Greater Mumbai, Maharashtra, India (`19.0760° N, 72.8777° E`) |
| **Study Period** | 2015 – 2023 (9 Years, 469 Weekly Observations) |
| **Target Variable** | Weekly Dengue Cases & Dengue Incidence Rate per 100,000 |
| **Exogenous Predictors** | `Temp_Mean_C`, `Temp_Max_C`, `Temp_Min_C`, `Rainfall_Cumulative_mm`, `Relative_Humidity_Pct` |
| **Auxiliary Predictors** | `malaria_cases`, `malaria_incidence_per_100k`, `population` |
| **Primary Data Source (Climate)** | Open-Meteo ERA5 Reanalysis API (ECMWF) |
| **Primary Data Source (Health)** | BMC Public Health Dept / NCVBDC / OpenDengue |
| **Master Dataset Path** | [`data/final/mumbai_dengue_master.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/final/mumbai_dengue_master.csv) |

---

## 3. Key Questions Your Mentor Might Ask (and How to Answer)

### Q1: "How did you solve the temporal mismatch between daily weather and weekly disease data?"
* **Answer**: "We used ISO standard weekly resampling (`W-MON`, Monday start). For state variables like temperature and relative humidity, we computed 7-day moving means. For flux variables like rainfall, we computed 7-day cumulative sums. This aligns perfectly with municipal surveillance reporting weeks."

### Q2: "Why did you include Malaria data in a Dengue forecasting model?"
* **Answer**: "In Mumbai, both diseases are vector-borne and co-circulate during the monsoon. Malaria peaks slightly earlier (July–August) than Dengue (August–October). Including Malaria gives our deep learning model an auxiliary early-warning signal regarding vector population surges and healthcare system load."

### Q3: "Why did you normalize cases per 100,000 population?"
* **Answer**: "Mumbai's population grew by over 2.2 million between 2015 and 2023. Raw case counts alone would introduce demographic bias into a long-term model. Normalizing to incidence rate per 100,000 decouples population growth from epidemiological outbreak severity."

### Q4: "Why are climate variables critical for Dengue deep learning models?"
* **Answer**: "Vector biology strictly dictates dengue transmission. Rainfall provides mosquito breeding sites, temperature controls mosquito development speed and viral replication (Extrinsic Incubation Period), and relative humidity determines adult mosquito survival. These relationships involve biological lags of 2 to 8 weeks, which our downstream Deep Learning models (LSTM/Transformer) will capture."

---

## 4. Master Data File Structure

```
dengue_mumbai_project/
├── data/
│   ├── raw/                       <-- Original raw files
│   │   ├── dengue_bmc_raw.csv
│   │   ├── malaria_bmc_raw.csv
│   │   ├── climate_openmeteo_raw.csv
│   │   └── population_raw.csv
│   ├── intermediate/              <-- Cleaned weekly components
│   │   ├── climate_daily_clean.csv
│   │   ├── climate_weekly.csv
│   │   ├── dengue_weekly.csv
│   │   └── malaria_weekly.csv
│   └── final/                     <-- ML-Ready Master Dataset
│       └── mumbai_dengue_master.csv
├── docs/                          <-- Complete Documentation Folder
│   ├── 01_DATA_SOURCES_AND_METADATA.md
│   ├── 02_DATA_ARCHITECTURE_AND_LINEAGE.md
│   ├── 03_MENTOR_SUMMARY_CHEAT_SHEET.md
│   └── README.md
├── src/                           <-- Automated Pipeline Scripts
│   ├── data_pipeline.py
│   └── data_processing.py
```
