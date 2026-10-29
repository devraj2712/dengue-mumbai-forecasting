# Data Sources, Variables, and Metadata

This document provides a detailed specification of all raw and processed datasets used in the research project **"Predicting Dengue Outbreaks in Mumbai: A Deep Learning Approach Integrating Climatic and Epidemiological Datasets"**.

---

## 1. Primary Datasets Overview

The project integrates **four distinct data streams** spanning 9 years (**2015 – 2023**, total **469 weeks**):

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA SOURCES                                    │
├───────────────────┬──────────────────────┬──────────────────┬───────────────────┤
│   Epidemiological │    Co-Circulating    │  Meteorological  │    Demographic    │
│      (Dengue)     │       (Malaria)      │     (ERA5)       │    (Population)   │
├───────────────────┼──────────────────────┼──────────────────┼───────────────────┤
│ BMC Epidemiology  │ BMC Surveillance     │ Open-Meteo ERA5  │ UN Urbanization   │
│ Cell / OpenDengue │ Cell / NCVBDC        │ Reanalysis       │ Prospects         │
└───────────────────┴──────────────────────┴──────────────────┴───────────────────┘
```

---

## 2. Dataset Specifications

### Dataset A: Dengue Surveillance Data (Target Variable)
* **Description**: Weekly recorded cases of Dengue virus infection (*Flaviviridae*) in Mumbai.
* **Source**: Brihanmumbai Municipal Corporation (BMC) Epidemiology Cell & National Center for Vector Borne Diseases Control (NCVBDC) / OpenDengue.
* **Primary File**: [`data/raw/dengue_bmc_raw.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/raw/dengue_bmc_raw.csv)
* **Processed File**: [`data/intermediate/dengue_weekly.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/intermediate/dengue_weekly.csv)
* **Temporal Resolution**: Weekly (ISO Week, Monday start).
* **Time Range**: Jan 2015 – Dec 2023 (469 records).
* **Variables**:
  * `date` (YYYY-MM-DD): Start date of the surveillance week.
  * `dengue_cases` (Integer): Confirmed laboratory / clinical reported cases of Dengue.

---

### Dataset B: Malaria Surveillance Data (Auxiliary Co-Infection Feature)
* **Description**: Weekly recorded cases of Malaria (*Plasmodium vivax / Plasmodium falciparum*) in Mumbai.
* **Source**: BMC Public Health Department & NCVBDC.
* **Primary File**: [`data/raw/malaria_bmc_raw.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/raw/malaria_bmc_raw.csv)
* **Processed File**: [`data/intermediate/malaria_weekly.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/intermediate/malaria_weekly.csv)
* **Temporal Resolution**: Weekly (ISO Week, Monday start).
* **Time Range**: Jan 2015 – Dec 2023.
* **Variables**:
  * `date` (YYYY-MM-DD): Start date of the surveillance week.
  * `malaria_cases` (Integer): Confirmed malaria cases.
* **Scientific Rationale**: Malaria shares vector ecological niches and peaks 2–4 weeks prior to Dengue in Mumbai during the early monsoon season, serving as an auxiliary predictive feature.

---

### Dataset C: Meteorological / Climate Data (Exogenous Predictors)
* **Description**: Daily atmospheric weather data for Greater Mumbai.
* **Source**: [Open-Meteo ERA5 Historical Archive API](https://open-meteo.com/en/docs/historical-weather-api) (ECMWF ERA5 Reanalysis).
* **Coordinates**: Latitude `19.0760° N`, Longitude `72.8777° E` (Center of Greater Mumbai).
* **Elevation**: ~14m above sea level.
* **Timezone**: `Asia/Kolkata` (UTC+05:30).
* **Primary File**: [`data/raw/climate_openmeteo_raw.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/raw/climate_openmeteo_raw.csv)
* **Processed Files**: 
  * [`data/intermediate/climate_daily_clean.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/intermediate/climate_daily_clean.csv) (Daily)
  * [`data/intermediate/climate_weekly.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/intermediate/climate_weekly.csv) (Weekly Aggregated)
* **Variables & Specifications**:

| Variable Name in Master | Raw API Variable | Unit | Aggregation Method (Daily $\rightarrow$ Weekly) | Biological & Vector Relevance |
| :--- | :--- | :--- | :--- | :--- |
| `Temp_Mean_C` | `temperature_2m_mean` | °C | 7-day Average | Governs viral replication rate (Extrinsic Incubation Period, EIP). |
| `Temp_Max_C` | `temperature_2m_max` | °C | 7-day Average | Thermal threshold limit for adult mosquito survival. |
| `Temp_Min_C` | `temperature_2m_min` | °C | 7-day Average | Minimum thermal limit for larval development. |
| `Rainfall_Cumulative_mm` | `precipitation_sum` | mm | 7-day Sum | Creates stagnant freshwater breeding sites for *Aedes aegypti*. |
| `Relative_Humidity_Pct` | `relative_humidity_2m` | % | 7-day Average of Hourly | Higher humidity (>60%) extends adult mosquito lifespan and feeding frequency. |

---

### Dataset D: Demographic Data (Population Baseline)
* **Description**: Annual human population estimates for Greater Mumbai Urban Agglomeration.
* **Source**: United Nations World Urbanization Prospects & Census of India Projections.
* **Primary File**: [`data/raw/population_raw.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/raw/population_raw.csv)
* **Time Range**: 2015 – 2023 (Annual).
* **Values**:
  * 2015: 19,397,000
  * 2016: 19,688,000
  * 2017: 19,980,000
  * 2018: 20,185,000
  * 2019: 20,411,000
  * 2020: 20,667,000
  * 2021: 20,961,000
  * 2022: 21,296,000
  * 2023: 21,673,000
* **Usage**: Normalizes case counts into **Incidence per 100,000 population**:
  $$\text{Incidence Rate} = \left( \frac{\text{Cases}}{\text{Population}} \right) \times 100,000$$

---

## 3. Final Master Dataset Feature Dictionary

* **Master File Path**: [`data/final/mumbai_dengue_master.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/final/mumbai_dengue_master.csv)
* **Total Rows**: 469 weekly records (2015-W01 to 2023-W52).
* **Total Columns**: 12.

| # | Column Name | Data Type | Description |
| :---: | :--- | :---: | :--- |
| 1 | `date` | String / Date | Start date of the surveillance week (YYYY-MM-DD, Monday) |
| 2 | `year` | Integer | Calendar year (2015–2023) |
| 3 | `population` | Integer | Estimated population for Greater Mumbai in that year |
| 4 | `Temp_Mean_C` | Float | Weekly mean temperature in °C |
| 5 | `Temp_Max_C` | Float | Weekly average maximum temperature in °C |
| 6 | `Temp_Min_C` | Float | Weekly average minimum temperature in °C |
| 7 | `Rainfall_Cumulative_mm` | Float | Total weekly precipitation in millimeters |
| 8 | `Relative_Humidity_Pct` | Float | Weekly mean relative humidity percentage |
| 9 | `malaria_cases` | Integer | Total weekly reported malaria cases |
| 10 | `malaria_incidence_per_100k` | Float | Malaria cases per 100,000 population |
| 11 | `dengue_cases` | Integer | **Target Variable**: Total weekly reported dengue cases |
| 12 | `dengue_incidence_per_100k` | Float | **Normalized Target**: Dengue cases per 100,000 population |
