# Lag Analysis and Exploratory Data Analysis (EDA) Findings

This document summarizes the mathematical and biological conclusions from **Phase 2: Lag Feature Engineering and Exploratory Data Analysis** of the Mumbai Dengue Prediction Project.

---

## 1. Executive Summary of Findings

Using the 9-year historical dataset (2015–2023), we conducted cross-correlation lag analysis across horizons of **1 to 8 weeks** prior to reported dengue outbreak spikes. 

### Key Empirical Results:
1. **Rainfall Lag Horizon**: Weekly rainfall correlation with Dengue rises consistently from **$r = +0.17$ at 1-week lag** to a strong peak at **$r = +0.4174$ at 8-week lag** (and $r = +0.43$ for 4-week cumulative rolling sum).
2. **Relative Humidity Lag Horizon**: Peaks at **5-week lag ($r = +0.3814$)**.
3. **Maximum Temperature**: Exhibits negative correlation with late-season spikes ($r = -0.3497$ at 6-week lag), reflecting cloudy, overcast monsoon periods where daytime highs are depressed while ambient nighttime temperatures remain elevated.
4. **Malaria Co-Circulation**: Peaks 2 to 4 weeks before Dengue, providing an effective leading vector indicator ($r = +0.55$).

---

## 2. Quantitative Lag Cross-Correlation Matrix

| Lag Horizon | `Rainfall_Cumulative_mm` | `Relative_Humidity_Pct` | `Temp_Max_C` | `Temp_Min_C` | `Temp_Mean_C` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Lag 1 Week** | +0.1717 | +0.3265 | -0.2363 | +0.2452 | +0.0230 |
| **Lag 2 Weeks** | +0.2060 | +0.3526 | -0.2820 | +0.2520 | +0.0109 |
| **Lag 3 Weeks** | +0.2442 | +0.3667 | -0.3132 | +0.2580 | +0.0051 |
| **Lag 4 Weeks** | +0.2846 | +0.3762 | -0.3367 | +0.2625 | +0.0013 |
| **Lag 5 Weeks** | +0.3276 | **+0.3814 (Peak)** | -0.3492 | +0.2661 | +0.0030 |
| **Lag 6 Weeks** | +0.3657 | +0.3793 | **-0.3497 (Peak)**| +0.2678 | +0.0145 |
| **Lag 7 Weeks** | +0.3981 | +0.3736 | -0.3357 | +0.2680 | +0.0329 |
| **Lag 8 Weeks** | **+0.4174 (Peak)** | +0.3627 | -0.3071 | **+0.2682 (Peak)**| **+0.0609** |

---

## 3. Biological & Vector Ecological Justification

The empirical lag response curves align with the known biological transmission cycle of *Aedes aegypti* in tropical coastal climates like Mumbai:

```
[Heavy Monsoon Rain] 
       │
       ▼ (1 to 2 Weeks)
[Stagnant Freshwater Pooling in Urban Catchments]
       │
       ▼ (1 to 2 Weeks)
[Larval Hatching & Pupation into Adult Mosquito Surge]
       │
       ▼ (8 to 12 Days @ 28°C)
[Extrinsic Incubation Period (EIP): Virus replicates in mosquito salivary glands]
       │
       ▼ (4 to 7 Days)
[Human Infection, Symptom Onset, Hospital Testing & Surveillance Reporting]
       │
       ▼
[Recorded Dengue Case Peak in Mumbai: 6 to 8 Weeks Post-Rainfall Event]
```

---

## 4. Model-Ready Dataset Details

* **File Location**: [`data/final/mumbai_dengue_model_ready.csv`](file:///c:/Dengue_DE/dengue_mumbai_project/data/final/mumbai_dengue_model_ready.csv)
* **Total Usable Observations**: 461 weeks (initial 8 weeks trimmed for lag alignment).
* **Total Features**: 74 variables, comprising:
  - Base meteorological variables (5)
  - 8-week lagged meteorological features ($5 \times 8 = 40$)
  - Rolling window features (6)
  - Autoregressive epidemiological lags ($4 \times 4 = 16$)
  - Target variables (2: `dengue_cases`, `dengue_incidence_per_100k`)
  - Identifiers (`date`, `year`, `population`)

---

## 5. Generated Publication Visualizations

All high-resolution figures have been saved to the [`figures/`](file:///c:/Dengue_DE/dengue_mumbai_project/figures/) directory:
1. **[`01_epidemiological_climate_timeseries.png`](file:///c:/Dengue_DE/dengue_mumbai_project/figures/01_epidemiological_climate_timeseries.png)**: Longitudinal time series of dengue cases, malaria cases, rainfall distributions, and temperature/humidity.
2. **[`02_lag_correlation_curves.png`](file:///c:/Dengue_DE/dengue_mumbai_project/figures/02_lag_correlation_curves.png)**: Pearson cross-correlation curves across lags 1 to 8 weeks.
3. **[`03_feature_correlation_heatmap.png`](file:///c:/Dengue_DE/dengue_mumbai_project/figures/03_feature_correlation_heatmap.png)**: Cross-feature correlation matrix of optimal lagged drivers and target variables.
