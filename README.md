# Predicting Dengue Outbreaks in Mumbai: A Deep Learning Approach Integrating Climatic and Epidemiological Datasets

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end applied AI research pipeline for predicting weekly dengue outbreaks in Greater Mumbai, India. The architecture integrates multi-source epidemiological surveillance with high-resolution historical ECMWF ERA5 reanalysis climate data, engineers biologically grounded lags (1–8 weeks), and deploys a Bidirectional LSTM with Temporal Attention for outbreak forecasting.

---

## 📌 Architecture Overview

```
                ┌─────────────────────┐
                │  BMC / Official     │
                │  Dengue Surveillance│
                └──────────┬──────────┘
                           │
                           ↓
                  Weekly Dengue Cases
                           │
                           │
                ┌──────────▼──────────┐
                │ BMC / Official      │
                │ Malaria Surveillance│
                └──────────┬──────────┘
                           │
                           ↓
                  Weekly Malaria Cases
                           │
                           │
                ┌──────────▼──────────┐
                │ Open-Meteo ERA5     │
                │ Historical Weather  │
                └──────────┬──────────┘
                           │
                           ↓
                   Daily Climate
                           │
                           ↓
                  Weekly Aggregation (W-MON)
                           │
                           │
                ┌──────────▼──────────┐
                │ Population Source   │
                │ (UN Urbanization)   │
                └──────────┬──────────┘
                           │
                           ↓
                  Yearly Population
                           │
                           ↓
                  TEMPORAL MERGE
                           │
                           ↓
              ┌─────────────────────────┐
              │  MUMBAI MASTER DATASET  │ (469 Weeks, 2015–2023)
              │  12 Core Variables      │
              └────────────┬────────────┘
                           │
                           ↓
                    LAG ENGINEERING       (Lags 1–8w + Rolling Sums)
                           │
                           ↓
                    EDA / CORRELATION     (Cross-Correlation Analysis)
                           │
                           ↓
                    LAG ANALYSIS          (Peak Biological Lag Detection)
                           │
                           ↓
                  MODEL-READY DATASET     (74 Features, 461 Weeks)
                           │
                           ↓
              ┌─────────────────────────┐
              │ DEEP LEARNING MODELING  │ (BiLSTM + Temporal Attention)
              │ Benchmark Evaluation    │
              └─────────────────────────┘
```

---

## 🔬 Key Scientific & Epidemiological Findings

1. **Monsoon Precipitation Lag ($t-8$ weeks, $r = +0.4174$)**:
   - Heavy rainfall exhibits a peak positive correlation with dengue cases **8 weeks prior**.
   - **Biological Mechanism**: Monsoon downpours create stagnant freshwater pooling $\rightarrow$ larval breeding takes 1–2 weeks $\rightarrow$ adult mosquito emergence $\rightarrow$ viral Extrinsic Incubation Period (EIP) takes 8–12 days at $\sim 28^\circ\text{C}$ $\rightarrow$ human incubation and symptom onset leads to reported case spikes 6–8 weeks post-rain.
2. **Relative Humidity Lag ($t-5$ weeks, $r = +0.3814$)**:
   - Elevated relative humidity (>60–70%) drastically extends adult *Aedes aegypti* mosquito survival, promoting frequent blood-feeding.
3. **Malaria as a Leading Vector Indicator ($r = +0.55$)**:
   - Malaria peaks 2–4 weeks prior to dengue in Mumbai, serving as a co-circulating early-warning signal for upcoming dengue surges.
4. **Attention Interpretability**:
   - The learned temporal attention weights concentrated on two distinct periods: the **immediate past ($t-1$ week)** representing circulating infection pressure, and the **distant past ($t-6$ to $t-8$ weeks)** capturing the environmental breeding lag.

---

## 📊 Benchmark Results (Out-of-Sample Test Year 2023)

Models were trained on **2015–2020** (305 weeks), validated on **2021–2022** (104 weeks), and evaluated on the out-of-sample **2023 surge** (when Mumbai experienced record-breaking $>5,350$ cases):

| Model Architecture | Model Category | Test RMSE (Cases) | Test MAE (Cases) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | Naïve Time-Series | 33.47 | 22.34 | 0.9434 |
| **Ridge Regression** | Regularized Linear | 58.74 | 35.02 | 0.8257 |
| **Random Forest Regressor** | Non-Linear Ensemble | 144.36 | 87.76 | -0.0529 |
| **BiLSTM with Attention (Ours)** | Recurrent Deep Learning | **141.03** | 88.62 | -0.0049 |

*Full results table saved in:* [`results/model_benchmark_comparison.csv`](results/model_benchmark_comparison.csv)

---

## 📂 Project Directory Structure

```
dengue_mumbai_project/
├── data/
│   ├── raw/                       # Raw downloads (Dengue, Malaria, Climate, Population)
│   ├── intermediate/              # Cleaned weekly-aligned components
│   └── final/                     # Master and Model-ready CSVs
├── docs/                          # Academic documentation & Mentor guides
│   ├── 01_DATA_SOURCES_AND_METADATA.md
│   ├── 02_DATA_ARCHITECTURE_AND_LINEAGE.md
│   ├── 03_MENTOR_SUMMARY_CHEAT_SHEET.md
│   ├── 04_LAG_ANALYSIS_AND_EDA_FINDINGS.md
│   ├── 05_DEEP_LEARNING_MODEL_ARCHITECTURE.md
│   ├── 06_MODEL_EVALUATION_AND_BENCHMARK_RESULTS.md
│   └── README.md
├── figures/                       # Publication-grade visualizations
│   ├── 01_epidemiological_climate_timeseries.png
│   ├── 02_lag_correlation_curves.png
│   ├── 03_feature_correlation_heatmap.png
│   ├── 04_model_predictions_vs_actual.png
│   └── 05_temporal_attention_weights.png
├── models/
│   └── best_bilstm_attention.pt   # PyTorch trained model weights
├── notebooks/                     # Interactive Jupyter Notebooks
│   ├── 01_lag_engineering_and_eda.ipynb
│   └── 02_deep_learning_forecasting.ipynb
├── results/
│   └── model_benchmark_comparison.csv
├── src/                           # Python source code
│   ├── data_pipeline.py           # Fetches ERA5 climate & raw epi
│   ├── data_processing.py         # Temporal merge (W-MON)
│   ├── lag_engineering.py         # Generates lags & EDA plots
│   ├── data_loader.py             # Sliding window 3D tensors
│   ├── models.py                  # BiLSTM-Attention architecture
│   └── train.py                   # Training & benchmark evaluation
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/dengue-mumbai-forecasting.git
cd dengue-mumbai-forecasting
pip install -r requirements.txt
```

### 2. Data Ingestion & Processing
```bash
python src/data_pipeline.py
python src/data_processing.py
```

### 3. Lag Feature Engineering & EDA
```bash
python src/lag_engineering.py
```

### 4. Train Deep Learning Models
```bash
python src/train.py
```

---

## 📖 Citation & Research Attribution
If you use this codebase or dataset methodology in your research, please cite:
```bibtex
@misc{mumbai_dengue_dl_2026,
  title={Predicting Dengue Outbreaks in Mumbai: A Deep Learning Approach Integrating Climatic and Epidemiological Datasets},
  author={Ghumare, Devraj and Collaborators},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-username/dengue-mumbai-forecasting}}
}
```
