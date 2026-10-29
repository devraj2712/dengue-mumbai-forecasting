import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_lag_engineering_and_eda():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    final_dir = os.path.join(base_dir, 'data', 'final')
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    master_path = os.path.join(final_dir, 'mumbai_dengue_master.csv')
    print(f"Loading master dataset from: {master_path}")
    df = pd.read_csv(master_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print("\n" + "="*60)
    print("STEP 1: LAG ENGINEERING (1 to 8 Weeks)")
    print("="*60)
    
    climate_vars = ['Temp_Mean_C', 'Temp_Max_C', 'Temp_Min_C', 'Rainfall_Cumulative_mm', 'Relative_Humidity_Pct']
    epi_vars = ['dengue_cases', 'dengue_incidence_per_100k', 'malaria_cases', 'malaria_incidence_per_100k']
    
    # 1. Generate Climate Lags (t-1 to t-8)
    for var in climate_vars:
        for lag in range(1, 9):
            df[f"{var}_lag{lag}"] = df[var].shift(lag)
            
    # 2. Generate Autoregressive Epidemiological Lags (t-1 to t-4)
    for var in epi_vars:
        for lag in range(1, 5):
            df[f"{var}_lag{lag}"] = df[var].shift(lag)
            
    # 3. Rolling Window Cumulative / Moving Average Features
    df['Rainfall_2w_sum'] = df['Rainfall_Cumulative_mm'].rolling(window=2).sum()
    df['Rainfall_4w_sum'] = df['Rainfall_Cumulative_mm'].rolling(window=4).sum()
    df['Temp_Mean_2w_avg'] = df['Temp_Mean_C'].rolling(window=2).mean()
    df['Temp_Mean_4w_avg'] = df['Temp_Mean_C'].rolling(window=4).mean()
    df['Relative_Humidity_2w_avg'] = df['Relative_Humidity_Pct'].rolling(window=2).mean()
    df['Relative_Humidity_4w_avg'] = df['Relative_Humidity_Pct'].rolling(window=4).mean()
    
    print(f"Dataset shape after lag feature engineering: {df.shape}")
    
    # Drop initial 8 weeks with NaN due to lag generation
    df_clean = df.dropna().reset_index(drop=True)
    model_ready_path = os.path.join(final_dir, 'mumbai_dengue_model_ready.csv')
    df_clean.to_csv(model_ready_path, index=False)
    print(f"Model-ready dataset saved to: {model_ready_path}")
    print(f"Clean rows available for modeling: {len(df_clean)} weeks")
    
    print("\n" + "="*60)
    print("STEP 2: LAG CROSS-CORRELATION ANALYSIS")
    print("="*60)
    
    lags = list(range(1, 9))
    corr_results = {var: [] for var in climate_vars}
    
    for var in climate_vars:
        for lag in lags:
            corr = df_clean['dengue_cases'].corr(df_clean[f"{var}_lag{lag}"])
            corr_results[var].append(corr)
            
    corr_df = pd.DataFrame(corr_results, index=[f"Lag_{l}w" for l in lags])
    print("\nCross-Correlation Matrix (Climate Lags vs Dengue Cases):")
    print(corr_df.round(4))
    
    # Identify optimal lag for each variable
    print("\nOptimal Lag Windows (Highest Absolute Correlation with Dengue):")
    for var in climate_vars:
        corrs = np.abs(corr_results[var])
        best_lag_idx = np.argmax(corrs)
        best_lag = lags[best_lag_idx]
        actual_corr = corr_results[var][best_lag_idx]
        print(f" - {var:<24}: Optimal Lag = {best_lag} weeks (r = {actual_corr:+.4f})")
        
    print("\n" + "="*60)
    print("STEP 3: GENERATING PUBLICATION-GRADE FIGURES")
    print("="*60)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # FIGURE 1: Epidemiological & Climate Multi-Year Time Series
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    
    # Subplot 1: Dengue & Malaria Cases
    ax1.plot(df['date'], df['dengue_cases'], color='#d62728', lw=2, label='Dengue Cases (Target)')
    ax1.plot(df['date'], df['malaria_cases'], color='#1f77b4', lw=1.5, ls='--', alpha=0.7, label='Malaria Cases (Auxiliary)')
    ax1.set_ylabel('Weekly Reported Cases', fontsize=12, fontweight='bold')
    ax1.set_title('Epidemiological Surveillance Dynamics in Greater Mumbai (2015–2023)', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Rainfall (Cumulative)
    ax2.bar(df['date'], df['Rainfall_Cumulative_mm'], width=6, color='#0288d1', alpha=0.8, label='Weekly Rainfall (mm)')
    ax2.set_ylabel('Rainfall (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('Monsoon Precipitation Regimes', fontsize=12, fontweight='bold', pad=8)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Temperature & Relative Humidity
    color_temp = '#ff7f0e'
    ax3.plot(df['date'], df['Temp_Mean_C'], color=color_temp, lw=1.8, label='Mean Temperature (°C)')
    ax3.set_ylabel('Temperature (°C)', color=color_temp, fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=color_temp)
    
    ax3_twin = ax3.twinx()
    color_hum = '#2ca02c'
    ax3_twin.plot(df['date'], df['Relative_Humidity_Pct'], color=color_hum, lw=1.5, ls=':', label='Relative Humidity (%)')
    ax3_twin.set_ylabel('Relative Humidity (%)', color=color_hum, fontsize=12, fontweight='bold')
    ax3_twin.tick_params(axis='y', labelcolor=color_hum)
    ax3.set_title('Thermal & Humidity Variations', fontsize=12, fontweight='bold', pad=8)
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, '01_epidemiological_climate_timeseries.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # FIGURE 2: Lag Cross-Correlation Curves
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#ff7f0e', '#d62728', '#8c564b', '#0288d1', '#2ca02c']
    
    for i, var in enumerate(climate_vars):
        ax.plot(lags, corr_results[var], marker=markers[i], lw=2.2, color=colors[i], label=var)
        
    ax.axhline(0, color='black', linestyle='--', alpha=0.6)
    ax.set_xlabel('Lag Horizon (Weeks Prior)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pearson Correlation (r) with Dengue Cases', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Correlation Lag Curves: Climate Drivers vs Dengue Outbreaks', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(lags)
    ax.legend(title='Climate Predictor', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, '02_lag_correlation_curves.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # FIGURE 3: Core Feature Correlation Heatmap (Optimal Lags & Targets)
    selected_features = [
        'dengue_cases', 'dengue_incidence_per_100k',
        'malaria_cases', 'malaria_cases_lag2',
        'Rainfall_Cumulative_mm', 'Rainfall_Cumulative_mm_lag4', 'Rainfall_Cumulative_mm_lag6', 'Rainfall_4w_sum',
        'Temp_Mean_C', 'Temp_Mean_C_lag4',
        'Relative_Humidity_Pct', 'Relative_Humidity_Pct_lag4'
    ]
    
    corr_matrix = df_clean[selected_features].corr(method='pearson')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        vmin=-0.5, 
        vmax=1.0, 
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )
    plt.title('Correlation Heatmap: Dengue Targets & Key Lagged Climate Features', fontsize=14, fontweight='bold', pad=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, '03_feature_correlation_heatmap.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    print("\nLag Engineering and EDA successfully finished!")

if __name__ == '__main__':
    perform_lag_engineering_and_eda()
