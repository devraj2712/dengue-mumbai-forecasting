import pandas as pd
import os

def process_and_merge_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    inter_dir = os.path.join(base_dir, 'data', 'intermediate')
    final_dir = os.path.join(base_dir, 'data', 'final')
    
    print("Loading raw datasets...")
    df_dengue = pd.read_csv(os.path.join(raw_dir, 'dengue_bmc_raw.csv'))
    df_malaria = pd.read_csv(os.path.join(raw_dir, 'malaria_bmc_raw.csv'))
    df_climate = pd.read_csv(os.path.join(raw_dir, 'climate_openmeteo_raw.csv'))
    df_pop = pd.read_csv(os.path.join(raw_dir, 'population_raw.csv'))
    
    # 1. Process Epidemiological Data (already weekly, just need to ensure date format)
    df_dengue['date'] = pd.to_datetime(df_dengue['date'])
    df_malaria['date'] = pd.to_datetime(df_malaria['date'])
    
    # Save to intermediate
    df_dengue.to_csv(os.path.join(inter_dir, 'dengue_weekly.csv'), index=False)
    df_malaria.to_csv(os.path.join(inter_dir, 'malaria_weekly.csv'), index=False)
    
    # 2. Process Climate Data (Daily -> Weekly)
    df_climate['date'] = pd.to_datetime(df_climate['date'])
    # Set date as index to resample
    df_climate.set_index('date', inplace=True)
    
    # Resample to weekly (starting Monday to align with epi data)
    df_climate_weekly = df_climate.resample('W-MON').agg({
        'Temp_Max_C': 'mean',
        'Temp_Min_C': 'mean',
        'Temp_Mean_C': 'mean',
        'Rainfall_Cumulative_mm': 'sum',
        'Relative_Humidity_Pct': 'mean'
    }).reset_index()
    
    df_climate_weekly.to_csv(os.path.join(inter_dir, 'climate_weekly.csv'), index=False)
    
    # Save daily cleaned just in case
    df_climate.reset_index().to_csv(os.path.join(inter_dir, 'climate_daily_clean.csv'), index=False)
    print("Intermediate datasets saved.")
    
    # 3. Temporal Merge
    print("Merging datasets into Master Dataset...")
    master = pd.merge(df_dengue, df_malaria, on='date', how='outer')
    master = pd.merge(master, df_climate_weekly, on='date', how='left')
    
    # Add population data (yearly -> weekly mapping)
    master['year'] = master['date'].dt.year
    master = pd.merge(master, df_pop, on='year', how='left')
    
    # 4. Feature Engineering (Incidence rates)
    master['dengue_incidence_per_100k'] = (master['dengue_cases'] / master['population']) * 100000
    master['malaria_incidence_per_100k'] = (master['malaria_cases'] / master['population']) * 100000
    
    # Reorder columns
    master = master[['date', 'year', 'population', 'Temp_Mean_C', 'Temp_Max_C', 'Temp_Min_C', 
                     'Rainfall_Cumulative_mm', 'Relative_Humidity_Pct', 'malaria_cases', 'malaria_incidence_per_100k', 
                     'dengue_cases', 'dengue_incidence_per_100k']]
    
    # Fill NAs if any at the edges
    master.fillna(method='ffill', inplace=True)
    master.fillna(method='bfill', inplace=True)
    
    # 5. Save Final Master Dataset
    master.to_csv(os.path.join(final_dir, 'mumbai_dengue_master.csv'), index=False)
    print(f"Master Dataset saved to {os.path.join(final_dir, 'mumbai_dengue_master.csv')}")
    print(f"Master Dataset shape: {master.shape}")

if __name__ == "__main__":
    process_and_merge_data()
