import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sliding_sequences(features, targets, seq_length=8, horizon=1):
    """
    Creates 3D tensors: (num_samples, seq_length, num_features)
    Target: value at (t + horizon - 1)
    """
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_length - horizon + 1):
        X_seq.append(features[i : i + seq_length])
        y_seq.append(targets[i + seq_length + horizon - 1])
    return np.array(X_seq), np.array(y_seq)

def prepare_forecasting_data(seq_length=8, horizon=1, batch_size=16):
    """
    Loads model-ready data, performs chronological train/val/test splits,
    fits scalers strictly on training set, and returns DataLoaders & metadata.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'final', 'mumbai_dengue_model_ready.csv')
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Chronological Split (No data leakage across future events)
    train_mask = df['year'] <= 2020
    val_mask = (df['year'] >= 2021) & (df['year'] <= 2022)
    test_mask = df['year'] == 2023
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    # Exclude metadata columns from predictors
    meta_cols = ['date', 'year', 'population']
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in ['dengue_cases', 'dengue_incidence_per_100k']]
    target_col = 'dengue_cases'
    
    print(f"Features count for deep learning: {len(feature_cols)}")
    print(f"Train weeks: {len(train_df)} (2015-2020)")
    print(f"Validation weeks: {len(val_df)} (2021-2022)")
    print(f"Test weeks: {len(test_df)} (2023 - Out-of-sample)")
    
    # Fit scalers ONLY on training data to prevent lookahead data leakage
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols].values)
    y_train_scaled = scaler_y.fit_transform(train_df[[target_col]].values).flatten()
    
    X_val_scaled = scaler_X.transform(val_df[feature_cols].values)
    y_val_scaled = scaler_y.transform(val_df[[target_col]].values).flatten()
    
    X_test_scaled = scaler_X.transform(test_df[feature_cols].values)
    y_test_scaled = scaler_y.transform(test_df[[target_col]].values).flatten()
    
    # Create 3D temporal sequences (Sliding window length = seq_length)
    X_train_seq, y_train_seq = create_sliding_sequences(X_train_scaled, y_train_scaled, seq_length, horizon)
    X_val_seq, y_val_seq = create_sliding_sequences(X_val_scaled, y_val_scaled, seq_length, horizon)
    X_test_seq, y_test_seq = create_sliding_sequences(X_test_scaled, y_test_scaled, seq_length, horizon)
    
    print(f"Train tensor shape: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"Val tensor shape:   X={X_val_seq.shape}, y={y_val_seq.shape}")
    print(f"Test tensor shape:  X={X_test_seq.shape}, y={y_test_seq.shape}")
    
    # Build PyTorch DataLoaders
    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
    val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
    test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    metadata = {
        'feature_cols': feature_cols,
        'target_col': target_col,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'seq_length': seq_length,
        'horizon': horizon,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }
    
    return train_loader, val_loader, test_loader, metadata

if __name__ == '__main__':
    train_loader, val_loader, test_loader, meta = prepare_forecasting_data()
    print("DataLoader test successful!")
