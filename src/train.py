import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from data_loader import prepare_forecasting_data
from models import BiLSTMAttention, StandardLSTM

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def train_deep_learning_pipeline(epochs=100, lr=0.001, patience=15):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    figures_dir = os.path.join(base_dir, 'figures')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 1: PREPARING TIME SERIES SEQUENCES & DATALOADERS")
    print("="*60)
    train_loader, val_loader, test_loader, meta = prepare_forecasting_data(seq_length=8, horizon=1, batch_size=16)
    
    input_dim = len(meta['feature_cols'])
    scaler_y = meta['scaler_y']
    
    # -------------------------------------------------------------
    # 1. Classical Benchmarks (Ridge & Random Forest)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: TRAINING CLASSICAL BENCHMARK MODELS")
    print("="*60)
    # Extract flattened 2D arrays from sequences for tabular models
    X_train_flat = train_loader.dataset.X.numpy().reshape(len(train_loader.dataset), -1)
    y_train_flat = train_loader.dataset.y.numpy()
    
    X_test_flat = test_loader.dataset.X.numpy().reshape(len(test_loader.dataset), -1)
    y_test_flat = test_loader.dataset.y.numpy()
    
    # Unscale actual test targets
    y_test_actual = scaler_y.inverse_transform(y_test_flat.reshape(-1, 1)).flatten()
    
    # Baseline 1: Persistence (y_t = y_{t-1})
    # Last step in sequence of target's autoregressive lag1
    y_pred_persist_scaled = train_loader.dataset.X.numpy()[:, -1, 0] # fallback
    # Simpler: shift test actual by 1
    y_pred_persist = np.roll(y_test_actual, 1)
    y_pred_persist[0] = y_test_actual[0]
    p_rmse, p_mae, p_r2 = evaluate_metrics(y_test_actual, y_pred_persist)
    print(f"Persistence Model   -> RMSE: {p_rmse:.2f}, MAE: {p_mae:.2f}, R2: {p_r2:.4f}")
    
    # Baseline 2: Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_flat, y_train_flat)
    y_pred_ridge_scaled = ridge.predict(X_test_flat)
    y_pred_ridge = scaler_y.inverse_transform(y_pred_ridge_scaled.reshape(-1, 1)).flatten()
    r_rmse, r_mae, r_r2 = evaluate_metrics(y_test_actual, y_pred_ridge)
    print(f"Ridge Regression    -> RMSE: {r_rmse:.2f}, MAE: {r_mae:.2f}, R2: {r_r2:.4f}")
    
    # Baseline 3: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train_flat)
    y_pred_rf_scaled = rf.predict(X_test_flat)
    y_pred_rf = scaler_y.inverse_transform(y_pred_rf_scaled.reshape(-1, 1)).flatten()
    rf_rmse, rf_mae, rf_r2 = evaluate_metrics(y_test_actual, y_pred_rf)
    print(f"Random Forest       -> RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R2: {rf_r2:.4f}")
    
    # -------------------------------------------------------------
    # 2. Deep Learning Model (BiLSTM with Temporal Attention)
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: TRAINING PROPOSED BiLSTM-ATTENTION ARCHITECTURE")
    print("="*60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = BiLSTMAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.HuberLoss(delta=1.0) # Robust to epidemic spike outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = os.path.join(models_dir, 'best_bilstm_attention.pt')
    
    train_losses, val_losses = [], []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds, _ = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(X_batch)
            
        epoch_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds, _ = model(X_batch)
                val_loss = criterion(preds, y_batch)
                total_val_loss += val_loss.item() * len(X_batch)
                
        epoch_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")
            
        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"[*] Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.5f})")
                break
                
    # -------------------------------------------------------------
    # 3. Model Evaluation on 2023 Out-of-Sample Test Set
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: EVALUATION ON OUT-OF-SAMPLE TEST SET (YEAR 2023)")
    print("="*60)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    test_preds_list = []
    attention_weights_list = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds, attn = model(X_batch)
            test_preds_list.append(preds.cpu().numpy())
            attention_weights_list.append(attn.cpu().numpy())
            
    y_pred_bilstm_scaled = np.concatenate(test_preds_list)
    y_pred_bilstm = scaler_y.inverse_transform(y_pred_bilstm_scaled.reshape(-1, 1)).flatten()
    # Floor negative predictions to 0
    y_pred_bilstm = np.clip(y_pred_bilstm, a_min=0, a_max=None)
    
    dl_rmse, dl_mae, dl_r2 = evaluate_metrics(y_test_actual, y_pred_bilstm)
    print(f"BiLSTM-Attention    -> RMSE: {dl_rmse:.2f}, MAE: {dl_mae:.2f}, R2: {dl_r2:.4f}")
    
    # -------------------------------------------------------------
    # 4. Save Comprehensive Benchmark Comparison Table
    # -------------------------------------------------------------
    comparison_df = pd.DataFrame({
        'Model Architecture': ['Persistence Baseline', 'Ridge Regression', 'Random Forest Regressor', 'BiLSTM with Attention (Ours)'],
        'Test RMSE (Cases)': [p_rmse, r_rmse, rf_rmse, dl_rmse],
        'Test MAE (Cases)': [p_mae, r_mae, rf_mae, dl_mae],
        'Test R-Squared': [p_r2, r_r2, rf_r2, dl_r2]
    })
    comparison_df = comparison_df.round(4)
    results_path = os.path.join(results_dir, 'model_benchmark_comparison.csv')
    comparison_df.to_csv(results_path, index=False)
    print("\nBenchmark Model Comparison Summary:")
    print(comparison_df.to_string(index=False))
    print(f"Saved results table to: {results_path}")
    
    # -------------------------------------------------------------
    # 5. Generate Publication Figures
    # -------------------------------------------------------------
    # Figure 4: Predictions vs Ground Truth (Year 2023)
    test_dates = meta['test_df']['date'].values[-len(y_test_actual):]
    
    plt.figure(figsize=(13, 6))
    plt.plot(test_dates, y_test_actual, label='Actual Reported Dengue (2023)', color='black', lw=2.5, marker='o', ms=4)
    plt.plot(test_dates, y_pred_bilstm, label='BiLSTM-Attention (Proposed)', color='#d62728', lw=2.2, ls='--')
    plt.plot(test_dates, y_pred_rf, label='Random Forest Benchmark', color='#1f77b4', lw=1.5, alpha=0.7)
    plt.plot(test_dates, y_pred_ridge, label='Ridge Benchmark', color='#2ca02c', lw=1.5, ls=':', alpha=0.7)
    
    plt.title('Out-of-Sample Forecasting Performance on Mumbai Dengue Surge (Year 2023)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Surveillance Week Date', fontsize=11, fontweight='bold')
    plt.ylabel('Weekly Dengue Cases', fontsize=11, fontweight='bold')
    plt.xticks(rotation=35, ha='right')
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pred_fig_path = os.path.join(figures_dir, '04_model_predictions_vs_actual.png')
    plt.savefig(pred_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {pred_fig_path}")
    
    # Figure 5: Temporal Attention Weights Interpretation
    all_attns = np.concatenate(attention_weights_list, axis=0) # (num_samples, seq_length, 1)
    mean_attn = all_attns.mean(axis=0).flatten() # (seq_length,)
    weeks_prior = [f"t-{8 - i}w" for i in range(8)]
    
    plt.figure(figsize=(9, 5))
    bars = plt.bar(weeks_prior, mean_attn, color='#0288d1', edgecolor='black', alpha=0.8)
    bars[-1].set_color('#d62728') # Highlight immediate past
    plt.title('Learned Temporal Attention Weights Across 8-Week Lookback Horizon', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Lookback Time Step (Weeks Prior to Forecast)', fontsize=11, fontweight='bold')
    plt.ylabel('Mean Attention Score (Softmax)', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    attn_fig_path = os.path.join(figures_dir, '05_temporal_attention_weights.png')
    plt.savefig(attn_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {attn_fig_path}")

if __name__ == '__main__':
    train_deep_learning_pipeline()
