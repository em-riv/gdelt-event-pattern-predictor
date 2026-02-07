"""
Update Production Models and Generate Predictions
This script:
1. Trains both XGBoost and GRU models on the latest data
2. Saves models to disk
3. Generates predictions for all countries and dates
4. Updates the database with new predictions
5. Creates ensemble predictions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Advanced models
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

# Import database
from database import get_db

# Constants
DATA_PATH = Path(r"C:\Users\Emman\Documents\AI_dev\GDELT_ConflictPredictor\data\features_multiresolution")
MODELS_PATH = Path(r"C:\Users\Emman\Documents\AI_dev\GDELT_ConflictPredictor\prototype\backend\saved_models")
MODELS_PATH.mkdir(exist_ok=True)

print("="*80)
print("UPDATING PRODUCTION MODELS (XGBoost + GRU)")
print("="*80)

# ==================== DEFINE GRU MODEL ====================
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        out = self.bn(gru_out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out

# ==================== LOAD DATA ====================
print("\n[1/9] Loading data...")
# Use 2023-2025 data
df = pd.read_parquet(DATA_PATH / "country_day" / "country_day_features_2023_2025.parquet")
print(f"      Loaded {len(df):,} observations, {len(df.columns)} features")
print(f"      Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"      Countries: {df['Country'].nunique()}")

# ==================== PREPARE DATA ====================
print("\n[2/9] Preparing data...")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Country', 'Date'])

# Create target
conflict_col = 'IsHighConflict_sum'
df['NextDay_Conflict'] = df.groupby('Country')[conflict_col].shift(-1)
df['NextDay_HighConflict'] = (df['NextDay_Conflict'] >= df['NextDay_Conflict'].quantile(0.75)).astype(int)
df_ml = df.dropna(subset=['NextDay_Conflict', 'NextDay_HighConflict'])

# Select features
exclude_cols = ['Country', 'Date', 'NextDay_Conflict', 'NextDay_HighConflict', 'TopRegion']
feature_cols = [col for col in df_ml.columns if col not in exclude_cols
                and df_ml[col].dtype in ['int64', 'float64', 'int32', 'float32']]

X = df_ml[feature_cols].copy()
y = df_ml['NextDay_HighConflict'].copy()

# Handle missing values
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"      Features: {len(feature_cols)}, Samples: {X.shape[0]:,}")

# ==================== TRAIN/TEST SPLIT ====================
print("\n[3/9] Splitting data...")
latest_date = df_ml['Date'].max()
test_start_date = latest_date - timedelta(days=30)

train_mask = df_ml['Date'] < test_start_date
test_mask = df_ml['Date'] >= test_start_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)

print(f"      Train: {len(X_train):,}, Test: {len(X_test):,}")

# ==================== TRAIN XGBOOST ====================
print("\n[4/9] Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc'
)
xgb_model.fit(X_train, y_train)

y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)

print(f"      XGBoost - ROC-AUC: {roc_auc_score(y_test, y_prob_xgb)*100:.2f}%, F1: {f1_score(y_test, y_pred_xgb)*100:.2f}%")

# Save XGBoost
xgb_path = MODELS_PATH / "xgboost_model.pkl"
with open(xgb_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"      Saved to {xgb_path}")

# ==================== PREPARE SEQUENCES FOR GRU ====================
print("\n[5/9] Preparing sequences for GRU...")

def create_sequences(X, y, countries, seq_length=7):
    X_seq, y_seq = [], []
    for country in np.unique(countries):
        mask = countries == country
        X_country = X[mask]
        y_country = y[mask]
        for i in range(len(X_country) - seq_length):
            X_seq.append(X_country[i:i+seq_length])
            y_seq.append(y_country[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 7
countries_train = df_ml[train_mask]['Country'].values
countries_test = df_ml[test_mask]['Country'].values

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, countries_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, countries_test, seq_length)

print(f"      Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")

# ==================== TRAIN GRU ====================
print("\n[6/9] Training Bidirectional GRU...")
# Force CPU to avoid CUDA compatibility issues
device = torch.device('cpu')
print(f"      Using device: {device} (CPU mode for stability)")

input_size = X_train_seq.shape[2]
gru_model = GRUClassifier(input_size).to(device)

pos_weight = torch.tensor([scale_pos_weight]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

# Prepare data loaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
val_size = int(0.2 * len(train_dataset))
train_data, val_data = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_size, val_size])
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

# Training
print(f"      Training for up to 50 epochs...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(50):
    gru_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = gru_model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        clip_grad_norm_(gru_model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation
    gru_model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = gru_model(X_batch).squeeze()
            val_loss += criterion(outputs, y_batch).item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = gru_model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"      Epoch {epoch+1}/50 - Val Loss: {val_loss:.4f}")

    if patience_counter >= 10:
        print(f"      Early stopping at epoch {epoch+1}")
        break

# Load best model
gru_model.load_state_dict(best_model_state)

# Evaluate GRU
test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq))
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

gru_model.eval()
y_prob_gru = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = gru_model(X_batch).squeeze()
        probs = torch.sigmoid(outputs)
        y_prob_gru.extend(probs.cpu().numpy())

y_prob_gru = np.array(y_prob_gru)
y_pred_gru = (y_prob_gru >= 0.5).astype(int)

print(f"      GRU - ROC-AUC: {roc_auc_score(y_test_seq, y_prob_gru)*100:.2f}%, F1: {f1_score(y_test_seq, y_pred_gru)*100:.2f}%")

# Save GRU
gru_path = MODELS_PATH / "gru_model.pth"
torch.save({
    'model_state_dict': gru_model.state_dict(),
    'input_size': input_size,
    'seq_length': seq_length
}, gru_path)
print(f"      Saved to {gru_path}")

# Save scaler and feature names
scaler_path = MODELS_PATH / "scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

feature_names_path = MODELS_PATH / "feature_names.json"
with open(feature_names_path, 'w') as f:
    json.dump(feature_cols, f)

# ==================== GENERATE ALL PREDICTIONS ====================
print("\n[7/9] Generating predictions for all data...")

# XGBoost predictions for all data
all_probs_xgb = xgb_model.predict_proba(X_all_scaled)[:, 1]

# GRU predictions for all data with sequences
print(f"      Creating sequences for full dataset...")
countries_all = df_ml['Country'].values
X_all_seq, y_all_seq = create_sequences(X_all_scaled, y.values, countries_all, seq_length)
print(f"      Generated {len(X_all_seq):,} sequences")

all_dataset = TensorDataset(torch.FloatTensor(X_all_seq))
all_loader = DataLoader(all_dataset, batch_size=128, shuffle=False)

gru_model.eval()
all_probs_gru = []
with torch.no_grad():
    for (X_batch,) in all_loader:
        X_batch = X_batch.to(device)
        outputs = gru_model(X_batch).squeeze()
        probs = torch.sigmoid(outputs)
        all_probs_gru.extend(probs.cpu().numpy())

all_probs_gru = np.array(all_probs_gru)

# Align GRU predictions with XGBoost (GRU has fewer predictions due to sequence requirements)
# For simplicity, use XGBoost for all predictions, GRU for where available
print(f"      XGBoost predictions: {len(all_probs_xgb):,}")
print(f"      GRU predictions: {len(all_probs_gru):,}")

# Create ensemble: average of XGBoost and GRU where both are available
# For the first seq_length days per country, use only XGBoost
ensemble_probs = all_probs_xgb.copy()

# Map sequence predictions back to original indices
seq_idx = 0
for i, (country, date) in enumerate(zip(df_ml['Country'].values, df_ml['Date'].values)):
    # Check if we have a sequence prediction for this datapoint
    if seq_idx < len(all_probs_gru):
        # Use ensemble (average of both models)
        ensemble_probs[i] = (all_probs_xgb[i] + all_probs_gru[seq_idx]) / 2
        seq_idx += 1

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'country': df_ml['Country'].values,
    'date': df_ml['Date'].values,
    'prediction_date': df_ml['Date'].values + timedelta(days=1),
    'conflict_probability': ensemble_probs,
    'predicted_conflict': (ensemble_probs >= 0.5).astype(int),
    'risk_level': pd.cut(ensemble_probs,
                        bins=[0, 0.25, 0.5, 0.75, 1.0],
                        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']).astype(str),
    'model_name': 'Ensemble',
    'model_version': '1.0.0'
})

print(f"      Generated {len(predictions_df):,} ensemble predictions")

# ==================== UPDATE DATABASE ====================
print("\n[8/9] Updating database...")
db = get_db()

print("      Clearing old predictions...")
db.conn.execute("DELETE FROM predictions")
db.conn.commit()

print("      Inserting new predictions...")
insert_count = 0
for _, row in predictions_df.iterrows():
    try:
        db.conn.execute("""
            INSERT INTO predictions (
                country, date, prediction_date, conflict_probability,
                predicted_conflict, risk_level, model_name, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['country'],
            row['date'].strftime('%Y-%m-%d'),
            row['prediction_date'].strftime('%Y-%m-%d'),
            float(row['conflict_probability']),
            int(row['predicted_conflict']),
            row['risk_level'],
            row['model_name'],
            row['model_version']
        ))
        insert_count += 1
        if insert_count % 10000 == 0:
            print(f"        {insert_count:,} inserted...")
            db.conn.commit()
    except Exception as e:
        continue

db.conn.commit()
print(f"      Inserted {insert_count:,} predictions")

# ==================== UPDATE MODEL METADATA ====================
print("\n[9/9] Updating model metadata...")

# XGBoost metrics
xgb_metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred_xgb)),
    'precision': float(precision_score(y_test, y_pred_xgb)),
    'recall': float(recall_score(y_test, y_pred_xgb)),
    'f1_score': float(f1_score(y_test, y_pred_xgb)),
    'roc_auc': float(roc_auc_score(y_test, y_prob_xgb))
}

# GRU metrics
gru_metrics = {
    'accuracy': float(accuracy_score(y_test_seq, y_pred_gru)),
    'precision': float(precision_score(y_test_seq, y_pred_gru)),
    'recall': float(recall_score(y_test_seq, y_pred_gru)),
    'f1_score': float(f1_score(y_test_seq, y_pred_gru)),
    'roc_auc': float(roc_auc_score(y_test_seq, y_prob_gru))
}

# Update models table
db.conn.execute("DELETE FROM models WHERE model_name IN ('XGBoost', 'GRU', 'Ensemble')")

for model_name, metrics, model_path in [
    ('XGBoost', xgb_metrics, str(xgb_path)),
    ('GRU', gru_metrics, str(gru_path)),
    ('Ensemble', xgb_metrics, str(xgb_path))  # Use XGBoost metrics as proxy
]:
    db.conn.execute("""
        INSERT INTO models (
            model_name, model_version, model_path, trained_at,
            train_samples, test_samples, metrics, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name,
        '1.0.0',
        model_path,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        len(X_train),
        len(X_test),
        json.dumps(metrics),
        1
    ))

db.conn.commit()

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("UPDATE COMPLETE!")
print("="*80)

summary = db.get_data_summary()
print(f"\nDatabase Summary:")
print(f"  Predictions:       {summary.get('total_predictions', 0):,}")
print(f"  Countries:         {summary.get('total_countries', 0):,}")
print(f"  Latest prediction: {summary.get('latest_prediction', 'N/A')}")

print(f"\nModel Performance:")
print(f"  XGBoost - ROC-AUC: {xgb_metrics['roc_auc']*100:.2f}%, F1: {xgb_metrics['f1_score']*100:.2f}%")
print(f"  GRU     - ROC-AUC: {gru_metrics['roc_auc']*100:.2f}%, F1: {gru_metrics['f1_score']*100:.2f}%")

print(f"\nSaved Files:")
print(f"  XGBoost: {xgb_path}")
print(f"  GRU:     {gru_path}")
print(f"  Scaler:  {scaler_path}")

print("\n" + "="*80)
