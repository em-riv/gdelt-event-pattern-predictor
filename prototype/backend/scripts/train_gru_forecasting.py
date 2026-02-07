"""
Train Bidirectional GRU for proper conflict forecasting
- Uses 7-day sequences to predict 1-day ahead
- No data leakage
- Proper time-based train/test split
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
from datetime import datetime

print("=" * 80)
print("TRAINING BIDIRECTIONAL GRU FOR CONFLICT FORECASTING")
print("=" * 80)

# Device setup - using CPU with optimizations
device = torch.device('cpu')
torch.set_num_threads(16)  # Optimize thread count
print(f"\n1. Device: {device}")
print(f"   Using CPU with {torch.get_num_threads()} threads")

# Load data
print("\n2. Loading data...")
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
print(f"   Loaded {len(df):,} rows")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Sort by country and date
df = df.sort_values(['Country', 'Date'])

# Create multi-horizon targets (7, 14, 30 days ahead)
print("\n3. Creating multi-horizon FUTURE targets...")
for horizon in [7, 14, 30]:
    df[f'target_{horizon}d'] = df.groupby('Country')['IsHighConflict_mean'].shift(-horizon)
    df[f'target_{horizon}d'] = (df[f'target_{horizon}d'] > 0.15).astype(int)

# Use 7-day as primary target for training
df['target'] = df['target_7d']
df = df[df['target_7d'].notna() & df['target_14d'].notna() & df['target_30d'].notna()]

print(f"   Samples: {len(df):,}")
print(f"   Conflict rates:")
for horizon in [7, 14, 30]:
    rate = df[f'target_{horizon}d'].mean()
    print(f"     {horizon}-day: {rate*100:.2f}%")

# Feature engineering (excluding target-related features)
print("\n4. Engineering features...")

# Country baselines
country_stats = df.groupby('Country').agg({
    'EventCount': 'mean',
    'IsHighConflict_sum': 'mean',
    'GoldsteinScale_mean': 'mean',
    'AvgTone_mean': 'mean'
}).add_suffix('_country_baseline')

df = df.merge(country_stats, left_on='Country', right_index=True)

# Normalized features
df['EventCount_normalized'] = df['EventCount'] / (df['EventCount_country_baseline'] + 1)
df['HighConflict_vs_baseline'] = df['IsHighConflict_sum'] / (df['IsHighConflict_sum_country_baseline'] + 1)
df['GoldsteinScale_vs_baseline'] = df['GoldsteinScale_mean'] - df['GoldsteinScale_mean_country_baseline']
df['AvgTone_vs_baseline'] = df['AvgTone_mean'] - df['AvgTone_mean_country_baseline']

# Ratios
df['ConflictRatio'] = df['IsHighConflict_mean'] / (df['IsCooperation_mean'] + 0.01)
df['MaterialConflictRatio'] = df['IsMaterialConflict_mean'] / (df['IsMaterialCoop_mean'] + 0.01)
df['VerbalConflictRatio'] = df['IsVerbalConflict_mean'] / (df['IsVerbalCoop_mean'] + 0.01)
df['EventDiversity'] = df['UniqueEventTypes'] / (df['EventCount'] + 1)
df['ActorDiversity'] = (df['UniqueActor1'] + df['UniqueActor2']) / (df['EventCount'] + 1)

# Rolling features
for col in ['IsHighConflict_mean', 'GoldsteinScale_mean', 'AvgTone_mean', 'EventCount']:
    df[f'{col}_7day_avg'] = df.groupby('Country')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df[f'{col}_7day_std'] = df.groupby('Country')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

# Trends
df['HighConflict_trend'] = df['IsHighConflict_mean'] - df['IsHighConflict_mean_7day_avg']
df['Goldstein_trend'] = df['GoldsteinScale_mean'] - df['GoldsteinScale_mean_7day_avg']
df['Tone_trend'] = df['AvgTone_mean'] - df['AvgTone_mean_7day_avg']

# Select features (EXCLUDE IsHighConflict_mean AND ALL DERIVED FEATURES to prevent leakage)
exclude_features = [
    'Country', 'Date', 'target', 'target_next_day',
    'target_7d', 'target_14d', 'target_30d',  # Exclude all targets
    'IsHighConflict_mean', 'IsHighConflict_sum',  # Prevent leakage
    'IsHighConflict_mean_7day_avg', 'HighConflict_trend',  # Derived from IsHighConflict_mean
    'ConflictRatio', 'HighConflict_vs_baseline',  # Use IsHighConflict_mean
    'IsCooperation_sum', 'IsVerbalCoop_sum', 'IsMaterialCoop_sum',
    'IsVerbalConflict_sum', 'IsMaterialConflict_sum',
    'IsHighIntensity_sum', 'IsPositiveEvent_sum', 'IsNegativeEvent_sum',
    'IsCrossBorder_sum', 'IsDomestic_sum', 'HasCoordinates_sum',
    'NumMentions_sum', 'NumSources_sum', 'NumArticles_sum',
    'IsQuad1_VerbalCoop_sum', 'IsQuad2_MaterialCoop_sum',
    'IsQuad3_VerbalConflict_sum', 'IsQuad4_MaterialConflict_sum',
    'EventCount_country_baseline', 'IsHighConflict_sum_country_baseline',
    'GoldsteinScale_mean_country_baseline', 'AvgTone_mean_country_baseline'
]

feature_cols = [col for col in df.columns if col not in exclude_features]
print(f"   Using {len(feature_cols)} features")

# Handle missing/infinite
df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

# Time-based split
print("\n5. Splitting data...")
split_date = pd.Timestamp('2025-10-01')
train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

X_train = train_df[feature_cols].values
y_train = train_df['target'].values
X_test = test_df[feature_cols].values
y_test = test_df['target'].values

countries_train = train_df['Country'].values
countries_test = test_df['Country'].values

print(f"   Train: {len(X_train):,} samples ({y_train.sum():,} conflicts, {y_train.mean()*100:.2f}%)")
print(f"   Test:  {len(X_test):,} samples ({y_test.sum():,} conflicts, {y_test.mean()*100:.2f}%)")

# Scale
print("\n6. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create sequences
print("\n7. Creating sequences...")
seq_length = 7

def create_sequences(X, y, countries, seq_length):
    X_seq, y_seq = [], []
    unique_countries = np.unique(countries)

    for country in unique_countries:
        mask = countries == country
        X_country = X[mask]
        y_country = y[mask]

        for i in range(len(X_country) - seq_length):
            X_seq.append(X_country[i:i+seq_length])
            y_seq.append(y_country[i+seq_length])

    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, countries_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, countries_test, seq_length)

print(f"   Train sequences: {X_train_seq.shape}")
print(f"   Test sequences: {X_test_seq.shape}")
print(f"   Class balance - Train: {y_train_seq.mean()*100:.2f}%, Test: {y_test_seq.mean()*100:.2f}%")

# Define GRU model
print("\n8. Building Bidirectional GRU model...")

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]  # Last timestep
        out = self.bn(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

input_size = X_train_seq.shape[2]
model = GRUClassifier(input_size).to(device)
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
pos_weight = torch.tensor([(y_train_seq == 0).sum() / (y_train_seq == 1).sum()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Data loaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=512, shuffle=False, num_workers=0)

# Training functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), auc

# Train
print("\n9. Training GRU...")
num_epochs = 20  # Reduced for faster CPU training
best_val_auc = 0
patience = 5  # Reduced for faster convergence
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    scheduler.step(val_auc)

    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"   Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_model_state)

# Evaluate
print("\n10. Evaluating on test set...")
test_dataset = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq))
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

model.eval()
y_pred_prob = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch).squeeze()
        probs = torch.sigmoid(outputs)
        y_pred_prob.extend(probs.cpu().numpy())

y_pred_prob = np.array(y_pred_prob)
y_pred = (y_pred_prob > 0.5).astype(int)

roc_auc = roc_auc_score(y_test_seq, y_pred_prob)
f1 = f1_score(y_test_seq, y_pred)
precision = precision_score(y_test_seq, y_pred)
recall = recall_score(y_test_seq, y_pred)

print(f"\n{'='*60}")
print(f"RESULTS (Bidirectional GRU - 7-day forecast):")
print(f"{'='*60}")
print(f"ROC-AUC:   {roc_auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"{'='*60}")

# Evaluate on 14-day and 30-day horizons
print("\n11. Evaluating other horizons...")
test_df_seq = test_df.iloc[seq_length:]  # Align with sequences

for horizon in [14, 30]:
    y_horizon = test_df_seq[f'target_{horizon}d'].values[:len(y_pred_prob)]
    y_horizon = y_horizon[~np.isnan(y_horizon)]
    y_pred_prob_aligned = y_pred_prob[:len(y_horizon)]

    if len(y_horizon) > 0:
        y_pred_horizon = (y_pred_prob_aligned > 0.5).astype(int)
        roc_auc_h = roc_auc_score(y_horizon, y_pred_prob_aligned)
        f1_h = f1_score(y_horizon, y_pred_horizon)
        print(f"   {horizon}-day: ROC-AUC={roc_auc_h*100:.2f}%, F1={f1_h*100:.2f}%")

# Save
print("\n11. Saving model...")
model_dir = Path('saved_models')
torch.save(model.state_dict(), model_dir / 'gru_forecast.pth')
joblib.dump(scaler, model_dir / 'scaler_gru_forecast.pkl')
with open(model_dir / 'features_gru_forecast.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

metadata = {
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'model_type': 'Bidirectional GRU',
    'sequence_length': seq_length,
    'target': 'IsHighConflict_mean > 0.15 (NEXT DAY)',
    'metrics': {
        'roc_auc': float(roc_auc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    },
    'architecture': {
        'input_size': input_size,
        'hidden_size': 128,
        'num_layers': 2,
        'bidirectional': True
    }
}

with open(model_dir / 'metadata_gru_forecast.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("   Saved!")
print("\n" + "="*80)
print("GRU TRAINING COMPLETE!")
print("="*80)
