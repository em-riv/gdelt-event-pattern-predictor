"""
Generate multi-horizon conflict predictions using trained GRU model
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime, timedelta
import sqlite3

# Device setup
device = torch.device('cpu')
print("=" * 80)
print("GENERATING GRU PREDICTIONS")
print("=" * 80)
print(f"\n1. Device: {device}")

# Load model architecture
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

# Load model
print("\n2. Loading GRU model...")
model_dir = Path('saved_models')
with open(model_dir / 'features_gru_forecast.json', 'r') as f:
    feature_cols = json.load(f)

scaler = joblib.load(model_dir / 'scaler_gru_forecast.pkl')
input_size = len(feature_cols)
model = GRUClassifier(input_size).to(device)
model.load_state_dict(torch.load(model_dir / 'gru_forecast.pth', map_location=device))
model.eval()
print(f"   Model loaded: {input_size} features")
print(f"   Features: {len(feature_cols)}")

# Load data
print("\n3. Loading latest data...")
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025_fixed.parquet')
print(f"   Loaded {len(df):,} rows")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Sort by country and date
df = df.sort_values(['Country', 'Date'])

# Feature engineering (same as training)
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

# Get latest data for each country
print("\n5. Preparing prediction data...")
latest_dates = df.groupby('Country')['Date'].max().reset_index()
latest_dates.columns = ['Country', 'LatestDate']
df = df.merge(latest_dates, on='Country')

# Get last 7 days for each country for sequences
seq_length = 7
countries = df['Country'].unique()
print(f"   Countries: {len(countries)}")

# Create sequences for each country
def create_prediction_sequences(df, country, feature_cols, seq_length):
    country_df = df[df['Country'] == country].sort_values('Date').tail(seq_length)
    if len(country_df) < seq_length:
        return None
    X = country_df[feature_cols].values
    return X

# Scale and predict
print("\n6. Generating predictions...")
all_predictions = []

for country in countries:
    seq = create_prediction_sequences(df, country, feature_cols, seq_length)

    if seq is None:
        continue

    # Handle NaN values before scaling
    seq_filled = pd.DataFrame(seq, columns=feature_cols)
    
    # Fill NaN values with appropriate defaults
    for col in seq_filled.columns:
        if seq_filled[col].isna().any():
            if 'mean' in col.lower() or 'avg' in col.lower():
                seq_filled[col] = seq_filled[col].fillna(0)
            elif 'std' in col.lower():
                seq_filled[col] = seq_filled[col].fillna(0) 
            elif 'ratio' in col.lower():
                seq_filled[col] = seq_filled[col].fillna(1)
            elif 'trend' in col.lower():
                seq_filled[col] = seq_filled[col].fillna(0)
            elif 'diversity' in col.lower():
                seq_filled[col] = seq_filled[col].fillna(0)
            elif 'normalized' in col.lower():
                seq_filled[col] = seq_filled[col].fillna(1)
            elif 'baseline' in col.lower():
                seq_filled[col] = seq_filled[col].fillna(0)
            else:
                seq_filled[col] = seq_filled[col].fillna(seq_filled[col].median())
    
    seq = seq_filled.values
    
    # Scale
    seq_scaled = scaler.transform(seq)

    # Create tensor
    X_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)  # (1, seq_length, features)

    # Predict
    with torch.no_grad():
        logits = model(X_tensor)
        prob = torch.sigmoid(logits).cpu().numpy()[0][0]
        
        # Debug: Check for invalid probability  
        if np.isnan(prob) or np.isinf(prob) or prob is None:
            print(f"   [WARNING] Invalid probability for {country}: {prob}, using default 0.5")
            prob = 0.5
        else:
            # Show valid predictions for debugging
            if country in ['USA', 'CHN', 'RUS', 'UKR', 'ISR']:
                print(f"   [INFO] {country}: {prob:.4f}")

    # Get latest date for this country
    latest_date = df[df['Country'] == country]['Date'].max()
    prediction_date = datetime.now().strftime('%Y-%m-%d')

    # Generate predictions for 7, 14, 30 day horizons
    for horizon_days in [7, 14, 30]:
        target_date = (pd.to_datetime(latest_date) + timedelta(days=horizon_days)).strftime('%Y-%m-%d')

        # Use same probability for all horizons (model was trained on 7-day)
        # Note: This is a simplification - ideally we'd have separate models or multi-output
        conflict_probability = float(prob)

        # Handle NaN
        if np.isnan(conflict_probability):
            conflict_probability = 0.5  # Default to 50% if NaN

        # Classify risk level
        if conflict_probability >= 0.75:
            risk_level = 'CRITICAL'
        elif conflict_probability >= 0.50:
            risk_level = 'HIGH'
        elif conflict_probability >= 0.30:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        # Confidence based on model performance (74% ROC-AUC)
        confidence = 0.74

        all_predictions.append({
            'country': country,
            'prediction_date': prediction_date,
            'target_date': target_date,
            'horizon_days': horizon_days,
            'conflict_probability': conflict_probability,
            'risk_level': risk_level,
            'confidence': confidence,
            'model_version': 'GRU_v1.0'
        })

print(f"   Generated {len(all_predictions)} predictions ({len(countries)} countries  3 horizons)")

# Save to database
print("\n7. Saving predictions to database...")
db_path = Path('conflict_predictor.db')

# Create connection
conn = sqlite3.connect(db_path)

# Create table if not exists
conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        country TEXT NOT NULL,
        prediction_date TEXT NOT NULL,
        target_date TEXT NOT NULL,
        horizon_days INTEGER NOT NULL,
        conflict_probability REAL NOT NULL,
        risk_level TEXT NOT NULL,
        confidence REAL NOT NULL,
        model_version TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(country, target_date, horizon_days)
    )
""")
conn.commit()
print("   Ensured table exists")

# Clear old predictions
conn.execute("DELETE FROM predictions")
conn.commit()
print("   Cleared old predictions")

# Insert new predictions
skipped_count = 0
for pred in all_predictions:
    # Final validation before insertion
    if (pred['conflict_probability'] is None or 
        not isinstance(pred['conflict_probability'], (int, float)) or 
        np.isnan(pred['conflict_probability']) or 
        np.isinf(pred['conflict_probability'])):
        print(f"   [SKIP] Invalid prediction for {pred['country']} horizon {pred['horizon_days']}: {pred['conflict_probability']}")
        skipped_count += 1
        continue
        
    try:
        conn.execute("""
            INSERT INTO predictions
            (country, prediction_date, target_date, horizon_days, conflict_probability, risk_level, confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred['country'],
            pred['prediction_date'],
            pred['target_date'],
            pred['horizon_days'],
            pred['conflict_probability'],
            pred['risk_level'],
            pred['confidence'],
            pred['model_version']
        ))
    except Exception as e:
        print(f"   [ERROR] Failed to insert {pred['country']} horizon {pred['horizon_days']}: {e}")
        skipped_count += 1
        continue

conn.commit()
conn.close()
successful_count = len(all_predictions) - skipped_count
print(f"   Saved {successful_count}/{len(all_predictions)} predictions to database")
if skipped_count > 0:
    print(f"   Skipped {skipped_count} invalid predictions")

# Print sample predictions
print("\n8. Sample predictions:")
sample_countries = ['Israel', 'Palestine', 'Ukraine', 'Afghanistan', 'United States']
for country in sample_countries:
    country_preds = [p for p in all_predictions if p['country'] == country]
    if country_preds:
        print(f"\n   {country}:")
        for p in country_preds:
            print(f"      {p['horizon_days']:2d}-day: {p['conflict_probability']*100:5.1f}% {p['risk_level']:8s}")

print("\n" + "=" * 80)
print("PREDICTIONS GENERATED SUCCESSFULLY!")
print("=" * 80)
