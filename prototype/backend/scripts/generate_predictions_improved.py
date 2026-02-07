"""
Generate multi-horizon predictions using the improved model
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
from database import Database

print("=" * 80)
print("GENERATING MULTI-HORIZON PREDICTIONS (Improved Model)")
print("=" * 80)

# Load model
print("\n1. Loading improved model...")
model = joblib.load('saved_models/xgboost_model_improved.pkl')
scaler = joblib.load('saved_models/scaler_improved.pkl')
with open('saved_models/feature_names_improved.json', 'r') as f:
    feature_cols = json.load(f)

print(f"   Model loaded: {len(feature_cols)} features")

# Load data
print("\n2. Loading latest data...")
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
print(f"   Loaded {len(df):,} rows")

# Apply same feature engineering as training
print("\n3. Engineering features...")

# Group by country to get baselines
country_stats = df.groupby('Country').agg({
    'EventCount': 'mean',
    'IsHighConflict_sum': 'mean',
    'GoldsteinScale_mean': 'mean',
    'AvgTone_mean': 'mean'
}).add_suffix('_country_baseline')

df = df.merge(country_stats, left_on='Country', right_index=True)

# Create normalized features
df['EventCount_normalized'] = df['EventCount'] / (df['EventCount_country_baseline'] + 1)
df['HighConflict_vs_baseline'] = df['IsHighConflict_sum'] / (df['IsHighConflict_sum_country_baseline'] + 1)
df['GoldsteinScale_vs_baseline'] = df['GoldsteinScale_mean'] - df['GoldsteinScale_mean_country_baseline']
df['AvgTone_vs_baseline'] = df['AvgTone_mean'] - df['AvgTone_mean_country_baseline']

# Conflict intensity ratios
df['ConflictRatio'] = df['IsHighConflict_mean'] / (df['IsCooperation_mean'] + 0.01)
df['MaterialConflictRatio'] = df['IsMaterialConflict_mean'] / (df['IsMaterialCoop_mean'] + 0.01)
df['VerbalConflictRatio'] = df['IsVerbalConflict_mean'] / (df['IsVerbalCoop_mean'] + 0.01)

# Event diversity
df['EventDiversity'] = df['UniqueEventTypes'] / (df['EventCount'] + 1)
df['ActorDiversity'] = (df['UniqueActor1'] + df['UniqueActor2']) / (df['EventCount'] + 1)

# Rolling averages
df = df.sort_values(['Country', 'Date'])
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

# Fill missing values
df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

# Get latest date and data for each country
print("\n4. Preparing prediction data...")
latest_date = df['Date'].max()
print(f"   Latest date in data: {latest_date}")

latest_data = df[df['Date'] == latest_date].copy()
print(f"   Countries: {len(latest_data)}")

# Generate predictions for multiple horizons
print("\n5. Generating multi-horizon predictions...")
horizons = [7, 14, 30]
all_predictions = []

for horizon in horizons:
    print(f"\n   Horizon: {horizon} days")

    # Extract features
    X = latest_data[feature_cols]
    X_scaled = scaler.transform(X)

    # Predict
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Apply slight decay for longer horizons
    decay_factor = 1.0 - (horizon - 7) * 0.005
    adjusted_probs = probabilities * decay_factor

    # Calculate target date
    target_date = latest_date + timedelta(days=horizon)

    # Create predictions
    for idx, row in latest_data.iterrows():
        prob = adjusted_probs[latest_data.index.get_loc(idx)]

        # Determine risk level
        if prob >= 0.70:
            risk_level = 'CRITICAL'
        elif prob >= 0.50:
            risk_level = 'HIGH'
        elif prob >= 0.30:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        all_predictions.append({
            'country': row['Country'],
            'prediction_date': datetime.now().date(),
            'target_date': target_date.date(),
            'horizon_days': horizon,
            'conflict_probability': float(prob),
            'risk_level': risk_level,
            'confidence': float(probabilities[latest_data.index.get_loc(idx)]),
            'model_version': 'improved_1.0.0'
        })

    print(f"      Generated {len(latest_data)} predictions")

print(f"\n   Total predictions: {len(all_predictions)}")

# Show statistics
print("\n6. Prediction statistics:")
pred_df = pd.DataFrame(all_predictions)
for horizon in horizons:
    h_preds = pred_df[pred_df['horizon_days'] == horizon]
    print(f"\n   {horizon}-day horizon:")
    print(f"      Average probability: {h_preds['conflict_probability'].mean()*100:.2f}%")
    print(f"      Risk levels: {h_preds['risk_level'].value_counts().to_dict()}")

# Save to database
print("\n7. Saving predictions to database...")
db = Database()

# Add horizon_days column if it doesn't exist
try:
    db.conn.execute("ALTER TABLE predictions ADD COLUMN horizon_days INTEGER")
    db.conn.commit()
    print("   Added horizon_days column")
except:
    pass  # Column already exists

# Clear old predictions
db.conn.execute("DELETE FROM predictions")
db.conn.commit()

# Insert new predictions
for pred in all_predictions:
    db.conn.execute("""
        INSERT OR REPLACE INTO predictions
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

db.conn.commit()
print(f"   Saved {len(all_predictions)} predictions to database")

# Show top predictions by horizon
print("\n8. Top 10 predictions by horizon:")
for horizon in horizons:
    h_preds = pred_df[pred_df['horizon_days'] == horizon].sort_values('conflict_probability', ascending=False).head(10)
    print(f"\n   {horizon}-day horizon:")
    for idx, row in h_preds.iterrows():
        print(f"      {row['country']}: {row['conflict_probability']*100:.1f}% ({row['risk_level']})")

print("\n" + "="*80)
print("PREDICTION GENERATION COMPLETE!")
print("="*80)
