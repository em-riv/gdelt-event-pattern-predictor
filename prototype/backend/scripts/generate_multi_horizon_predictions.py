"""
Generate Multi-Horizon Predictions (7, 14, 30 days)
Uses rolling windows and trained XGBoost model
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from database import get_db

# Load saved model and scaler
MODELS_PATH = Path("saved_models")
DATA_PATH = Path(r"C:\Users\Emman\Documents\AI_dev\GDELT_ConflictPredictor\data\features_multiresolution")

print("="*80)
print("GENERATING MULTI-HORIZON PREDICTIONS (7, 14, 30 DAYS)")
print("="*80)

# Load model
print("\n[1/5] Loading trained model...")
with open(MODELS_PATH / "xgboost_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open(MODELS_PATH / "scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open(MODELS_PATH / "feature_names.json", 'r') as f:
    import json
    feature_cols = json.load(f)

print(f"      Model loaded with {len(feature_cols)} features")

# Load data
print("\n[2/5] Loading and preparing data...")
df = pd.read_parquet(DATA_PATH / "country_day" / "country_day_features_2023_2025.parquet")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Country', 'Date'])

print(f"      Loaded {len(df):,} observations")
print(f"      Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"      Countries: {df['Country'].nunique()}")

# Create target variable
conflict_col = 'IsHighConflict_sum'
df['NextDay_Conflict'] = df.groupby('Country')[conflict_col].shift(-1)
df['NextDay_HighConflict'] = (df['NextDay_Conflict'] >= df['NextDay_Conflict'].quantile(0.75)).astype(int)

# Get latest date per country for current predictions
latest_dates = df.groupby('Country')['Date'].max().reset_index()
latest_dates.columns = ['Country', 'LatestDate']

print(f"\n[3/5] Creating rolling window features for multiple horizons...")

# Function to create rolling window features
def create_rolling_features(country_df, window_size=7):
    """Create rolling window aggregations"""
    features = {}

    # Rolling mean
    for col in ['GoldsteinScale_mean', 'NumArticles_sum', 'AvgTone_mean']:
        if col in country_df.columns:
            features[f'{col}_roll{window_size}'] = country_df[col].rolling(window=window_size, min_periods=1).mean().iloc[-1]

    # Rolling std
    for col in ['GoldsteinScale_mean', 'AvgTone_mean']:
        if col in country_df.columns:
            features[f'{col}_roll{window_size}_std'] = country_df[col].rolling(window=window_size, min_periods=1).std().iloc[-1]

    # Trend (last value - first value in window)
    for col in ['GoldsteinScale_mean', 'AvgTone_mean']:
        if col in country_df.columns:
            values = country_df[col].tail(window_size)
            if len(values) > 1:
                features[f'{col}_trend{window_size}'] = values.iloc[-1] - values.iloc[0]
            else:
                features[f'{col}_trend{window_size}'] = 0

    return features

# Prepare predictions for each horizon
horizons = [7, 14, 30]
all_predictions = []

print("\n[4/5] Generating predictions for each country and horizon...")

for country in df['Country'].unique():
    country_df = df[df['Country'] == country].sort_values('Date')

    # Get latest complete data point
    latest_data = country_df[country_df['Date'] == country_df['Date'].max()].iloc[0]
    latest_date = latest_data['Date']

    # Extract base features
    base_features = {}
    for col in feature_cols:
        if col in latest_data.index:
            base_features[col] = latest_data[col]
        else:
            base_features[col] = 0  # Default for missing features

    # Add rolling window features
    for window in [7, 14, 30]:
        rolling_feats = create_rolling_features(country_df, window_size=window)
        for key, val in rolling_feats.items():
            if key in feature_cols:
                base_features[key] = val

    # Create feature vector
    X = np.array([base_features[col] if col in base_features else 0 for col in feature_cols]).reshape(1, -1)

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    X_scaled = scaler.transform(X)

    # Get base probability
    base_prob = model.predict_proba(X_scaled)[0, 1]

    # Generate predictions for each horizon
    for horizon in horizons:
        # Adjust probability based on horizon (longer horizons = more uncertainty)
        # Add some decay factor for longer horizons
        decay_factor = 1.0 - (horizon - 7) * 0.005  # Small decay for longer horizons
        adjusted_prob = base_prob * decay_factor

        # Calculate target date (from latest data point + 1 day, then add horizon)
        # This ensures we're predicting horizon days FROM tomorrow
        target_date = latest_date + timedelta(days=1 + horizon)

        # Determine risk level
        if adjusted_prob < 0.25:
            risk_level = 'LOW'
        elif adjusted_prob < 0.5:
            risk_level = 'MEDIUM'
        elif adjusted_prob < 0.75:
            risk_level = 'HIGH'
        else:
            risk_level = 'CRITICAL'

        all_predictions.append({
            'country': country,
            'prediction_date': datetime.now().date(),
            'target_date': target_date.date(),
            'horizon_days': horizon,
            'conflict_probability': float(adjusted_prob),
            'risk_level': risk_level,
            'confidence': float(base_prob),  # Use base prob as confidence
            'model_version': '1.0.0'
        })

print(f"      Generated {len(all_predictions):,} predictions")
print(f"      Horizons: {horizons} days")
print(f"      Countries: {len(df['Country'].unique())}")

# Print distribution
print("\n      Risk level distribution:")
pred_df = pd.DataFrame(all_predictions)
for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
    count = (pred_df['risk_level'] == level).sum()
    pct = count / len(pred_df) * 100
    print(f"        {level:8s}: {count:4,} ({pct:5.2f}%)")

print("\n      Distribution by horizon:")
for horizon in horizons:
    count = (pred_df['horizon_days'] == horizon).sum()
    print(f"        {horizon:2d} days: {count:3,} predictions")

# Populate database
print("\n[5/5] Updating database...")
db = get_db()

print("      Clearing old predictions...")
db.conn.execute("DELETE FROM predictions")
db.conn.commit()

print("      Inserting new predictions...")
insert_count = 0

for pred in all_predictions:
    try:
        db.conn.execute("""
            INSERT INTO predictions (
                country, prediction_date, target_date,
                conflict_probability, risk_level, confidence, model_version, horizon_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred['country'],
            pred['prediction_date'].strftime('%Y-%m-%d'),
            pred['target_date'].strftime('%Y-%m-%d'),
            pred['conflict_probability'],
            pred['risk_level'],
            pred['confidence'],
            pred['model_version'],
            pred['horizon_days']
        ))
        insert_count += 1

        if insert_count % 100 == 0:
            db.conn.commit()

    except Exception as e:
        if insert_count == 0:
            print(f"        Error: {e}")
        continue

db.conn.commit()

print(f"      Inserted {insert_count:,} predictions")

# Summary
print("\n" + "="*80)
print("MULTI-HORIZON PREDICTIONS COMPLETE!")
print("="*80)

summary = db.get_data_summary()
print(f"\nDatabase Summary:")
print(f"  Total predictions: {summary.get('total_predictions', 0):,}")
print(f"  Total countries:   {summary.get('total_countries', 0)}")
print(f"  Latest prediction: {summary.get('latest_prediction')}")

print(f"\nPrediction Horizons:")
print(f"  7-day forecast:  {len([p for p in all_predictions if p['horizon_days'] == 7])} countries")
print(f"  14-day forecast: {len([p for p in all_predictions if p['horizon_days'] == 14])} countries")
print(f"  30-day forecast: {len([p for p in all_predictions if p['horizon_days'] == 30])} countries")

print("\n" + "="*80)
print(" Backend at http://localhost:8000")
print(" Frontend at http://localhost:4200")
print("="*80)
