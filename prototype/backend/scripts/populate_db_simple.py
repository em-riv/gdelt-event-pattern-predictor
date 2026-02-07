"""
Simple script to populate database with XGBoost predictions
Uses existing trained model
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
print("POPULATING DATABASE WITH PREDICTIONS")
print("="*80)

# Load model
print("\n[1/4] Loading trained model...")
with open(MODELS_PATH / "xgboost_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open(MODELS_PATH / "scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

print("      Model and scaler loaded")

# Load data
print("\n[2/4] Loading data...")
df = pd.read_parquet(DATA_PATH / "country_day" / "country_day_features_2023_2025.parquet")
print(f"      Loaded {len(df):,} observations")

# Prepare data
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
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Scale and predict
X_scaled = scaler.transform(X)

print("\n[3/4] Generating predictions...")
probs = model.predict_proba(X_scaled)[:, 1]
print(f"      Generated {len(probs):,} predictions")

# Calculate risk levels
def get_risk_level(prob):
    if prob < 0.25:
        return 'LOW'
    elif prob < 0.5:
        return 'MEDIUM'
    elif prob < 0.75:
        return 'HIGH'
    else:
        return 'CRITICAL'

# Populate database
print("\n[4/4] Updating database...")
db = get_db()

print("      Clearing old predictions...")
db.conn.execute("DELETE FROM predictions")
db.conn.commit()

print("      Inserting new predictions...")
insert_count = 0
today = datetime.now().date()

for i, (idx, row) in enumerate(df_ml.iterrows()):
    country = row['Country']
    target_date = row['Date'].date() + timedelta(days=1)  # Predicting next day
    prob = float(probs[i])
    risk_level = get_risk_level(prob)

    try:
        db.conn.execute("""
            INSERT INTO predictions (
                country, prediction_date, target_date,
                conflict_probability, risk_level, confidence, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            country,
            today.strftime('%Y-%m-%d'),
            target_date.strftime('%Y-%m-%d'),
            prob,
            risk_level,
            prob,  # Use probability as confidence
            '1.0.0'
        ))
        insert_count += 1

        if insert_count % 10000 == 0:
            print(f"        {insert_count:,} inserted...")
            db.conn.commit()

    except Exception as e:
        if insert_count == 0:  # Print first error for debugging
            print(f"        Error: {e}")
        continue

db.conn.commit()

print(f"      Inserted {insert_count:,} predictions")

# Summary
print("\n" + "="*80)
print("DATABASE UPDATE COMPLETE!")
print("="*80)

summary = db.get_data_summary()
print(f"\nDatabase Summary:")
print(f"  Total predictions: {summary.get('total_predictions', 0):,}")
print(f"  Total countries:   {summary.get('total_countries', 0)}")
print(f"  Latest prediction: {summary.get('latest_prediction')}")

print("\n" + "="*80)
print(" Backend at http://localhost:8000")
print(" Frontend at http://localhost:4200")
print("="*80)
