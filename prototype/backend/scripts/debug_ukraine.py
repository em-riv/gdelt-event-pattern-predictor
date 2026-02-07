import pandas as pd
import numpy as np
import joblib
import json

# Load model
model = joblib.load('saved_models/xgboost_model_improved.pkl')
scaler = joblib.load('saved_models/scaler_improved.pkl')
with open('saved_models/feature_names_improved.json', 'r') as f:
    feature_cols = json.load(f)

# Load data
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')

# Apply feature engineering
country_stats = df.groupby('Country').agg({
    'EventCount': 'mean',
    'IsHighConflict_sum': 'mean',
    'GoldsteinScale_mean': 'mean',
    'AvgTone_mean': 'mean'
}).add_suffix('_country_baseline')

df = df.merge(country_stats, left_on='Country', right_index=True)

df['EventCount_normalized'] = df['EventCount'] / (df['EventCount_country_baseline'] + 1)
df['HighConflict_vs_baseline'] = df['IsHighConflict_sum'] / (df['IsHighConflict_sum_country_baseline'] + 1)
df['GoldsteinScale_vs_baseline'] = df['GoldsteinScale_mean'] - df['GoldsteinScale_mean_country_baseline']
df['AvgTone_vs_baseline'] = df['AvgTone_mean'] - df['AvgTone_mean_country_baseline']

df['ConflictRatio'] = df['IsHighConflict_mean'] / (df['IsCooperation_mean'] + 0.01)
df['MaterialConflictRatio'] = df['IsMaterialConflict_mean'] / (df['IsMaterialCoop_mean'] + 0.01)
df['VerbalConflictRatio'] = df['IsVerbalConflict_mean'] / (df['IsVerbalCoop_mean'] + 0.01)

df['EventDiversity'] = df['UniqueEventTypes'] / (df['EventCount'] + 1)
df['ActorDiversity'] = (df['UniqueActor1'] + df['UniqueActor2']) / (df['EventCount'] + 1)

df = df.sort_values(['Country', 'Date'])
for col in ['IsHighConflict_mean', 'GoldsteinScale_mean', 'AvgTone_mean', 'EventCount']:
    df[f'{col}_7day_avg'] = df.groupby('Country')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df[f'{col}_7day_std'] = df.groupby('Country')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

df['HighConflict_trend'] = df['IsHighConflict_mean'] - df['IsHighConflict_mean_7day_avg']
df['Goldstein_trend'] = df['GoldsteinScale_mean'] - df['GoldsteinScale_mean_7day_avg']
df['Tone_trend'] = df['AvgTone_mean'] - df['AvgTone_mean_7day_avg']

df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

# Get Ukraine latest
latest_date = df['Date'].max()
ukr_latest = df[(df['Country'] == 'UKR') & (df['Date'] == latest_date)]

if len(ukr_latest) > 0:
    print(f"Ukraine on {latest_date}:")
    print(f"  IsHighConflict_mean: {ukr_latest['IsHighConflict_mean'].values[0]:.4f}")
    print(f"  ConflictRatio: {ukr_latest['ConflictRatio'].values[0]:.4f}")
    print(f"  GoldsteinScale_mean: {ukr_latest['GoldsteinScale_mean'].values[0]:.4f}")

    # Make prediction
    X = ukr_latest[feature_cols]
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[:, 1][0]

    print(f"\n  Predicted probability: {prob*100:.2f}%")
    print(f"  Class prediction: {model.predict(X_scaled)[0]}")

    # Check feature values
    print(f"\n  Top features:")
    for feat in ['IsHighConflict_mean', 'ConflictRatio', 'GoldsteinScale_mean', 'IntensityScore_mean']:
        if feat in feature_cols:
            print(f"    {feat}: {ukr_latest[feat].values[0]:.4f}")
else:
    print("Ukraine not found in latest data!")
