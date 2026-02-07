"""
Debug GRU model predictions to find why all predictions are NaN
"""
import pandas as pd
import numpy as np
import json
import joblib
import torch
from pathlib import Path

print("=" * 60)
print("DEBUGGING GRU PREDICTIONS")
print("=" * 60)

# Load the expected features
with open('saved_models/features_gru_forecast.json', 'r') as f:
    expected_features = json.load(f)

print(f"Expected features: {len(expected_features)}")
print("First 10 expected features:")
for i, feat in enumerate(expected_features[:10]):
    print(f"  {i+1:2d}. {feat}")

# Load the data
print(f"\nLoading data...")
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
print(f"Data shape: {df.shape}")
print(f"Data columns: {len(df.columns)}")

# Sort by country and date
df = df.sort_values(['Country', 'Date'])

# Apply the same feature engineering as in the generation script
print(f"\nApplying feature engineering...")

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

print(f"After feature engineering: {df.shape}")

# Check which expected features are missing
available_features = df.columns.tolist()
missing_features = [feat for feat in expected_features if feat not in available_features]
extra_features = [feat for feat in available_features if feat not in expected_features]

print(f"\nFeature analysis:")
print(f"Expected: {len(expected_features)}")
print(f"Available: {len(available_features)}")
print(f"Missing: {len(missing_features)}")
print(f"Extra: {len(extra_features)}")

if missing_features:
    print(f"\nMISSING FEATURES ({len(missing_features)}):")
    for feat in missing_features:
        print(f"  - {feat}")

# Test with one country to see what happens
test_country = 'USA'
print(f"\nTesting with {test_country}...")

country_df = df[df['Country'] == test_country].sort_values('Date').tail(7)
print(f"Country data shape: {country_df.shape}")

if len(country_df) > 0:
    print(f"Date range: {country_df['Date'].min()} to {country_df['Date'].max()}")
    
    # Try to extract features
    try:
        X = country_df[expected_features].values
        print(f"Features extracted: {X.shape}")
        print(f"Features stats:")
        print(f"  NaN values: {np.isnan(X).sum()}")
        print(f"  Inf values: {np.isinf(X).sum()}")
        print(f"  Min: {np.nanmin(X)}")
        print(f"  Max: {np.nanmax(X)}")
        
        # Load scaler and try scaling
        scaler = joblib.load('saved_models/scaler_gru_forecast.pkl')
        X_scaled = scaler.transform(X)
        print(f"After scaling:")
        print(f"  NaN values: {np.isnan(X_scaled).sum()}")
        print(f"  Inf values: {np.isinf(X_scaled).sum()}")
        print(f"  Min: {np.nanmin(X_scaled)}")
        print(f"  Max: {np.nanmax(X_scaled)}")
        
        # Create tensor and test model
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
        print(f"Tensor shape: {X_tensor.shape}")
        print(f"Tensor stats:")
        print(f"  NaN: {torch.isnan(X_tensor).sum()}")
        print(f"  Inf: {torch.isinf(X_tensor).sum()}")
        
        # Load and test model
        from generate_predictions_gru import GRUClassifier
        model = GRUClassifier(len(expected_features))
        model.load_state_dict(torch.load('saved_models/gru_forecast.pth', map_location='cpu'))
        model.eval()
        
        with torch.no_grad():
            logits = model(X_tensor)
            prob = torch.sigmoid(logits)
            print(f"Model output:")
            print(f"  Logits: {logits}")
            print(f"  Probability: {prob}")
            print(f"  Is NaN: {torch.isnan(prob)}")
            
    except Exception as e:
        print(f"Error during feature extraction/prediction: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)