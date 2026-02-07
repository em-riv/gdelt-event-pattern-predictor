"""
Detailed analysis of which features are causing NaN values in GRU predictions
"""
import pandas as pd
import numpy as np
import json

print("=" * 60)
print("ANALYZING NaN FEATURES IN GRU MODEL")
print("=" * 60)

# Load the expected features
with open('saved_models/features_gru_forecast.json', 'r') as f:
    expected_features = json.load(f)

# Load the data
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
print(f"Original data shape: {df.shape}")

# Sort by country and date
df = df.sort_values(['Country', 'Date'])

# Apply the same feature engineering as in the generation script
print(f"Applying feature engineering...")

# Country baselines
country_stats = df.groupby('Country').agg({
    'EventCount': 'mean',
    'IsHighConflict_sum': 'mean', 
    'GoldsteinScale_mean': 'mean',
    'AvgTone_mean': 'mean'
}).add_suffix('_country_baseline')

df = df.merge(country_stats, left_on='Country', right_index=True)

# Check baseline features
print(f"\nCountry baseline features:")
for col in ['EventCount_country_baseline', 'IsHighConflict_sum_country_baseline', 
            'GoldsteinScale_mean_country_baseline', 'AvgTone_mean_country_baseline']:
    nan_count = df[col].isna().sum()
    print(f"  {col}: {nan_count} NaNs")

# Normalized features
df['EventCount_normalized'] = df['EventCount'] / (df['EventCount_country_baseline'] + 1)
df['HighConflict_vs_baseline'] = df['IsHighConflict_sum'] / (df['IsHighConflict_sum_country_baseline'] + 1)
df['GoldsteinScale_vs_baseline'] = df['GoldsteinScale_mean'] - df['GoldsteinScale_mean_country_baseline']
df['AvgTone_vs_baseline'] = df['AvgTone_mean'] - df['AvgTone_mean_country_baseline']

# Check normalized features
print(f"\nNormalized features:")
for col in ['EventCount_normalized', 'HighConflict_vs_baseline', 
            'GoldsteinScale_vs_baseline', 'AvgTone_vs_baseline']:
    nan_count = df[col].isna().sum()
    print(f"  {col}: {nan_count} NaNs")

# Ratios
df['ConflictRatio'] = df['IsHighConflict_mean'] / (df['IsCooperation_mean'] + 0.01)
df['MaterialConflictRatio'] = df['IsMaterialConflict_mean'] / (df['IsMaterialCoop_mean'] + 0.01)
df['VerbalConflictRatio'] = df['IsVerbalConflict_mean'] / (df['IsVerbalCoop_mean'] + 0.01)
df['EventDiversity'] = df['UniqueEventTypes'] / (df['EventCount'] + 1)
df['ActorDiversity'] = (df['UniqueActor1'] + df['UniqueActor2']) / (df['EventCount'] + 1)

# Check ratio features
print(f"\nRatio features:")
for col in ['ConflictRatio', 'MaterialConflictRatio', 'VerbalConflictRatio', 
            'EventDiversity', 'ActorDiversity']:
    nan_count = df[col].isna().sum()
    print(f"  {col}: {nan_count} NaNs")

# Rolling features - this is likely where the NaNs come from
print(f"\nApplying rolling features...")
for col in ['IsHighConflict_mean', 'GoldsteinScale_mean', 'AvgTone_mean', 'EventCount']:
    df[f'{col}_7day_avg'] = df.groupby('Country')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df[f'{col}_7day_std'] = df.groupby('Country')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

# Check rolling features
print(f"\nRolling average features:")
for col in ['IsHighConflict_mean_7day_avg', 'GoldsteinScale_mean_7day_avg', 
            'AvgTone_mean_7day_avg', 'EventCount_7day_avg']:
    nan_count = df[col].isna().sum()
    print(f"  {col}: {nan_count} NaNs")

print(f"\nRolling std features:")
for col in ['IsHighConflict_mean_7day_std', 'GoldsteinScale_mean_7day_std', 
            'AvgTone_mean_7day_std', 'EventCount_7day_std']:
    nan_count = df[col].isna().sum()
    print(f"  {col}: {nan_count} NaNs")

# Trends
df['HighConflict_trend'] = df['IsHighConflict_mean'] - df['IsHighConflict_mean_7day_avg']
df['Goldstein_trend'] = df['GoldsteinScale_mean'] - df['GoldsteinScale_mean_7day_avg']
df['Tone_trend'] = df['AvgTone_mean'] - df['AvgTone_mean_7day_avg']

print(f"\nTrend features:")
for col in ['HighConflict_trend', 'Goldstein_trend', 'Tone_trend']:
    nan_count = df[col].isna().sum()
    print(f"  {col}: {nan_count} NaNs")

print(f"\nAfter all feature engineering: {df.shape}")

# Test with USA last 7 days
test_country = 'USA'
country_df = df[df['Country'] == test_country].sort_values('Date').tail(7)
print(f"\n{test_country} test data:")
print(f"Shape: {country_df.shape}")
print(f"Date range: {country_df['Date'].min()} to {country_df['Date'].max()}")

# Check which expected features have NaN values
print(f"\nNaN analysis for {test_country} (last 7 days):")
nan_features = []
for feat in expected_features:
    if feat in country_df.columns:
        nan_count = country_df[feat].isna().sum()
        if nan_count > 0:
            nan_features.append((feat, nan_count))
            print(f"  {feat}: {nan_count}/7 NaNs")

print(f"\nTotal NaN features: {len(nan_features)}")
print(f"Total NaN values: {sum([count for _, count in nan_features])}")

# Show the actual values for a few problematic features
if nan_features:
    print(f"\nSample NaN feature values:")
    for feat, nan_count in nan_features[:5]:
        print(f"  {feat}:")
        values = country_df[feat].values
        print(f"    {values}")

print("\n" + "=" * 60)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 60)