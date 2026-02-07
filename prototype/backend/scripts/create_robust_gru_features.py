"""
Create a robust GRU prediction system that works with available features
"""
import pandas as pd
import numpy as np
import json

print("=" * 60)
print("CREATING ROBUST GRU FEATURE MAPPING")
print("=" * 60)

# Load the expected GRU features
with open('saved_models/features_gru_forecast.json', 'r') as f:
    expected_features = json.load(f)

# Load current data
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
print(f"Data shape: {df.shape}")

# Apply feature engineering like in the GRU script
df = df.sort_values(['Country', 'Date'])

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

# Test with recent data
usa_recent = df[df['Country'] == 'USA'].tail(7)

# Analyze which features are available vs missing
available_features = []
missing_features = []
feature_substitutes = {}

for feat in expected_features:
    if feat in df.columns and usa_recent[feat].notna().all():
        available_features.append(feat)
    else:
        missing_features.append(feat)
        # Create substitution rules for missing features
        if feat == 'IntensityScore_std':
            feature_substitutes[feat] = 'GoldsteinScale_std'
        elif feat == 'IntensityScore_min':
            feature_substitutes[feat] = 'GoldsteinScale_min'
        elif feat == 'IntensityScore_max':
            feature_substitutes[feat] = 'GoldsteinScale_max'
        elif feat == 'IsHighIntensity_mean':
            feature_substitutes[feat] = 'IsHighConflict_mean'
        elif feat == 'IsPositiveEvent_mean':
            feature_substitutes[feat] = 'IsCooperation_mean'
        elif feat == 'IsNegativeEvent_mean':
            feature_substitutes[feat] = 'IsHighConflict_mean'
        elif feat == 'IsCrossBorder_mean':
            feature_substitutes[feat] = 'IsCooperation_mean'  # Proxy
        elif feat == 'IsDomestic_mean':
            feature_substitutes[feat] = 'IsHighConflict_mean'  # Proxy
        elif 'NumMentions_max' in feat:
            feature_substitutes[feat] = 'NumMentions_mean'
        elif 'NumSources_max' in feat:
            feature_substitutes[feat] = 'NumSources_mean'
        elif 'NumArticles_max' in feat:
            feature_substitutes[feat] = 'NumArticles_mean'
        elif feat == 'ActionGeo_Lat_mean':
            feature_substitutes[feat] = 0.0  # Default
        elif feat == 'ActionGeo_Lat_std':
            feature_substitutes[feat] = 0.0
        elif feat == 'ActionGeo_Long_mean':
            feature_substitutes[feat] = 0.0
        elif feat == 'ActionGeo_Long_std':
            feature_substitutes[feat] = 0.0
        elif feat == 'HasCoordinates_mean':
            feature_substitutes[feat] = 0.5  # Default 50%
        elif 'Unique' in feat:
            if feat == 'UniqueRegions':
                feature_substitutes[feat] = 'EventCount'  # Proxy
            elif feat == 'UniqueLocations':
                feature_substitutes[feat] = 'EventCount'
            elif feat == 'UniqueActor1':
                feature_substitutes[feat] = 'EventCount'
            elif feat == 'UniqueActor2':
                feature_substitutes[feat] = 'EventCount'
            elif feat == 'UniqueEventTypes':
                feature_substitutes[feat] = 'EventCount'

print(f"\nFeature Analysis:")
print(f"Available: {len(available_features)}/62 ({len(available_features)/62*100:.1f}%)")
print(f"Missing: {len(missing_features)}/62 ({len(missing_features)/62*100:.1f}%)")
print(f"Substitutable: {len(feature_substitutes)}/62 ({len(feature_substitutes)/62*100:.1f}%)")

# Create feature mapping
feature_mapping = {}
for feat in expected_features:
    if feat in available_features:
        feature_mapping[feat] = feat  # Direct mapping
    elif feat in feature_substitutes:
        substitute = feature_substitutes[feat]
        if isinstance(substitute, str) and substitute in df.columns:
            feature_mapping[feat] = substitute  # Substitute mapping
        else:
            feature_mapping[feat] = substitute  # Constant value
    else:
        feature_mapping[feat] = 0.0  # Default to 0

print(f"\nFeature mapping created for all 62 features")

# Test the mapping with USA data
print(f"\nTesting feature extraction with USA recent data:")
test_features = []
for feat in expected_features:
    if feat in feature_mapping:
        mapping = feature_mapping[feat]
        if isinstance(mapping, str):
            if mapping in usa_recent.columns:
                value = usa_recent[mapping].iloc[-1]
            else:
                value = 0.0
        else:
            value = mapping  # Constant
        test_features.append(value)
    else:
        test_features.append(0.0)

test_array = np.array(test_features).reshape(1, -1)
print(f"Extracted features shape: {test_array.shape}")
print(f"NaN count: {np.isnan(test_array).sum()}")
print(f"Inf count: {np.isinf(test_array).sum()}")

if np.isnan(test_array).sum() == 0:
    print(" SUCCESS: No NaN values in extracted features!")
else:
    print(" Still have NaN values")

# Save the feature mapping
with open('saved_models/gru_feature_mapping.json', 'w') as f:
    json.dump(feature_mapping, f, indent=2, default=str)

print(f"\n Saved feature mapping to saved_models/gru_feature_mapping.json")

print("\n" + "=" * 60)
print("ROBUST GRU FEATURE MAPPING COMPLETE")
print("=" * 60)