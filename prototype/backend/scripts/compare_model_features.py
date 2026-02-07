"""
Compare feature sets between XGBoost and GRU models to identify discrepancies
"""
import json

print("=" * 80)
print("COMPARING MODEL FEATURE SETS")
print("=" * 80)

# Load XGBoost features
with open('saved_models/features_forecast.json', 'r') as f:
    xgboost_features = json.load(f)

# Load GRU features  
with open('saved_models/features_gru_forecast.json', 'r') as f:
    gru_features = json.load(f)

print(f"XGBoost model features: {len(xgboost_features)}")
print(f"GRU model features: {len(gru_features)}")

# Find differences
xgboost_only = set(xgboost_features) - set(gru_features)
gru_only = set(gru_features) - set(xgboost_features)
common = set(xgboost_features) & set(gru_features)

print(f"\nCommon features: {len(common)}")
print(f"XGBoost-only features: {len(xgboost_only)}")
print(f"GRU-only features: {len(gru_only)}")

if xgboost_only:
    print(f"\nFEATURES ONLY IN XGBOOST ({len(xgboost_only)}):")
    for feat in sorted(xgboost_only):
        print(f"  - {feat}")

if gru_only:
    print(f"\nFEATURES ONLY IN GRU ({len(gru_only)}):")
    for feat in sorted(gru_only):
        print(f"  - {feat}")

# Now test which features work with current data
import pandas as pd
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
usa_recent = df[df['Country'] == 'USA'].tail(1)

print(f"\n" + "="*50)
print("TESTING FEATURE AVAILABILITY IN CURRENT DATA")
print("="*50)

# Test XGBoost features
xgb_available = 0
xgb_missing = []
for feat in xgboost_features:
    if feat in df.columns and not usa_recent[feat].isna().iloc[0]:
        xgb_available += 1
    else:
        xgb_missing.append(feat)

print(f"\nXGBoost features in current data: {xgb_available}/{len(xgboost_features)} ({xgb_available/len(xgboost_features)*100:.1f}%)")
if xgb_missing:
    print(f"XGBoost missing features ({len(xgb_missing)}):")
    for feat in xgb_missing[:10]:  # Show first 10
        print(f"  - {feat}")
    if len(xgb_missing) > 10:
        print(f"  ... and {len(xgb_missing)-10} more")

# Test GRU features  
gru_available = 0
gru_missing = []
for feat in gru_features:
    if feat in df.columns and not usa_recent[feat].isna().iloc[0]:
        gru_available += 1
    else:
        gru_missing.append(feat)

print(f"\nGRU features in current data: {gru_available}/{len(gru_features)} ({gru_available/len(gru_features)*100:.1f}%)")
if gru_missing:
    print(f"GRU missing features ({len(gru_missing)}):")
    for feat in gru_missing[:10]:  # Show first 10
        print(f"  - {feat}")
    if len(gru_missing) > 10:
        print(f"  ... and {len(gru_missing)-10} more")

print(f"\n" + "="*80)
print("FEATURE COMPARISON COMPLETE")
print("="*80)