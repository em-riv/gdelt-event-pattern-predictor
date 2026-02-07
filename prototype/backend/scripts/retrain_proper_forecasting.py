"""
Proper forecasting model: Predict FUTURE conflict from CURRENT features
Fixes data leakage by shifting target forward
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
from datetime import datetime
import json

print("=" * 80)
print("TRAINING PROPER FORECASTING MODEL (No Data Leakage)")
print("=" * 80)

# Load data
print("\n1. Loading data...")
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Create proper target: Predict conflict 1 day ahead
print("\n2. Creating FUTURE target (1-day ahead forecast)...")
df = df.sort_values(['Country', 'Date'])

# Shift IsHighConflict_mean forward by 1 day within each country
df['target_next_day'] = df.groupby('Country')['IsHighConflict_mean'].shift(-1)

# Create binary target
df['target'] = (df['target_next_day'] > 0.15).astype(int)

# Remove last day for each country (no future data)
df = df[df['target_next_day'].notna()]

print(f"   Samples after removing last day: {len(df):,}")
print(f"   Conflict rate: {df['target'].mean()*100:.2f}%")
print(f"   Class distribution:")
print(df['target'].value_counts())

# Feature engineering (without using target-day data)
print("\n3. Engineering features from CURRENT day only...")

# Group by country to get baselines
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

# Conflict ratios (current day)
df['ConflictRatio'] = df['IsHighConflict_mean'] / (df['IsCooperation_mean'] + 0.01)
df['MaterialConflictRatio'] = df['IsMaterialConflict_mean'] / (df['IsMaterialCoop_mean'] + 0.01)
df['VerbalConflictRatio'] = df['IsVerbalConflict_mean'] / (df['IsVerbalCoop_mean'] + 0.01)

# Diversity
df['EventDiversity'] = df['UniqueEventTypes'] / (df['EventCount'] + 1)
df['ActorDiversity'] = (df['UniqueActor1'] + df['UniqueActor2']) / (df['EventCount'] + 1)

# Rolling features (using past data only)
print("   Computing rolling features from past...")
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

print(f"   Total features: {len(df.columns)}")

# Select features (EXCLUDING IsHighConflict_mean which causes leakage)
print("\n4. Selecting features (EXCLUDING target-related features)...")

exclude_features = [
    'Country', 'Date', 'target', 'target_next_day',
    # CRITICAL: Exclude features that would leak future information
    'IsHighConflict_mean',  # This is too correlated with next-day conflict
    'IsHighConflict_sum',
    # Exclude other sums
    'IsCooperation_sum', 'IsVerbalCoop_sum', 'IsMaterialCoop_sum',
    'IsVerbalConflict_sum', 'IsMaterialConflict_sum',
    'IsHighIntensity_sum', 'IsPositiveEvent_sum', 'IsNegativeEvent_sum',
    'IsCrossBorder_sum', 'IsDomestic_sum', 'HasCoordinates_sum',
    'NumMentions_sum', 'NumSources_sum', 'NumArticles_sum',
    'IsQuad1_VerbalCoop_sum', 'IsQuad2_MaterialCoop_sum',
    'IsQuad3_VerbalConflict_sum', 'IsQuad4_MaterialConflict_sum',
    # Baselines
    'EventCount_country_baseline', 'IsHighConflict_sum_country_baseline',
    'GoldsteinScale_mean_country_baseline', 'AvgTone_mean_country_baseline'
]

feature_cols = [col for col in df.columns if col not in exclude_features]

print(f"   Using {len(feature_cols)} features")
print(f"   Excluded {len(exclude_features)} features to prevent leakage")

# Handle missing
df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

# Time-based split
print("\n5. Splitting data (time-based)...")
split_date = pd.Timestamp('2025-10-01')
train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

X_train = train_df[feature_cols]
y_train = train_df['target']
X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"   Train: {len(X_train):,} samples ({y_train.sum():,} conflicts, {y_train.mean()*100:.2f}%)")
print(f"   Test:  {len(X_test):,} samples ({y_test.sum():,} conflicts, {y_test.mean()*100:.2f}%)")

# Scale
print("\n6. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
print("\n7. Balancing with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"   After SMOTE: {len(X_train_balanced):,} samples")

# Train
print("\n8. Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

model.fit(
    X_train_balanced,
    y_train_balanced,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

# Evaluate
print("\n9. Evaluating...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"RESULTS (1-DAY AHEAD FORECASTING):")
print(f"{'='*60}")
print(f"ROC-AUC:   {roc_auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"{'='*60}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Conflict', 'Conflict']))

# Feature importance
print("\n10. Top 20 features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(20).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Save
print("\n11. Saving model...")
model_dir = Path('saved_models')
joblib.dump(model, model_dir / 'xgboost_forecast.pkl')
joblib.dump(scaler, model_dir / 'scaler_forecast.pkl')
with open(model_dir / 'features_forecast.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

metadata = {
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'model_type': 'XGBoost Forecasting (1-day ahead)',
    'target': 'IsHighConflict_mean > 0.15 (NEXT DAY)',
    'metrics': {
        'roc_auc': float(roc_auc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    },
    'features_excluded': ['IsHighConflict_mean - prevents leakage'],
    'top_features': feature_importance.head(10).to_dict('records')
}

with open(model_dir / 'metadata_forecast.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   Model saved!")

print("\n" + "="*80)
print("PROPER FORECASTING MODEL COMPLETE!")
print("="*80)
