"""
Retrain model with better features focusing on rates/proportions instead of absolute counts
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
print("RETRAINING WITH IMPROVED FEATURES (Rate-Based)")
print("=" * 80)

# Load data
print("\n1. Loading data...")
df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Countries: {df['Country'].nunique()}")

# Create better target variable
print("\n2. Creating improved target variable...")
# Use threshold on IsHighConflict_mean (proportion of high conflict events)
# A country-day with >15% high conflict events is considered a conflict day
# Lowered from 20% to capture Ukraine, Syria, Yemen which hover around 15-18%
df['target'] = (df['IsHighConflict_mean'] > 0.15).astype(int)

print(f"   Conflict rate: {df['target'].mean()*100:.2f}%")
print(f"   Class distribution:")
print(df['target'].value_counts())

# Feature engineering: Add rate-based and normalized features
print("\n3. Engineering better features...")

# Group by country to get baselines for normalization
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

# Event diversity (more diverse = potentially more chaotic)
df['EventDiversity'] = df['UniqueEventTypes'] / (df['EventCount'] + 1)
df['ActorDiversity'] = (df['UniqueActor1'] + df['UniqueActor2']) / (df['EventCount'] + 1)

# Rolling averages for temporal context (7-day window)
print("   Computing rolling features...")
df = df.sort_values(['Country', 'Date'])
for col in ['IsHighConflict_mean', 'GoldsteinScale_mean', 'AvgTone_mean', 'EventCount']:
    df[f'{col}_7day_avg'] = df.groupby('Country')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df[f'{col}_7day_std'] = df.groupby('Country')[col].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

# Trend indicators
df['HighConflict_trend'] = df['IsHighConflict_mean'] - df['IsHighConflict_mean_7day_avg']
df['Goldstein_trend'] = df['GoldsteinScale_mean'] - df['GoldsteinScale_mean_7day_avg']
df['Tone_trend'] = df['AvgTone_mean'] - df['AvgTone_mean_7day_avg']

print(f"   Total features after engineering: {len(df.columns)}")

# Select features for modeling
print("\n4. Selecting features for training...")

# EXCLUDE absolute count features that cause bias
exclude_features = [
    'Country', 'Date', 'target',
    # Exclude absolute sums that bias toward high-coverage countries
    'IsCooperation_sum', 'IsVerbalCoop_sum', 'IsMaterialCoop_sum',
    'IsHighConflict_sum', 'IsVerbalConflict_sum', 'IsMaterialConflict_sum',
    'IsHighIntensity_sum', 'IsPositiveEvent_sum', 'IsNegativeEvent_sum',
    'IsCrossBorder_sum', 'IsDomestic_sum', 'HasCoordinates_sum',
    'NumMentions_sum', 'NumSources_sum', 'NumArticles_sum',
    'IsQuad1_VerbalCoop_sum', 'IsQuad2_MaterialCoop_sum',
    'IsQuad3_VerbalConflict_sum', 'IsQuad4_MaterialConflict_sum',
    # Exclude baseline features used for normalization
    'EventCount_country_baseline', 'IsHighConflict_sum_country_baseline',
    'GoldsteinScale_mean_country_baseline', 'AvgTone_mean_country_baseline'
]

# Include only mean/proportion features and our engineered normalized features
feature_cols = [col for col in df.columns if col not in exclude_features]

print(f"   Using {len(feature_cols)} features")
print(f"   Excluded {len(exclude_features)} biased features")

# Handle missing values
df[feature_cols] = df[feature_cols].fillna(0)
df = df.replace([np.inf, -np.inf], 0)

# Prepare train/test split (time-based)
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

# Scale features
print("\n6. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
print("\n7. Balancing classes with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"   After SMOTE: {len(X_train_balanced):,} samples")
print(f"   Class distribution: {np.bincount(y_train_balanced)}")

# Train XGBoost
print("\n8. Training XGBoost classifier...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    scale_pos_weight=1,  # Already balanced with SMOTE
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
print("\n9. Evaluating model...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"{'='*60}")
print(f"ROC-AUC:   {roc_auc*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"{'='*60}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Conflict', 'Conflict']))

# Feature importance
print("\n10. Top 20 most important features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(20).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Save model
print("\n11. Saving model...")
model_dir = Path('saved_models')
model_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save model
model_path = model_dir / 'xgboost_model_improved.pkl'
joblib.dump(model, model_path)
print(f"   Model saved: {model_path}")

# Save scaler
scaler_path = model_dir / 'scaler_improved.pkl'
joblib.dump(scaler, scaler_path)
print(f"   Scaler saved: {scaler_path}")

# Save feature names
features_path = model_dir / 'feature_names_improved.json'
with open(features_path, 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"   Features saved: {features_path}")

# Save metadata
metadata = {
    'timestamp': timestamp,
    'model_type': 'XGBoost',
    'n_features': len(feature_cols),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'metrics': {
        'roc_auc': float(roc_auc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    },
    'feature_importance_top10': feature_importance.head(10).to_dict('records'),
    'target_definition': 'IsHighConflict_mean > 0.15',
    'improvements': [
        'Removed absolute count features (sum)',
        'Added country-normalized features',
        'Added conflict intensity ratios',
        'Added rolling averages and trends',
        'New target: >20% high conflict events'
    ]
}

metadata_path = model_dir / 'metadata_improved.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   Metadata saved: {metadata_path}")

# Test predictions on known conflict zones
print("\n12. Testing predictions on known conflict zones...")
conflict_countries = ['UKR', 'ISR', 'PSE', 'RUS', 'SYR', 'YEM', 'AFG']
recent_data = df[df['Date'] >= '2025-11-01']

for country in conflict_countries:
    country_data = recent_data[recent_data['Country'] == country]
    if len(country_data) > 0:
        X_country = country_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_country_scaled = scaler.transform(X_country)
        pred_proba = model.predict_proba(X_country_scaled)[:, 1]
        avg_prob = pred_proba.mean()
        max_prob = pred_proba.max()
        print(f"   {country}: avg={avg_prob*100:.2f}%, max={max_prob*100:.2f}%")

print("\n" + "="*80)
print("RETRAINING COMPLETE!")
print("="*80)
