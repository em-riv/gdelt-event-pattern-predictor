"""
Update Production XGBoost Model and Generate Predictions
Simplified version using only XGBoost (no PyTorch dependency)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# XGBoost
import xgboost as xgb

# Import database
from database import get_db

# Constants
DATA_PATH = Path(r"C:\Users\Emman\Documents\AI_dev\GDELT_ConflictPredictor\data\features_multiresolution")
MODELS_PATH = Path(r"C:\Users\Emman\Documents\AI_dev\GDELT_ConflictPredictor\prototype\backend\saved_models")
MODELS_PATH.mkdir(exist_ok=True)

print("="*80)
print("UPDATING PRODUCTION XGBOOST MODEL")
print("="*80)

# ==================== LOAD DATA ====================
print("\n[1/7] Loading data...")
# Use 2023-2025 data
df = pd.read_parquet(DATA_PATH / "country_day" / "country_day_features_2023_2025.parquet")
print(f"      Loaded {len(df):,} observations, {len(df.columns)} features")
print(f"      Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"      Countries: {df['Country'].nunique()}")

# ==================== PREPARE DATA ====================
print("\n[2/7] Preparing data...")
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
y = df_ml['NextDay_HighConflict'].copy()

# Handle missing values
X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"      Features: {len(feature_cols)}, Samples: {X.shape[0]:,}")

# ==================== TRAIN/TEST SPLIT ====================
print("\n[3/7] Splitting data (last 30 days for testing)...")
latest_date = df_ml['Date'].max()
test_start_date = latest_date - timedelta(days=30)

train_mask = df_ml['Date'] < test_start_date
test_mask = df_ml['Date'] >= test_start_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)

print(f"      Train: {len(X_train):,}, Test: {len(X_test):,}")

# ==================== TRAIN XGBOOST ====================
print("\n[4/7] Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"      Scale pos weight: {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc'
)
xgb_model.fit(X_train, y_train)

# Evaluate
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)

print(f"\n      Test Results:")
print(f"        Accuracy:  {accuracy_score(y_test, y_pred_xgb)*100:.2f}%")
print(f"        Precision: {precision_score(y_test, y_pred_xgb)*100:.2f}%")
print(f"        Recall:    {recall_score(y_test, y_pred_xgb)*100:.2f}%")
print(f"        F1 Score:  {f1_score(y_test, y_pred_xgb)*100:.2f}%")
print(f"        ROC-AUC:   {roc_auc_score(y_test, y_prob_xgb)*100:.2f}%")

# Save XGBoost
xgb_path = MODELS_PATH / "xgboost_model.pkl"
with open(xgb_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"\n      Saved model to {xgb_path}")

# Save scaler and feature names
scaler_path = MODELS_PATH / "scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

feature_names_path = MODELS_PATH / "feature_names.json"
with open(feature_names_path, 'w') as f:
    json.dump(feature_cols, f)

print(f"      Saved scaler to {scaler_path}")
print(f"      Saved feature names to {feature_names_path}")

# ==================== GENERATE PREDICTIONS ====================
print("\n[5/7] Generating predictions for all data...")
all_probs = xgb_model.predict_proba(X_all_scaled)[:, 1]
all_preds = (all_probs >= 0.5).astype(int)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'country': df_ml['Country'].values,
    'date': df_ml['Date'].values,
    'prediction_date': pd.to_datetime(df_ml['Date'].values) + timedelta(days=1),
    'conflict_probability': all_probs,
    'predicted_conflict': all_preds,
    'risk_level': pd.cut(all_probs,
                        bins=[0, 0.25, 0.5, 0.75, 1.0],
                        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']).astype(str),
    'model_name': 'XGBoost',
    'model_version': '1.0.0'
})

print(f"      Generated {len(predictions_df):,} predictions")
print(f"\n      Risk level distribution:")
for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
    count = (predictions_df['risk_level'] == level).sum()
    pct = count / len(predictions_df) * 100
    print(f"        {level:8s}: {count:6,} ({pct:5.2f}%)")

# ==================== UPDATE DATABASE ====================
print("\n[6/7] Updating database...")
db = get_db()

print("      Clearing old predictions...")
db.conn.execute("DELETE FROM predictions")
db.conn.commit()

print("      Inserting new predictions...")
insert_count = 0
for _, row in predictions_df.iterrows():
    try:
        db.conn.execute("""
            INSERT INTO predictions (
                country, date, prediction_date, conflict_probability,
                predicted_conflict, risk_level, model_name, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['country'],
            row['date'].strftime('%Y-%m-%d'),
            row['prediction_date'].strftime('%Y-%m-%d'),
            float(row['conflict_probability']),
            int(row['predicted_conflict']),
            row['risk_level'],
            row['model_name'],
            row['model_version']
        ))
        insert_count += 1
        if insert_count % 10000 == 0:
            print(f"        {insert_count:,} inserted...")
            db.conn.commit()
    except Exception as e:
        print(f"        Error: {e}")
        continue

db.conn.commit()
print(f"      Inserted {insert_count:,} predictions total")

# ==================== UPDATE MODEL METADATA ====================
print("\n[7/7] Updating model metadata...")

metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred_xgb)),
    'precision': float(precision_score(y_test, y_pred_xgb)),
    'recall': float(recall_score(y_test, y_pred_xgb)),
    'f1_score': float(f1_score(y_test, y_pred_xgb)),
    'roc_auc': float(roc_auc_score(y_test, y_prob_xgb))
}

# Update models table
db.conn.execute("DELETE FROM models WHERE model_name = 'XGBoost'")
db.conn.execute("""
    INSERT INTO models (
        model_name, model_version, model_path, trained_at,
        train_samples, test_samples, metrics, is_active
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    'XGBoost',
    '1.0.0',
    str(xgb_path),
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    len(X_train),
    len(X_test),
    json.dumps(metrics),
    1
))
db.conn.commit()

print(f"      Updated model metadata")

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("UPDATE COMPLETE!")
print("="*80)

summary = db.get_data_summary()
print(f"\nDatabase Summary:")
print(f"  Total predictions: {summary.get('total_predictions', 0):,}")
print(f"  Total countries:   {summary.get('total_countries', 0):,}")
print(f"  Latest prediction: {summary.get('latest_prediction', 'N/A')}")
print(f"  Date range:        {summary.get('date_range', {}).get('min', 'N/A')} to {summary.get('date_range', {}).get('max', 'N/A')}")

print(f"\nModel Performance:")
print(f"  XGBoost - ROC-AUC: {metrics['roc_auc']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%")

print(f"\nSaved Files:")
print(f"  Model:         {xgb_path}")
print(f"  Scaler:        {scaler_path}")
print(f"  Feature names: {feature_names_path}")

print("\n" + "="*80)
print("Restart the backend server to use the new model!")
print("="*80)
