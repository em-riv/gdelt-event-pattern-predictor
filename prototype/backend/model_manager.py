"""
Model Manager
Handles training, saving, and loading ML models
"""

import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import json

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from database import get_db


class ModelManager:
    """Manages ML model lifecycle"""

    def __init__(self, models_dir: str = "saved_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.db = get_db()

        self.model = None
        self.scaler = None
        self.feature_cols = []
        self.metrics = {}

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load data from database and prepare for training
        Returns: X_train, X_test, y_train, y_test, feature_cols
        """
        print(" Preparing training data from database...")

        # Get ALL features from database (not just latest)
        all_features = self.db.get_all_features()

        if len(all_features) == 0:
            raise ValueError("No data in database. Run data_fetcher.py first to populate database.")

        # Convert to DataFrame
        data = []
        for row in all_features:
            # Parse JSON properly
            if isinstance(row['features_json'], str):
                features_dict = json.loads(row['features_json'])
            else:
                features_dict = row['features_json'] if row['features_json'] else {}
            features_dict['Country'] = row['country']
            features_dict['Date'] = row['date']
            data.append(features_dict)

        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Country', 'Date'])

        # Create target
        conflict_col = 'IsHighConflict_sum'
        df['NextDay_Conflict'] = df.groupby('Country')[conflict_col].shift(-1)
        df['NextDay_HighConflict'] = (
            df['NextDay_Conflict'] >= df['NextDay_Conflict'].quantile(0.75)
        ).astype(int)

        # Drop NaN
        df = df.dropna(subset=['NextDay_Conflict', 'NextDay_HighConflict'])

        # Select features
        exclude_cols = ['Country', 'Date', 'NextDay_Conflict', 'NextDay_HighConflict']
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        # Prepare X and y
        X = df[feature_cols].copy()
        y = df['NextDay_HighConflict'].copy()

        # Handle missing/infinite
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Time-based split
        train_mask = df['Date'] < '2024-01-01'
        test_mask = df['Date'] >= '2024-01-01'

        X_train = X[train_mask].values
        X_test = X[test_mask].values
        y_train = y[train_mask].values
        y_test = y[test_mask].values

        print(f" Data prepared:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Train samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Train positive rate: {y_train.mean()*100:.1f}%")

        return X_train, X_test, y_train, y_test, feature_cols

    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print(" Training XGBoost model...")

        # Calculate class weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"   Scale pos weight: {scale_pos_weight:.2f}")

        # Train scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train XGBoost
        self.model = xgb.XGBClassifier(
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

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': X_train.shape[1]
        }

        print(f" Model trained!")
        print(f"   ROC-AUC: {self.metrics['roc_auc']*100:.2f}%")
        print(f"   F1 Score: {self.metrics['f1']*100:.2f}%")
        print(f"   Precision: {self.metrics['precision']*100:.2f}%")
        print(f"   Recall: {self.metrics['recall']*100:.2f}%")

        return self.metrics

    def save_model(self, version: Optional[str] = None):
        """Save model, scaler, and metadata to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_name = "XGBoost"
        model_filename = f"{model_name}_{version}.pkl"
        scaler_filename = f"scaler_{version}.pkl"

        model_path = self.models_dir / model_filename
        scaler_path = self.models_dir / scaler_filename

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save feature columns
        features_path = self.models_dir / f"features_{version}.txt"
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_cols))

        # Save metadata to database
        self.db.save_model_metadata(
            model_name=model_name,
            model_version=version,
            model_path=str(model_path),
            metrics=self.metrics
        )

        print(f" Model saved:")
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Features: {features_path}")

        return str(model_path)

    def load_model(self, version: Optional[str] = None):
        """Load model from disk"""
        if version is None:
            # Load latest active model from database
            model_meta = self.db.get_active_model("XGBoost")

            if model_meta is None:
                raise ValueError("No active model found in database. Train a model first.")

            version = model_meta['model_version']
            print(f"[MODEL] Loading active model version: {version}")

        model_filename = f"XGBoost_{version}.pkl"
        scaler_filename = f"scaler_{version}.pkl"
        features_filename = f"features_{version}.txt"

        model_path = self.models_dir / model_filename
        scaler_path = self.models_dir / scaler_filename
        features_path = self.models_dir / features_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load feature columns
        with open(features_path, 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]

        # Load metrics from database
        model_meta = self.db.get_active_model("XGBoost")
        if model_meta:
            import json
            self.metrics = json.loads(model_meta['metrics_json'])

        print(f" Model loaded:")
        print(f"   Version: {version}")
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   ROC-AUC: {self.metrics.get('roc_auc', 0)*100:.2f}%")

        return version

    def train_and_save(self):
        """Complete workflow: prepare data, train, and save model"""
        # Prepare data
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_training_data()
        self.feature_cols = feature_cols

        # Train
        self.train_model(X_train, y_train, X_test, y_test)

        # Save
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_model(version)

        return version


if __name__ == "__main__":
    # Example usage: Train and save a model
    print("Training new model...")

    manager = ModelManager()
    version = manager.train_and_save()

    print(f"\n Model training complete!")
    print(f"   Version: {version}")
    print(f"   Metrics: {manager.metrics}")
