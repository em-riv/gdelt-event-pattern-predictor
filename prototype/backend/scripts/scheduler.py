"""
Prediction Scheduler
Runs daily to generate predictions for all countries
"""

from datetime import datetime, timedelta, date as date_type
import numpy as np
from typing import List, Dict
import json

from database import get_db
from model_manager import ModelManager
from data_fetcher import GDELTDataFetcher


class PredictionScheduler:
    """Schedules and runs daily predictions"""

    def __init__(self):
        self.db = get_db()
        self.model_manager = ModelManager()
        self.data_fetcher = GDELTDataFetcher()

    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def run_daily_predictions(self, target_date: date_type = None):
        """
        Generate predictions for all countries

        This should be run daily via cron job or task scheduler
        """
        if target_date is None:
            target_date = date_type.today() + timedelta(days=1)  # Predict tomorrow

        prediction_date = date_type.today()

        print(f" Running predictions for {target_date}")
        print(f"   Prediction date: {prediction_date}")

        # Load model
        try:
            self.model_manager.load_model()
        except Exception as e:
            print(f"  Could not load model: {e}")
            print("   Training new model...")
            self.model_manager.train_and_save()

        # Get latest features for all countries
        all_features = self.db.get_all_latest_features()

        if len(all_features) == 0:
            print("  No data in database. Cannot generate predictions.")
            return []

        print(f" Generating predictions for {len(all_features)} countries...")

        predictions_made = []

        for row in all_features:
            country = row['country']

            # Parse features
            features_dict = json.loads(row['features_json']) if isinstance(row['features_json'], str) else eval(row['features_json'])

            # Extract feature values in correct order
            X = []
            for feat_name in self.model_manager.feature_cols:
                X.append(features_dict.get(feat_name, 0.0))

            X = np.array(X).reshape(1, -1)

            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale
            X_scaled = self.model_manager.scaler.transform(X)

            # Predict
            prob = self.model_manager.model.predict_proba(X_scaled)[0, 1]
            risk_level = self._get_risk_level(prob)
            confidence = max(prob, 1 - prob)  # Distance from 0.5

            # Save prediction
            self.db.insert_prediction(
                country=country,
                prediction_date=prediction_date,
                target_date=target_date,
                probability=float(prob),
                risk_level=risk_level,
                confidence=float(confidence),
                model_version=self.model_manager.db.get_active_model("XGBoost")['model_version']
            )

            predictions_made.append({
                'country': country,
                'probability': prob,
                'risk_level': risk_level
            })

        print(f" Generated {len(predictions_made)} predictions")

        # Print summary
        risk_counts = {}
        for pred in predictions_made:
            risk_counts[pred['risk_level']] = risk_counts.get(pred['risk_level'], 0) + 1

        print(f"\n Prediction Summary:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = risk_counts.get(level, 0)
            print(f"   {level}: {count} countries")

        return predictions_made

    def update_data_and_predict(self):
        """
        Complete daily workflow:
        1. Fetch latest GDELT data
        2. Update database
        3. Generate predictions
        """
        print("=" * 60)
        print("DAILY UPDATE & PREDICTION WORKFLOW")
        print("=" * 60)

        # Step 1: Update data
        print("\n Step 1: Updating latest data...")
        self.data_fetcher.update_latest_data()

        # Step 2: Generate predictions
        print("\n Step 2: Generating predictions...")
        predictions = self.run_daily_predictions()

        print("\n" + "=" * 60)
        print(f" Daily workflow complete!")
        print(f"   Predictions generated: {len(predictions)}")
        print("=" * 60)

        return predictions


def run_scheduler():
    """
    Main function to run scheduler
    Call this from cron job or Windows Task Scheduler
    """
    scheduler = PredictionScheduler()
    scheduler.update_data_and_predict()


if __name__ == "__main__":
    # Run the scheduler
    print(f"Starting prediction scheduler at {datetime.now()}")
    run_scheduler()
