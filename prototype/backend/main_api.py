"""
GDELT Conflict Predictor API (Database Version)
FastAPI backend using database for predictions
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime, date as date_type, timedelta
import json

from models import (
    ConflictPrediction,
    CountryRiskScore,
    ModelPerformance,
    PredictionResponse,
    FeatureContribution,
    ModelMetrics
)
from database import get_db
from model_manager import ModelManager

# Initialize FastAPI app
app = FastAPI(
    title="GDELT Conflict Predictor API",
    description="Next-day conflict prediction using database-stored predictions",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and model manager
db = get_db()
model_manager = ModelManager()

# Try to load model on startup
try:
    model_manager.load_model()
    print("[OK] Model loaded from disk")
except Exception as e:
    print(f"[WARNING] Could not load model: {e}")
    print("          Model will be loaded on first prediction request")


@app.get("/")
async def root():
    """API health check"""
    stats = db.get_data_summary()
    model_info = db.get_active_model("XGBoost")

    return {
        "status": "online",
        "version": "2.0.0",
        "database": {
            "total_features": stats.get('total_features', 0),
            "total_countries": stats.get('total_countries', 0),
            "total_predictions": stats.get('total_predictions', 0),
            "latest_prediction": stats.get('latest_prediction'),
            "date_range": stats.get('date_range')
        },
        "model": {
            "name": model_info['model_name'] if model_info else None,
            "version": model_info['model_version'] if model_info else None,
            "trained_at": model_info['trained_at'] if model_info else None
        } if model_info else None,
        "endpoints": [
            "/api/predictions/latest",
            "/api/predictions/country/{country}",
            "/api/risk-scores",
            "/api/model-performance"
        ]
    }


@app.get("/api/predictions/latest", response_model=List[ConflictPrediction])
async def get_latest_predictions(
    min_probability: Optional[float] = Query(None, ge=0, le=1),
    risk_levels: Optional[str] = Query(None),
    countries: Optional[str] = Query(None)
):
    """
    Get latest predictions for all countries with optional filters

    Query Parameters:
    - min_probability: Minimum conflict probability (0-1)
    - risk_levels: Comma-separated risk levels (e.g., "HIGH,CRITICAL")
    - countries: Comma-separated country names
    """
    try:
        # Get predictions from database
        predictions_data = db.get_latest_predictions()

        predictions = []
        for row in predictions_data:
            pred = ConflictPrediction(
                country=row['country'],
                date=row['target_date'],
                prediction_date=row['prediction_date'],
                conflict_probability=row['conflict_probability'],
                risk_level=row['risk_level'],
                confidence=row['confidence']
            )

            # Apply filters
            if min_probability and pred.conflict_probability < min_probability:
                continue

            if risk_levels:
                allowed_levels = [level.strip().upper() for level in risk_levels.split(',')]
                if pred.risk_level not in allowed_levels:
                    continue

            if countries:
                allowed_countries = [c.strip() for c in countries.split(',')]
                if pred.country not in allowed_countries:
                    continue

            predictions.append(pred)

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/country/{country}", response_model=PredictionResponse)
async def get_country_prediction(country: str):
    """
    Get detailed prediction for a specific country
    """
    try:
        # Get prediction history
        history = db.get_country_prediction_history(country, days=7)

        if len(history) == 0:
            raise HTTPException(status_code=404, detail=f"No predictions for country: {country}")

        # Latest prediction
        latest = history[0]

        current_pred = ConflictPrediction(
            country=latest['country'],
            date=latest['target_date'],
            prediction_date=latest['prediction_date'],
            conflict_probability=latest['conflict_probability'],
            risk_level=latest['risk_level'],
            confidence=latest['confidence']
        )

        # 7-day trend (probabilities)
        trend_7day = [row['conflict_probability'] for row in reversed(history)]

        # Get feature importance (if model is loaded)
        top_features = []
        if model_manager.model is not None:
            feature_importance = model_manager.model.feature_importances_
            top_indices = feature_importance.argsort()[-5:][::-1]

            # Get latest features for this country
            features_data = db.get_latest_features(country, limit=1)
            if features_data:
                features_dict = json.loads(features_data[0]['features_json'])

                for idx in top_indices:
                    feat_name = model_manager.feature_cols[idx]
                    top_features.append(FeatureContribution(
                        feature_name=feat_name,
                        contribution=float(feature_importance[idx]),
                        value=float(features_dict.get(feat_name, 0))
                    ))

        # Get model accuracy
        model_info = db.get_active_model("XGBoost")
        historical_accuracy = None
        if model_info:
            metrics = json.loads(model_info['metrics_json'])
            historical_accuracy = metrics.get('roc_auc')

        return PredictionResponse(
            country=country,
            current_prediction=current_pred,
            trend_7day=trend_7day,
            top_features=top_features,
            historical_accuracy=historical_accuracy
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk-scores", response_model=List[CountryRiskScore])
async def get_risk_scores(limit: int = Query(20, ge=1, le=100)):
    """
    Get risk scores for top N countries
    """
    try:
        predictions_data = db.get_latest_predictions()

        scores = []
        for i, row in enumerate(predictions_data[:limit]):
            # Calculate weekly change (simplified - would need historical data)
            weekly_change = (row['conflict_probability'] - 0.5) * 20  # Placeholder

            if weekly_change > 5:
                trend = "increasing"
            elif weekly_change < -5:
                trend = "decreasing"
            else:
                trend = "stable"

            scores.append(CountryRiskScore(
                country=row['country'],
                risk_score=row['conflict_probability'] * 100,
                risk_level=row['risk_level'],
                weekly_change=weekly_change,
                trend=trend
            ))

        return scores

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-performance", response_model=ModelPerformance)
async def get_model_performance():
    """
    Get model performance metrics
    """
    try:
        model_info = db.get_active_model("XGBoost")

        if not model_info:
            raise HTTPException(status_code=404, detail="No active model found")

        metrics = json.loads(model_info['metrics_json'])

        best_metrics = ModelMetrics(
            model_name=model_info['model_name'],
            accuracy=metrics.get('accuracy', 0),
            precision=metrics.get('precision', 0),
            recall=metrics.get('recall', 0),
            f1_score=metrics.get('f1', 0),
            roc_auc=metrics.get('roc_auc', 0)
        )

        target_met = (
            metrics.get('roc_auc', 0) >= 0.85 and
            metrics.get('f1', 0) >= 0.60
        )

        return ModelPerformance(
            test_period="Based on latest model training",
            best_model=model_info['model_name'],
            best_model_metrics=best_metrics,
            all_models=[best_metrics],
            target_met=target_met
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/retrain")
async def retrain_model():
    """
    Admin endpoint to retrain the model
    In production, this would require authentication
    """
    try:
        version = model_manager.train_and_save()

        return {
            "status": "success",
            "message": "Model retrained successfully",
            "version": version,
            "metrics": model_manager.metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/run-predictions")
async def run_predictions():
    """
    Admin endpoint to generate predictions for all countries
    In production, this would require authentication
    """
    try:
        from scheduler import PredictionScheduler

        scheduler = PredictionScheduler()
        predictions = scheduler.run_daily_predictions()

        return {
            "status": "success",
            "message": f"Generated {len(predictions)} predictions",
            "predictions_count": len(predictions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/multi-horizon")
async def get_multi_horizon_predictions(
    horizon: Optional[int] = Query(None, description="Forecast horizon in days (7, 14, or 30)"),
    min_probability: Optional[float] = Query(None, ge=0, le=1),
    risk_levels: Optional[str] = Query(None)
):
    """
    Get multi-horizon predictions grouped by forecast window

    Query Parameters:
    - horizon: Specific horizon to filter (7, 14, or 30 days). If not specified, returns all.
    - min_probability: Minimum conflict probability (0-1)
    - risk_levels: Comma-separated risk levels (e.g., "HIGH,CRITICAL")
    """
    try:
        # Build query
        query = """
            SELECT country, prediction_date, target_date, horizon_days,
                   conflict_probability, risk_level, confidence
            FROM predictions
            WHERE 1=1
        """
        params = []

        if horizon:
            query += " AND horizon_days = ?"
            params.append(horizon)

        if min_probability is not None:
            query += " AND conflict_probability >= ?"
            params.append(min_probability)

        if risk_levels:
            levels = [level.strip() for level in risk_levels.split(',')]
            placeholders = ','.join(['?'] * len(levels))
            query += f" AND risk_level IN ({placeholders})"
            params.extend(levels)

        query += " ORDER BY horizon_days, conflict_probability DESC"

        cursor = db.conn.execute(query, params)
        rows = cursor.fetchall()

        # Group by horizon
        predictions_by_horizon = {}
        for row in rows:
            h = row[3]  # horizon_days
            if h not in predictions_by_horizon:
                predictions_by_horizon[h] = []

            predictions_by_horizon[h].append({
                "country": row[0],
                "prediction_date": row[1],
                "target_date": row[2],
                "horizon_days": row[3],
                "conflict_probability": round(row[4], 4),
                "risk_level": row[5],
                "confidence": round(row[6], 4)
            })

        return {
            "total_predictions": len(rows),
            "horizons": sorted(predictions_by_horizon.keys()),
            "predictions_by_horizon": predictions_by_horizon
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("[STARTING] GDELT Conflict Predictor API (Database Version)...")
    print("[INFO] API Documentation: http://localhost:8000/docs")
    print("[INFO] Health Check: http://localhost:8000")
    print("\n[INFO] To populate database, run:")
    print("       python data_fetcher.py")
    print("       python model_manager.py")
    print("   python scheduler.py")
    uvicorn.run(app, host="0.0.0.0", port=8000)
