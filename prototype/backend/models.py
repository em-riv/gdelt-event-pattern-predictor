"""
Data models for GDELT Conflict Predictor API
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date as date_type


class ConflictPrediction(BaseModel):
    """Prediction for a single country"""
    country: str
    date: date_type
    prediction_date: date_type = Field(description="Date this prediction was made")
    conflict_probability: float = Field(ge=0, le=1, description="Probability of high conflict (0-1)")
    risk_level: str = Field(description="LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(ge=0, le=1, description="Model confidence (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "country": "Myanmar",
                "date": "2024-12-12",
                "prediction_date": "2024-12-11",
                "conflict_probability": 0.87,
                "risk_level": "HIGH",
                "confidence": 0.92
            }
        }


class FeatureContribution(BaseModel):
    """Top features contributing to prediction"""
    feature_name: str
    contribution: float
    value: float


class PredictionResponse(BaseModel):
    """Detailed prediction response for a single country"""
    country: str
    current_prediction: ConflictPrediction
    trend_7day: List[float] = Field(description="7-day prediction history")
    top_features: List[FeatureContribution]
    historical_accuracy: Optional[float] = Field(None, description="Historical accuracy for this country")

    class Config:
        json_schema_extra = {
            "example": {
                "country": "Myanmar",
                "current_prediction": {
                    "country": "Myanmar",
                    "date": "2024-12-12",
                    "prediction_date": "2024-12-11",
                    "conflict_probability": 0.87,
                    "risk_level": "HIGH",
                    "confidence": 0.92
                },
                "trend_7day": [0.65, 0.71, 0.75, 0.79, 0.83, 0.85, 0.87],
                "top_features": [
                    {"feature_name": "AvgGoldstein_sum", "contribution": 0.23, "value": -5.8},
                    {"feature_name": "NumEvents_sum", "contribution": 0.18, "value": 245},
                    {"feature_name": "QuadClass_4_sum", "contribution": 0.15, "value": 67}
                ],
                "historical_accuracy": 0.85
            }
        }


class CountryRiskScore(BaseModel):
    """Country risk ranking"""
    country: str
    risk_score: float = Field(ge=0, le=100, description="Risk score 0-100")
    risk_level: str
    weekly_change: float = Field(description="Change in risk score from last week")
    trend: str = Field(description="increasing, decreasing, stable")

    class Config:
        json_schema_extra = {
            "example": {
                "country": "Myanmar",
                "risk_score": 87.5,
                "risk_level": "HIGH",
                "weekly_change": 12.3,
                "trend": "increasing"
            }
        }


class ModelMetrics(BaseModel):
    """Performance metrics for a single model"""
    model_config = {"protected_namespaces": ()}

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


class ModelPerformance(BaseModel):
    """Overall model performance"""
    test_period: str = Field(description="Date range of test set")
    best_model: str
    best_model_metrics: ModelMetrics
    all_models: List[ModelMetrics]
    target_met: bool = Field(description="Whether ROC-AUC >= 85% and F1 >= 60%")

    class Config:
        json_schema_extra = {
            "example": {
                "test_period": "2024-01-01 to 2024-12-11",
                "best_model": "XGBoost",
                "best_model_metrics": {
                    "model_name": "XGBoost",
                    "accuracy": 0.89,
                    "precision": 0.74,
                    "recall": 0.82,
                    "f1_score": 0.78,
                    "roc_auc": 0.91
                },
                "all_models": [],
                "target_met": True
            }
        }
