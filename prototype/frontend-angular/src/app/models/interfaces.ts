export interface ConflictPrediction {
  country: string;
  date?: string;
  prediction_date: string;
  target_date: string;
  conflict_probability: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  confidence: number;
  horizon_days?: number;
  model_version?: string;
}

export interface FeatureContribution {
  feature_name: string;
  contribution: number;
  value: number;
}

export interface PredictionResponse {
  country: string;
  current_prediction: ConflictPrediction;
  predictions: ConflictPrediction[];
  trend_7day: number[];
  top_features: FeatureContribution[];
  historical_accuracy: number | null;
  recent_news?: NewsStory[];
  data_source: string;
}

export interface CountryRiskScore {
  country: string;
  risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  weekly_change: number;
  trend: 'increasing' | 'decreasing' | 'stable';
}

export interface ModelMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
}

export interface ModelPerformance {
  test_period: string;
  best_model: string;
  best_model_metrics: ModelMetrics;
  all_models: ModelMetrics[];
  target_met: boolean;
}

export interface MultiHorizonResponse {
  total_predictions: number;
  horizons: number[];
  predictions_by_horizon: {
    [key: string]: ConflictPrediction[];
  };
}

export interface NewsStory {
  id: number;
  title: string;
  summary: string;
  country: string;
  sentiment_score: number;
  goldstein_scale?: number;
  date: string;
  source: string;
  category: string;
  url?: string;
  num_mentions?: number;
}

export interface DataStatus {
  predictions: {
    last_prediction_date: string;
    target_date_range: {
      min: string;
      max: string;
    };
    total_predictions: number;
    countries_monitored: number;
    horizons_available: number[];
    status: string;
    days_since_update: number | null;
  };
  model: {
    name: string;
    version: string;
    features: number;
    accuracy: number;
  };
  system: {
    database: string;
    framework: string;
    last_check: string;
  };
}

export interface DashboardOverview {
  summary: {
    total_countries_monitored: number;
    high_risk_countries: number;
    average_risk_level: number;
    last_updated: string;
  };
  predictions: ConflictPrediction[];
  positive_news: NewsStory[];
  negative_news: NewsStory[];
  trends: {
    risk_direction: string;
    top_risk_factors: string[];
    data_source: string;
  };
}
