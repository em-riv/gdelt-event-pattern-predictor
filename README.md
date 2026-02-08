# GDELT Event Pattern Predictor

ML system that predicts next-day GDELT event patterns across 229 countries using deep learning. Built to learn full-stack ML deployment: data pipeline, model training, REST API, and web frontend.

## What This Project Does

Processes 135M+ events from GDELT (Global Database of Events, Language and Tone) and predicts tomorrow's event intensity for each country. Backend serves predictions via REST API. Frontend displays trends on a dashboard.

**Important:** This predicts what gets reported in news, not what actually happens. GDELT is media coverage with all its biases. I discovered this the hard way when validating against real conflict data (ACLED) - the gap was huge.

## What's Here

- 98M events aggregated into 220K country-day records
- 53 features per observation (event counts, sentiment, conflict types, temporal lags)
- 4 models trained: GRU (82.57% ROC-AUC), XGBoost, LSTM, LightGBM
- FastAPI backend serving predictions from SQLite
- Angular frontend with Plotly visualizations
- 5 Jupyter notebooks showing feature engineering, baseline models, optimization

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- 2GB RAM minimum

### Backend Setup (5 min)

```bash
cd prototype/backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main_api.py
```

Backend runs at http://localhost:8000
Swagger docs at http://localhost:8000/docs

### Frontend Setup (5 min)

```bash
cd prototype/frontend-angular
npm install
npm start
```

Dashboard runs at http://localhost:4200

### Test Integration

1. Navigate to http://localhost:4200
2. Select a country from dropdown
3. View predictions and Goldstein score trends
4. Check backend health at http://localhost:8000/docs

## Architecture Overview

```
GDELT Events (98M+)
       |
       v
Data Pipeline (parsing, validation)
       |
       v
Feature Engineering (country-day aggregation)
       |
       v
SQLite Database (220K+ records)
       |
       v
Model Training (temporal split, SMOTE)
       |
       v
Saved Models (.pth, .pkl files)
       |
       v
FastAPI Backend (REST endpoints)
       |
       v
Angular Frontend (dashboard visualization)
```

## Data Pipeline Details

### Events -> Country-Day Aggregation

1. Parse 98M GDELT events spanning 2015-2024
2. Aggregate by country and date:
   - Event counts and article volume
   - Goldstein scale statistics (conflict intensity)
   - QuadClass distribution (conflict types)
   - Tone/sentiment metrics
3. Result: 220K country-day records

### Feature Engineering

53 features per country-day (7-day temporal window):
- Event aggregates (4): counts, articles, mentions, sources
- Goldstein statistics (7): mean, std, quartiles
- QuadClass distribution (4): verbal/material cooperation/conflict %
- Tone metrics (4): mean, std, min, max
- Temporal features (20+): rolling averages, lags, day-of-week

### Model Training

- Train/test split: by date (2023 train, 2024 test)
- Class imbalance: addressed with SMOTE
- Cross-validation: stratified 5-fold
- Hyperparameter tuning: grid search

Best model: Bidirectional GRU
- ROC-AUC: 82.57%
- F1 Score: 63.79% (precision-recall balance)
- Recall: 86.17% (catches high-activity days)

## Important Limitations

This system has important constraints you must understand:

### Circular Validation
- Model validated against GDELT patterns, not ground truth conflicts
- Performance metric (82.57% ROC-AUC) means "we predict what gets reported"
- NOT validated against real conflict data (ACLED, UCDP)
- Equivalent to predicting stock sentiment from headlines, not stock prices

### Media Bias
- GDELT = news coverage, not reality
- Reporting concentrated on certain countries/regions
- Missing unreported conflicts
- Over-representing conflicts with international media attention

### Forecast Horizon
- 1-day prediction window only
- Too short for operational decision-making
- Real conflict forecasting needs weeks/months lead time

### Scope
- Country-level only (no sub-national analysis)
- No geospatial weighting
- No actor-network analysis
- No real-time streaming (batch predictions only)

## API Endpoints

### GET /predictions/{country}
Get latest prediction for specific country

Response:
```json
{
  "country": "Syria",
  "date": "2024-12-15",
  "prediction_score": 0.78,
  "confidence": 0.82,
  "features": {
    "avg_goldstein": -2.1,
    "event_count": 47,
    "tone_mean": -15.3
  }
}
```

### GET /predictions/top
Get top 10 highest-risk countries today

### GET /health
API health check

Full docs: http://localhost:8000/docs (when running)

## Project Structure

```
prototype/
 backend/
    main_api.py              # FastAPI entry point
    database.py              # SQLite layer
    model_manager.py         # ML model loading/inference
    data_fetcher.py          # GDELT data ingestion
    models.py                # Pydantic schemas
    requirements.txt         # Python dependencies
    gdelt_predictor.db       # SQLite database
    saved_models/            # Trained model artifacts
        gru_model.pth
        xgboost_model.pkl
        scaler.pkl
        feature_names.json

 frontend-angular/
    src/
       app/
          components/      # Angular components
          services/        # API services
          models/          # TypeScript interfaces
       index.html
       main.ts
    angular.json
    package.json
    tsconfig.json

 README.md
```

## Technology Stack

Backend:
- FastAPI (REST API framework)
- PyTorch (deep learning models)
- scikit-learn (preprocessing, metrics)
- SQLite (data persistence)
- Pandas/NumPy (data processing)

Frontend:
- Angular 17 (framework)
- Plotly (visualizations)
- RxJS (reactive programming)
- TypeScript

## Development Notes

### Adding New Predictions

1. Update `data_fetcher.py` to ingest latest GDELT data
2. Run feature engineering pipeline
3. Load trained model via `model_manager.py`
4. Store predictions in SQLite
5. API automatically serves updated predictions

### Retraining Models

Training scripts in `/prototype/backend/`:
- `train_gru_forecasting.py` - Bidirectional GRU training
- Update `saved_models/` when complete
- No model management beyond file replacement (consider MLflow for production)

### Testing

```bash
cd prototype/backend
python -m pytest tests/
```

## Known Issues

- Frontend map visualization requires internet (Plotly CDN)
- SQLite not suitable for concurrent writes at scale
- Model loading on first request causes API delay
- No input validation on country codes
- Simple CORS (all origins allowed in dev)

## Production Deployment Considerations

Before deploying to production:

1. Database: Migrate from SQLite to PostgreSQL
2. Authentication: Add API key/JWT authentication
3. Rate limiting: Implement request throttling
4. Monitoring: Add logging and error tracking
5. Health checks: Implement liveness/readiness probes
6. Load testing: Validate performance under real load
7. Security: Run vulnerability scan, add input validation
8. Backups: Implement automated database backups

## Performance Metrics

Dataset: 220K country-day records | 229 countries | 7-year history

Model Performance (test set 2024):
- ROC-AUC: 0.8257
- F1 Score: 0.6379
- Precision: 0.6234
- Recall: 0.6617
- Accuracy: 0.7847

Baseline Comparison:
- Random classifier: 0.5000 ROC-AUC
- Logistic Regression: 0.7621 ROC-AUC
- XGBoost: 0.8140 ROC-AUC
- LightGBM: 0.8089 ROC-AUC
- LSTM: 0.8201 ROC-AUC
- Bidirectional GRU: 0.8257 ROC-AUC (best)

## Papers & References

Feature engineering based on academic literature:
- Goldstein scale: Goldstein, J.S. (1992). "A conflict-cooperation scale"
- CAMEO coding: Schrodt, P.A. (2012). "Event data and forecasting"
- Temporal features: Best practices from time-series forecasting

GDELT Dataset:
- https://www.gdeltproject.org/

Conflict Prediction Research:
- ACLED: https://acleddata.com/
- UCDP: https://ucdp.uu.se/

## License

MIT License - See LICENSE file

## Author

Created as a portfolio project demonstrating ML engineering practices.

## Questions or Issues?

See individual README files in backend/ and frontend-angular/ subdirectories for component-specific documentation.
