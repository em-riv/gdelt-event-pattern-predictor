# Backend - FastAPI API Server

REST API serving GDELT predictions from a trained GRU model and SQLite database.

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
python main_api.py
```

Server runs at http://localhost:8000

## API Documentation

Once running, visit http://localhost:8000/docs for interactive Swagger UI.

## Main Entry Point

Use **main_api.py** to start the server.

Other API files are experimental/deprecated (archived in archived_api_versions/):
- enhanced_gru_api.py - GRU-specific (archived)
- production_gru_api.py - Early GRU version (archived)
- simple_gru_api.py - Testing only (archived)
- stable_api.py - Testing only (archived)
- flask_api.py - Old Flask version (archived)

## Configuration

Set environment variables in `.env` file (see `.env.example`):

```
DATABASE_URL=sqlite:///gdelt_predictor.db
API_PORT=8000
FRONTEND_URL=http://localhost:4200
```

## Core Modules

### main_api.py
FastAPI application with all endpoints. Loads models via ModelManager.

Endpoints:
- GET /predictions/{country} - Latest prediction for country
- GET /predictions/top - Top 10 highest-risk countries
- GET /health - API health check
- POST /predictions/batch - Predictions for multiple countries
- GET /models - Available model list

### database.py
SQLite database layer. Manages connections and queries.

Key functions:
- get_predictions(country, limit)
- get_all_predictions(date)
- store_prediction(data)
- get_historical_features(country, days)

### model_manager.py
ML model loading and inference. Handles PyTorch and scikit-learn models.

Supports:
- Bidirectional GRU (.pth)
- XGBoost (.pkl)
- Feature scaling (scaler.pkl)
- Feature name mapping (feature_names.json)

### data_fetcher.py
GDELT data ingestion and processing.

Functions:
- fetch_gdelt_data(date_range)
- process_events(dataframe)
- aggregate_by_country_day()

### models.py
Pydantic schemas for request/response validation.

Key models:
- ConflictPrediction
- CountryRiskScore
- ModelPerformance
- PredictionResponse

## Database Schema

### country_features
```
id (INTEGER PRIMARY KEY)
country (TEXT)
date (DATE)
num_events (INTEGER)
avg_goldstein (FLOAT)
avg_tone (FLOAT)
quad_class_distribution (TEXT JSON)
... (48 more features)
```

### predictions
```
id (INTEGER PRIMARY KEY)
country (TEXT)
date (DATE)
prediction_score (FLOAT)
confidence (FLOAT)
model_version (TEXT)
created_at (DATETIME)
```

## Adding New Predictions

1. New GDELT data is collected daily
2. Features are computed for all countries
3. Trained models generate predictions
4. API serves from predictions table

Manual workflow:
```bash
python generate_predictions_gru.py  # Generate predictions
# or
python main_db.py  # API auto-loads on startup
```

## Model Files

Location: `saved_models/`

- gru_model.pth - Bidirectional GRU weights (PyTorch)
- xgboost_model.pkl - XGBoost classifier (scikit-learn)
- scaler.pkl - Feature normalizer
- feature_names.json - Feature list for model input
- metadata_gru_forecast.json - Model metadata

## Testing

```bash
# Test API endpoint
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"

# Test specific country
curl "http://localhost:8000/predictions/Syria"
```

## Troubleshooting

### Port 8000 already in use
```bash
netstat -tuln | grep 8000
# or change API_PORT in .env
```

### Database locked
Multiple model training jobs? Ensure only one main_api.py instance running.

### Models not loading
Check that saved_models/ directory exists with all .pkl/.pth files present.

### CORS errors in frontend
CORS is configured for localhost:4200 in main_api.py. Update if frontend runs on different port.

## Production Deployment

### Before deploying to production:

1. Switch database to PostgreSQL (SQLite not suitable for concurrent writes)
2. Add authentication (JWT tokens or API keys)
3. Implement rate limiting
4. Add request logging and monitoring
5. Set up automated database backups
6. Use Gunicorn/uWSGI instead of uvicorn dev server
7. Add health check endpoints for load balancer
8. Implement graceful shutdown handling

### Example uWSGI command:
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main_api:app --bind 0.0.0.0:8000
```

## Performance Notes

- Model inference: ~50ms per country
- Database query: ~10ms
- Full batch (229 countries): ~12 seconds
- API response time: ~100-150ms (including overhead)

## Development

To modify endpoints, edit main_api.py:

```python
@app.get("/predictions/{country}")
async def get_prediction(country: str):
    # Your logic here
    return prediction_response
```

All responses validated against Pydantic models in models.py.

## Monitoring

Check API logs for errors:
```bash
tail -f gdelt_api.log
```

## See Also

- [Backend README](./README.md) for quick start
- Main project [README](../README.md) for architecture overview
- [Frontend README](../frontend-angular/README.md) for dashboard details
