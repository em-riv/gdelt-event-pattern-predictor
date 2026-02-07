# Development Guide

How to work on this ML system locally.

## Project Layout

```
prototype/
 backend/                    # FastAPI REST API
    main_api.py            # Main entry point
    database.py            # SQLite data layer
    model_manager.py       # ML model loading and inference
    data_fetcher.py        # GDELT data ingestion
    models.py              # Pydantic request/response schemas
    requirements.txt       # Python dependencies
    README.md              # Backend documentation
    saved_models/          # Trained model artifacts (.pth, .pkl)
    scripts/               # Utility and training scripts (30 files)
       README.md         # Scripts documentation
    archived_api_versions/ # Previous API implementations
    .env.example           # Configuration template

 frontend-angular/          # Angular 17 dashboard
    src/app/              # Angular components and services
    package.json          # Node dependencies
    angular.json          # Angular build config
    README.md             # Frontend documentation
    tsconfig.json         # TypeScript config

 start_backend.bat         # Windows backend launcher

notebooks/                     # Analysis and development notebooks
 GDELT_EDA_Traditional.ipynb
 02_feature_engineering.ipynb
 03_baseline_models.ipynb
 04_advanced_models.ipynb
 05_final_optimization.ipynb

reports/                       # Analysis outputs
 data_quality_report.json
 visual_eda/              # Exploratory Data Analysis HTML/PNG files

README.md                      # Main project documentation
LICENSE                        # MIT License
TECHNICAL_ARCHITECTURE.md      # Deep technical reference
.gitignore                     # Git ignore rules
```

## Quick Start

### Backend

```bash
cd prototype/backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main_api.py
```

Runs at http://localhost:8000

### Frontend

```bash
cd prototype/frontend-angular
npm install
npm start
```

Runs at http://localhost:4200

## Core Modules

### Backend

**main_api.py**
- FastAPI application entry point
- RESTful endpoints for predictions
- CORS configuration
- Model serving

**database.py**
- SQLite connection management
- Query execution and caching
- Data persistence layer

**model_manager.py**
- Loading pre-trained models (.pth, .pkl)
- Feature scaling
- Inference execution

**data_fetcher.py**
- GDELT data ingestion
- Event processing
- Feature computation

**models.py**
- Pydantic validation schemas
- Request/response models
- Type hints

### Frontend

**app.component.ts** - Root component and app shell
**app.routes.ts** - Route definitions
**services/** - HTTP API communication
**components/** - Feature components (dashboard, country detail, etc.)
**models/** - TypeScript interfaces

## Development Workflow

### Adding an API Endpoint

1. Define schema in `backend/models.py`
2. Implement handler in `backend/main_api.py`
3. Test with `http://localhost:8000/docs`
4. Update frontend service in `frontend-angular/src/app/services/`

### Adding a Frontend Component

1. Generate: `ng generate component component-name`
2. Add route to `app.routes.ts`
3. Implement logic in `.component.ts`
4. Call API via injected service

### Retraining Models

In `backend/scripts/`:

```bash
# GRU Model
python train_gru_forecasting.py

# Update XGBoost
python update_xgboost_model.py
```

Models save to `saved_models/`

## Data Pipeline

1. **Ingestion** - `data_fetcher.py` fetches GDELT events
2. **Feature Engineering** - Aggregation scripts compute country-day features
3. **Model Training** - Training scripts in `scripts/` generate models
4. **Database** - Features and predictions stored in SQLite
5. **API** - `main_api.py` serves from database
6. **Frontend** - Dashboard displays predictions

## Testing

### Backend
```bash
cd prototype/backend
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

### Frontend
```bash
cd prototype/frontend-angular
ng test
```

## Deployment

### Local Production Build

**Backend (uWSGI)**
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main_api:app
```

**Frontend**
```bash
ng build --configuration production
# Serve dist/ with nginx
```

### Production Considerations

- Migrate from SQLite to PostgreSQL
- Add API authentication (JWT)
- Implement rate limiting
- Set up monitoring and logging
- Configure health checks
- Implement graceful shutdown
- Add request validation
- Security audit

## Configuration

### Backend

Copy `.env.example` to `.env` and update:

```
DATABASE_URL=sqlite:///gdelt_predictor.db
API_PORT=8000
FRONTEND_URL=http://localhost:4200
```

### Frontend

Environment files in `src/environments/`:
- `environment.ts` (development)
- `environment.prod.ts` (production)

## Architecture

See [TECHNICAL_ARCHITECTURE.md](../TECHNICAL_ARCHITECTURE.md) for:
- System design patterns
- Database schemas
- API specifications
- ML pipeline details
- Deployment strategies

## Important Files by Purpose

### Configuration
- `backend/requirements.txt` - Python dependencies
- `backend/.env.example` - Backend config template
- `frontend-angular/package.json` - Node dependencies
- `frontend-angular/angular.json` - Angular build config

### Core Logic
- `backend/main_api.py` - API server
- `backend/model_manager.py` - ML inference
- `backend/database.py` - Data persistence
- `frontend-angular/src/app/services/api.service.ts` - API client

### Models
- `backend/saved_models/gru_model.pth` - GRU weights
- `backend/saved_models/xgboost_model.pkl` - XGBoost classifier
- `backend/saved_models/scaler.pkl` - Feature scaler

## Scripts (in backend/scripts/)

### Data Operations
- `generate_predictions_gru.py` - Generate GRU predictions
- `process_raw_gdelt.py` - Process raw GDELT events
- `populate_db_simple.py` - Load features to database

### Model Training
- `train_gru_forecasting.py` - Train GRU model
- `update_xgboost_model.py` - Retrain XGBoost
- `generate_multi_horizon_predictions.py` - Multi-horizon forecasting

### Analysis & Debugging
- `test_api.py` - API integration tests
- `check_db.py` - Database validation
- `debug_gru_predictions.py` - GRU troubleshooting

See `scripts/README.md` for complete documentation.

## Common Tasks

### Update Backend Port
Edit `backend/.env` or `main_api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Update Frontend API URL
Edit `frontend-angular/src/app/app.config.ts`:
```typescript
export const API_BASE_URL = 'http://your-api.com:8000';
```

### Add New Dependencies

**Python:**
```bash
pip install new-package
pip freeze > requirements.txt
```

**Node:**
```bash
npm install new-package
```

### Database Backup

SQLite database is at `backend/gdelt_predictor.db`

```bash
cp gdelt_predictor.db gdelt_predictor.db.backup
```

## Troubleshooting

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux
lsof -i :8000
kill -9 <PID>
```

### Models Not Loading
- Check `backend/saved_models/` directory exists
- Verify `.pth` and `.pkl` files present
- Check file permissions

### Frontend CORS Errors
Backend CORS configured in `main_api.py` for `localhost:4200`
Update if running on different port.

### Database Locked
SQLite can't handle concurrent writes well.
- Close other connections to database
- Check for stuck processes
- Consider PostgreSQL for production

## Performance Profiling

**Backend:**
```bash
pip install py-spy
py-spy record -o profile.svg -- python main_api.py
```

**Frontend:**
Chrome DevTools Performance tab

## VS Code Extensions (Recommended)

- Pylance - Python language server
- Prettier - Code formatter
- Angular Language Service - Angular template support
- Thunder Client - API testing
- Git Graph - Git visualization

## References

- [README.md](./README.md) - Project overview
- [TECHNICAL_ARCHITECTURE.md](./TECHNICAL_ARCHITECTURE.md) - System design
- [backend/README.md](./prototype/backend/README.md) - Backend API
- [frontend-angular/README.md](./prototype/frontend-angular/README.md) - Frontend
- [backend/scripts/README.md](./prototype/backend/scripts/README.md) - Scripts documentation
