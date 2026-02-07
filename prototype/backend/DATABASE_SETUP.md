# Database Setup Guide

Complete guide for setting up the production database system for GDELT Conflict Predictor.

---

## Overview

The new database system provides:
- **SQLite database** for storing features and predictions
- **Model persistence** (save/load trained models)
- **Daily data updates** via scheduler
- **Fast API responses** (no retraining on startup)

---

## File Structure

```
backend/
├── database.py          # Database schema and operations
├── data_fetcher.py      # Fetches and processes GDELT data
├── model_manager.py     # Trains and manages ML models
├── scheduler.py         # Daily prediction scheduler
├── main_db.py           # FastAPI app (database version)
└── saved_models/        # Stored model files (created automatically)
```

---

## Initial Setup

### Step 1: Populate Database from Parquet

If you have existing parquet data:

```powershell
cd prototype\backend
.\venv\Scripts\Activate.ps1

# Import data to database
python data_fetcher.py
```

**Output:**
```
📂 Loading data from .../country_day_features.parquet
✅ Loaded 123,456 rows
📥 Importing 123,456 rows to database...
   Imported 1,000 / 123,456 rows...
   ...
✅ Imported 123,456 rows to database

📊 Database Stats:
   Total features: 123,456
   Countries: 50
   Date range: 2023-01-01 to 2024-12-11
```

###Step 2: Train and Save Model

```powershell
python model_manager.py
```

**Output:**
```
📊 Preparing training data from database...
✅ Data prepared:
   Features: 65
   Train samples: 89,234
   Test samples: 34,222
🔄 Training XGBoost model...
✅ Model trained!
   ROC-AUC: 83.57%
   F1 Score: 63.79%
✅ Model saved:
   Model: saved_models/XGBoost_20241212_123045.pkl
   Scaler: saved_models/scaler_20241212_123045.pkl
```

### Step 3: Generate Initial Predictions

```powershell
python scheduler.py
```

**Output:**
```
🔮 Running predictions for 2024-12-13
📊 Generating predictions for 50 countries...
✅ Generated 50 predictions

📊 Prediction Summary:
   CRITICAL: 3 countries
   HIGH: 12 countries
   MEDIUM: 20 countries
   LOW: 15 countries
```

### Step 4: Start API Server

```powershell
python main_db.py
```

**Output:**
```
🚀 Starting GDELT Conflict Predictor API (Database Version)...
✅ Model loaded from disk
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Database Schema

### Tables

**1. country_features**
```sql
country         TEXT      -- Country name
date            DATE      -- Date of features
num_events      INTEGER   -- Number of events
avg_goldstein   REAL      -- Average Goldstein scale
quad_class_1-4  INTEGER   -- QuadClass distributions
is_high_conflict INTEGER  -- Conflict indicator
features_json   TEXT      -- All features as JSON
created_at      TIMESTAMP -- When inserted
```

**2. predictions**
```sql
country              TEXT      -- Country name
prediction_date      DATE      -- When prediction was made
target_date          DATE      -- Date being predicted
conflict_probability REAL      -- Probability 0-1
risk_level           TEXT      -- LOW/MEDIUM/HIGH/CRITICAL
confidence           REAL      -- Model confidence
model_version        TEXT      -- Model version used
created_at           TIMESTAMP
```

**3. models**
```sql
model_name      TEXT      -- e.g., "XGBoost"
model_version   TEXT      -- Version timestamp
model_path      TEXT      -- Path to .pkl file
metrics_json    TEXT      -- Performance metrics
trained_at      TIMESTAMP
is_active       BOOLEAN   -- Only one active per model_name
```

**4. gdelt_events** (optional - for caching)
```sql
event_id        TEXT UNIQUE
date            DATE
country         TEXT
event_code      TEXT
goldstein_scale REAL
quad_class      INTEGER
raw_data_json   TEXT
```

---

## Daily Workflow

### Automated Daily Updates

Set up a scheduled task to run daily:

**Windows Task Scheduler:**
```powershell
# Create a .bat file: daily_update.bat
cd C:\Users\Emman\Documents\AI_dev\GDELT_ConflictPredictor\prototype\backend
call venv\Scripts\activate
python scheduler.py
```

Schedule to run at 2 AM daily.

**Or run manually:**
```powershell
python scheduler.py
```

This will:
1. Fetch latest GDELT data (or generate sample data)
2. Update `country_features` table
3. Generate predictions for all countries
4. Store in `predictions` table

---

## API Endpoints (Database Version)

### GET /
Health check with database stats

### GET /api/predictions/latest
Get latest predictions for all countries

**Query Parameters:**
- `min_probability`: Filter by minimum probability (0-1)
- `risk_levels`: Filter by risk level (comma-separated)
- `countries`: Filter by country names (comma-separated)

**Example:**
```
/api/predictions/latest?risk_levels=HIGH,CRITICAL&min_probability=0.6
```

### GET /api/predictions/country/{country}
Detailed prediction for a specific country
- 7-day trend
- Top 5 contributing features
- Historical accuracy

### GET /api/risk-scores
Top N countries by risk score

**Query Parameters:**
- `limit`: Number of countries (default: 20)

### GET /api/model-performance
Model metrics (ROC-AUC, F1, etc.)

### GET /api/admin/retrain
Retrain the model (admin only)

### GET /api/admin/run-predictions
Generate predictions for all countries (admin only)

---

## Database Operations

### View Data

```python
from database import get_db

db = get_db()

# Get stats
stats = db.get_data_summary()
print(stats)

# Get latest predictions
predictions = db.get_latest_predictions()
for pred in predictions[:5]:
    print(f"{pred['country']}: {pred['conflict_probability']:.2%}")

# Get country history
history = db.get_country_prediction_history('Myanmar', days=7)
for h in history:
    print(f"{h['prediction_date']}: {h['conflict_probability']:.2%}")
```

### Add New Data

```python
from database import get_db
from datetime import date

db = get_db()

# Add features
features = {
    'NumEvents_sum': 245,
    'AvgGoldstein_sum': -2.3,
    'QuadClass_1_sum': 30,
    'QuadClass_2_sum': 40,
    'QuadClass_3_sum': 60,
    'QuadClass_4_sum': 115,
    'IsHighConflict_sum': 42
}

db.insert_country_features(
    country='Myanmar',
    date=date.today(),
    features=features
)
```

### Retrain Model

```python
from model_manager import ModelManager

manager = ModelManager()
version = manager.train_and_save()
print(f"New model version: {version}")
```

---

## Advantages of Database System

✅ **Fast API startup** - No retraining every time
✅ **Historical tracking** - Store all predictions
✅ **Model versioning** - Track multiple model versions
✅ **Easy updates** - Add new data without full reload
✅ **Scalable** - Can switch to PostgreSQL later
✅ **Audit trail** - Know when each prediction was made
✅ **Feature reuse** - Compute features once, use many times

---

## Migration from Old System

**Old (predictor.py):**
- Trains model on startup
- Uses sample data if parquet missing
- Slow startup time
- No persistence

**New (main_db.py):**
- Loads pre-trained model
- Uses database for all data
- Fast startup (<1 second)
- Full persistence

**To switch:**
1. Run `python data_fetcher.py` to populate database
2. Run `python model_manager.py` to train and save model
3. Use `python main_db.py` instead of `python main.py`

---

## Troubleshooting

### "No data in database"

**Solution:**
```powershell
python data_fetcher.py
```

### "No active model found"

**Solution:**
```powershell
python model_manager.py
```

### "Database is locked"

**Solution:** Close other connections or use:
```python
# In database.py, change to:
self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
```

### Database file location

Default: `prototype/backend/gdelt_predictor.db`

To change, edit `database.py`:
```python
def __init__(self, db_path: str = "path/to/your/database.db"):
```

---

## Production Considerations

### Switch to PostgreSQL

Replace SQLite with PostgreSQL for production:

1. Install psycopg2: `pip install psycopg2-binary`
2. Update `database.py` to use PostgreSQL connector
3. Update connection string
4. Same schema works with minor modifications

### Security

Add authentication to admin endpoints:

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/api/admin/retrain")
async def retrain_model(credentials = Depends(security)):
    # Verify token
    ...
```

### Scaling

- **Read replicas** for API queries
- **Write master** for daily updates
- **Connection pooling** for high traffic
- **Caching layer** (Redis) for frequent queries

---

**Your database system is ready! Run the setup steps above to get started.** 🚀
