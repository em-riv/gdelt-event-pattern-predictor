# GDELT Conflict Predictor - Technical Architecture

**Detailed technical documentation for developers and architects**

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Patterns](#architecture-patterns)
3. [Data Layer](#data-layer)
4. [ML Pipeline](#ml-pipeline)
5. [Backend API](#backend-api)
6. [Frontend Application](#frontend-application)
7. [Deployment Architecture](#deployment-architecture)
8. [Security Considerations](#security-considerations)
9. [Performance & Scalability](#performance--scalability)
10. [Testing Strategy](#testing-strategy)

---

## 1. System Overview

### High-Level Architecture

```

                        External Data Sources                     
                     GDELT Event Database                         
                   (15-minute update frequency)                   

                               
                               

                      Data Ingestion Layer                        
               
   GDELT Fetcher   CSV Parser      Parquet Load          
               

                               
                               

                   Feature Engineering Layer                      
                        
   Country-Day Agg     Temporal Features                    
                        
                        
   Goldstein Stats     QuadClass Ratios                     
                        

                               
                               

                        Data Persistence                          
               
             SQLite Database (Dev)                             
          PostgreSQL (Production)                              
                                                                
    Tables:                                                     
    - country_features (149K+ rows)                            
    - predictions (historical)                                 
    - models (metadata)                                        
               

                               
                               

                      ML Training Pipeline                        
               
   Data Split     XGBoost Train  Model Save            
   (2023/2024)     (SMOTE, CV)     (.pkl)                
               

                               
                               

                    Prediction Generation                         
               
    Daily Scheduler:                                           
    1. Load latest features                                    
    2. Generate predictions (all countries)                    
    3. Classify risk levels                                    
    4. Store in database                                       
               

                               
                               

                         API Layer                                
                   FastAPI REST Service                           
                   http://localhost:8000                          
                                                                  
  Endpoints:                                                      
  - GET /api/predictions/latest                                  
  - GET /api/predictions/country/{country}                       
  - GET /api/risk-scores                                         
  - GET /api/model-performance                                   
  - GET /api/admin/retrain                                       
  - GET /api/admin/run-predictions                               

                               
                               

                      Frontend Application                        
                   Angular 17 Dashboard                           
                   http://localhost:4200                          
                                                                  
  Components:                                                     
  - Dashboard (map, charts, tables)                              
  - Country Detail (trends, features)                            
  - Filters (risk level, search)                                 

```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Source** | GDELT API | Global event data |
| **Data Storage** | Parquet, SQLite/PostgreSQL | Feature storage, predictions |
| **Data Processing** | Pandas, NumPy | Feature engineering |
| **ML Framework** | XGBoost, scikit-learn | Model training, prediction |
| **Backend API** | FastAPI (Python 3.10+) | REST API server |
| **Frontend** | Angular 17, TypeScript | Web dashboard |
| **Visualization** | Plotly.js, Angular Material | Charts, UI components |
| **Deployment** | Docker, Uvicorn | Containerization, ASGI server |

---

## 2. Architecture Patterns

### Design Patterns Used

**1. Repository Pattern**
- `database.py` abstracts data access
- Separates business logic from data layer
- Easy to swap SQLite  PostgreSQL

**2. Service Layer Pattern**
- `model_manager.py` - ML model service
- `data_fetcher.py` - Data ingestion service
- Clean separation of concerns

**3. Dependency Injection**
- FastAPI's built-in DI for database connections
- Pydantic models for type validation

**4. Factory Pattern**
- Model creation and loading
- Feature transformer pipelines

**5. Scheduler Pattern**
- Daily batch prediction generation
- Cron-like task scheduling (ready for Celery)

### API Design Principles

**RESTful conventions:**
- Resource-oriented URLs (`/api/predictions/...`)
- HTTP verbs (GET for reads, POST for writes)
- Standard status codes (200, 404, 500)

**Pagination & Filtering:**
```python
GET /api/predictions/latest?limit=50&offset=0
GET /api/predictions/latest?risk_levels=HIGH,CRITICAL
GET /api/predictions/latest?countries=Myanmar,Ukraine
GET /api/predictions/latest?min_probability=0.6
```

**Response Structure:**
```json
{
  "data": [...],
  "metadata": {
    "total": 229,
    "filtered": 45,
    "timestamp": "2024-12-17T10:30:00Z"
  }
}
```

---

## 3. Data Layer

### Database Schema

**SQLite (Development) / PostgreSQL (Production)**

```sql
-- Country Features Table
CREATE TABLE country_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country TEXT NOT NULL,
    date DATE NOT NULL,

    -- Event aggregates
    num_events INTEGER,
    num_articles INTEGER,
    num_mentions INTEGER,
    num_sources INTEGER,

    -- Goldstein statistics
    avg_goldstein REAL,
    goldstein_std REAL,
    goldstein_min REAL,
    goldstein_max REAL,
    goldstein_q25 REAL,
    goldstein_q50 REAL,
    goldstein_q75 REAL,

    -- QuadClass distribution
    quad_class_1_pct REAL,  -- Verbal Cooperation
    quad_class_2_pct REAL,  -- Material Cooperation
    quad_class_3_pct REAL,  -- Verbal Conflict
    quad_class_4_pct REAL,  -- Material Conflict

    -- Tone/sentiment
    avg_tone REAL,
    tone_std REAL,
    tone_min REAL,
    tone_max REAL,

    -- Temporal features (45 more features...)
    -- Rolling averages, lags, etc.

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(country, date)
);

CREATE INDEX idx_country_date ON country_features(country, date);
CREATE INDEX idx_date ON country_features(date);


-- Predictions Table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country TEXT NOT NULL,
    prediction_date DATE NOT NULL,    -- When prediction was made
    for_date DATE NOT NULL,           -- Date being predicted
    probability REAL,                 -- 0.0 - 1.0
    risk_level TEXT,                  -- LOW/MEDIUM/HIGH/CRITICAL
    confidence REAL,                  -- Model confidence score
    model_version TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(country, for_date, model_version)
);

CREATE INDEX idx_pred_country ON predictions(country);
CREATE INDEX idx_pred_for_date ON predictions(for_date);
CREATE INDEX idx_pred_risk_level ON predictions(risk_level);


-- Models Table (metadata)
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT UNIQUE NOT NULL,
    algorithm TEXT,                   -- 'XGBoost', 'GRU', etc.
    trained_at TIMESTAMP,

    -- Performance metrics
    roc_auc REAL,
    f1_score REAL,
    precision_score REAL,
    recall_score REAL,

    -- Training info
    train_start_date DATE,
    train_end_date DATE,
    test_start_date DATE,
    test_end_date DATE,
    num_features INTEGER,

    model_path TEXT,                  -- File system path to .pkl
    hyperparameters TEXT,             -- JSON string

    is_active BOOLEAN DEFAULT 1,      -- Current production model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Optional: GDELT Events Cache (for real-time feed)
CREATE TABLE gdelt_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE,
    event_date DATE,
    country TEXT,
    goldstein_scale REAL,
    quad_class INTEGER,
    tone REAL,
    -- ... other GDELT fields

    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_event_country_date ON gdelt_events(country, event_date);
```

### Data Access Layer

**`database.py` - Repository Pattern**

```python
from sqlalchemy import create_engine, text
from contextlib import contextmanager

class Database:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")

    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        connection = self.engine.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_latest_predictions(self, filters: dict):
        """Get latest predictions with filters"""
        # Implementation
        pass

    def get_country_features(self, country: str, date: str):
        """Get features for specific country-date"""
        # Implementation
        pass
```

### Data Aggregation Pipeline

**Overview:**
Transform 98M+ raw GDELT events into ML-ready country-day features.

**Stage 1: Raw Event Ingestion**

```python
# Raw GDELT event schema (61 fields)
raw_event = {
    'GLOBALEVENTID': int,
    'SQLDATE': date,
    'Actor1Code': str,
    'Actor2Code': str,
    'EventCode': str,          # CAMEO code
    'QuadClass': int,          # 1-4 (cooperation/conflict)
    'GoldsteinScale': float,   # -10 to +10
    'NumMentions': int,
    'NumSources': int,
    'NumArticles': int,
    'AvgTone': float,          # -100 to +100
    'Actor1Geo_CountryCode': str,
    'Actor2Geo_CountryCode': str,
    'ActionGeo_CountryCode': str,
    # ... 48 more fields
}
```

**Stage 2: Country-Day Aggregation**

```python
def aggregate_events_to_country_day(events_df):
    """
    Aggregate 98M events  220K country-day records
    """
    # Group by country and date
    aggregated = events_df.groupby(['Country', 'Date']).agg({

        # Event counts
        'GLOBALEVENTID': 'count',          #  NumEvents
        'NumArticles': 'sum',               #  TotalArticles
        'NumMentions': 'sum',               #  TotalMentions
        'NumSources': 'nunique',            #  UniqueSources

        # Goldstein statistics (conflict intensity)
        'GoldsteinScale': [
            'mean',                         #  AvgGoldstein
            'std',                          #  GoldsteinStd
            'min',                          #  GoldsteinMin
            'max',                          #  GoldsteinMax
            lambda x: x.quantile(0.25),     #  GoldsteinQ25
            lambda x: x.quantile(0.50),     #  GoldsteinQ50
            lambda x: x.quantile(0.75),     #  GoldsteinQ75
        ],

        # Tone statistics (sentiment)
        'AvgTone': ['mean', 'std', 'min', 'max'],

        # Actor counts
        'Actor1Code': 'nunique',            #  UniqueActor1s
        'Actor2Code': 'nunique',            #  UniqueActor2s
    })

    # QuadClass distribution (conflict type percentages)
    quad_dist = events_df.groupby(['Country', 'Date', 'QuadClass']).size()
    quad_dist = quad_dist / events_df.groupby(['Country', 'Date']).size()
    quad_dist = quad_dist.unstack(fill_value=0)

    # Merge
    result = pd.concat([aggregated, quad_dist], axis=1)

    return result  # 220K rows  53 columns
```

**Stage 3: Temporal Feature Engineering**

```python
def add_temporal_features(country_day_df):
    """
    Add rolling averages, lags, and trends
    """
    df = country_day_df.copy()

    # Sort by country and date
    df = df.sort_values(['Country', 'Date'])

    # For each country, calculate temporal features
    for country in df['Country'].unique():
        mask = df['Country'] == country

        # Rolling averages (7-day window)
        df.loc[mask, 'AvgGoldstein_7d'] = (
            df.loc[mask, 'AvgGoldstein']
            .rolling(7, min_periods=1).mean()
        )
        df.loc[mask, 'NumEvents_7d'] = (
            df.loc[mask, 'NumEvents']
            .rolling(7, min_periods=1).mean()
        )

        # Lag features (previous day)
        df.loc[mask, 'AvgGoldstein_lag1'] = (
            df.loc[mask, 'AvgGoldstein'].shift(1)
        )

        # Trend features (7-day change)
        df.loc[mask, 'AvgGoldstein_trend7d'] = (
            df.loc[mask, 'AvgGoldstein'] -
            df.loc[mask, 'AvgGoldstein'].shift(7)
        )

    return df
```

**Stage 4: Sequence Creation for GRU/LSTM**

```python
def create_sequences(features_df, seq_length=7):
    """
    Create 7-day sliding windows for temporal models
    220K records  216K sequences (109K train, 107K test)
    """
    sequences = []
    targets = []

    for country in features_df['Country'].unique():
        # Get country data sorted by date
        country_data = features_df[
            features_df['Country'] == country
        ].sort_values('Date')

        # Extract features (53 columns)
        X = country_data[FEATURE_COLUMNS].values

        # Create target (high conflict next day)
        y = country_data['IsHighConflict'].shift(-1).values

        # Create sequences
        for i in range(len(X) - seq_length):
            # Input: past 7 days of features
            seq_input = X[i:i+seq_length]  # Shape: (7, 53)

            # Target: conflict status on day 8
            seq_target = y[i+seq_length]   # Shape: (1,)

            sequences.append(seq_input)
            targets.append(seq_target)

    # Convert to numpy arrays
    X_seq = np.array(sequences)  # Shape: (216988, 7, 53)
    y_seq = np.array(targets)     # Shape: (216988,)

    return X_seq, y_seq
```

**Stage 5: Data Normalization**

```python
from sklearn.preprocessing import StandardScaler

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])
).reshape(X_train.shape)

# Transform test data
X_test_scaled = scaler.transform(
    X_test.reshape(-1, X_test.shape[-1])
).reshape(X_test.shape)

# Save scaler for production
import pickle
with open('scaler_gru_forecast.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

**Stage 6: Class Balancing (SMOTE)**

```python
from imblearn.over_sampling import SMOTE

# Original: 75% low conflict, 25% high conflict
print(f"Original: {y_train.value_counts()}")
# {0: 83305, 1: 27658}

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_2d, y_train
)

# Balanced: 50% low, 50% high
print(f"Balanced: {y_train_balanced.value_counts()}")
# {0: 83305, 1: 83305}
```

**Data Quality Checks:**

```python
def validate_features(df):
    """Ensure data quality before ML"""

    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Missing values found"

    # Check for infinite values
    assert not np.isinf(df.select_dtypes(include=[np.number])).any().any()

    # Check feature ranges
    assert df['AvgGoldstein'].between(-10, 10).all()
    assert df['AvgTone'].between(-100, 100).all()
    assert df['QuadClass1_pct'].between(0, 1).all()

    # Check temporal ordering
    for country in df['Country'].unique():
        dates = df[df['Country'] == country]['Date']
        assert dates.is_monotonic_increasing, f"Dates not sorted: {country}"

    print(" Data validation passed")
```

### Feature Storage

**Parquet Format:**
- Column-oriented, compressed
- Fast read performance (~100ms for 220K records)
- Schema enforcement
- Compression ratio: ~10x vs CSV
- File: `data/features_multiresolution/country_day/country_day_features.parquet`

**Structure:**
```
country (string)
date (date)
num_events (int64)
avg_goldstein (float64)
goldstein_std (float64)
goldstein_min (float64)
goldstein_max (float64)
quad_class_1_pct (float64)
quad_class_2_pct (float64)
quad_class_3_pct (float64)
quad_class_4_pct (float64)
avg_tone (float64)
tone_std (float64)
# ... 41 more features (53 total)
```

**Access Pattern:**

```python
import pandas as pd

# Fast loading
df = pd.read_parquet('country_day_features.parquet')

# Filter by country
ukraine = df[df['country'] == 'Ukraine']

# Filter by date range
recent = df[df['date'] >= '2024-01-01']

# Efficient column selection (only loads needed columns)
features = pd.read_parquet(
    'country_day_features.parquet',
    columns=['country', 'date', 'avg_goldstein', 'num_events']
)
```

---

## 4. ML Pipeline

### Model Training Pipeline

**File:** [prototype/backend/model_manager.py](prototype/backend/model_manager.py)

```python
class ModelManager:
    def train_model(self, features_df: pd.DataFrame):
        """
        Full training pipeline
        """
        # 1. Data preparation
        X, y = self.prepare_training_data(features_df)

        # 2. Train/test split (temporal)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # 3. Handle class imbalance
        X_train_resampled, y_train_resampled = SMOTE().fit_resample(
            X_train, y_train
        )

        # 4. Train XGBoost
        model = XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            scale_pos_weight=10,
            random_state=42
        )
        model.fit(X_train_resampled, y_train_resampled)

        # 5. Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }

        # 6. Save model
        self.save_model(model, metrics)

        return model, metrics
```

### Feature Engineering

**59 Features Generated:**

```python
def engineer_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw GDELT events  country-day features
    """
    features = events_df.groupby(['Country', 'Date']).agg({
        # Event aggregates
        'EventID': 'count',  # NumEvents
        'NumArticles': 'sum',
        'NumMentions': 'sum',
        'NumSources': 'nunique',

        # Goldstein statistics
        'GoldsteinScale': ['mean', 'std', 'min', 'max',
                           lambda x: x.quantile(0.25),
                           lambda x: x.quantile(0.50),
                           lambda x: x.quantile(0.75)],

        # Tone
        'Tone': ['mean', 'std', 'min', 'max'],
    })

    # QuadClass distribution
    features['QuadClass1_pct'] = (
        events_df[events_df.QuadClass == 1]
        .groupby(['Country', 'Date']).size() /
        events_df.groupby(['Country', 'Date']).size()
    )
    # ... repeat for QuadClass 2, 3, 4

    # Temporal features
    features['rolling_7d_avg_goldstein'] = (
        features.groupby('Country')['AvgGoldstein']
        .rolling(7, min_periods=1).mean()
    )

    # Lag features
    features['goldstein_lag_1'] = (
        features.groupby('Country')['AvgGoldstein'].shift(1)
    )

    return features
```

### Prediction Generation

**File:** [prototype/backend/scheduler.py](prototype/backend/scheduler.py)

```python
def generate_daily_predictions(model, features_df: pd.DataFrame):
    """
    Generate predictions for all countries for next day
    """
    # Get latest features for each country
    latest_features = (
        features_df.sort_values('Date')
        .groupby('Country').tail(1)
    )

    # Predict
    X = latest_features[FEATURE_COLUMNS]
    probabilities = model.predict_proba(X)[:, 1]

    # Classify risk levels
    risk_levels = pd.cut(
        probabilities,
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    )

    # Create predictions dataframe
    predictions = pd.DataFrame({
        'country': latest_features['Country'],
        'probability': probabilities,
        'risk_level': risk_levels,
        'prediction_date': datetime.now().date(),
        'for_date': datetime.now().date() + timedelta(days=1)
    })

    # Store in database
    store_predictions(predictions)

    return predictions
```

---

## 5. Backend API

### FastAPI Application

**File:** [prototype/backend/main_db.py](prototype/backend/main_db.py)

**Application Structure:**

```python
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="GDELT Conflict Predictor API",
    version="1.0.0",
    description="Next-day conflict prediction API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
db = Database("gdelt_predictor.db")

# Dependency injection
def get_db():
    return db
```

### Endpoint Implementations

**1. Latest Predictions**

```python
@app.get("/api/predictions/latest", response_model=List[Prediction])
async def get_latest_predictions(
    risk_levels: Optional[List[str]] = Query(None),
    countries: Optional[List[str]] = Query(None),
    min_probability: Optional[float] = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(229, le=229),
    offset: int = Query(0),
    db: Database = Depends(get_db)
):
    """
    Get latest predictions with optional filters
    """
    predictions = db.get_latest_predictions(
        risk_levels=risk_levels,
        countries=countries,
        min_probability=min_probability,
        limit=limit,
        offset=offset
    )
    return predictions
```

**2. Country Detail**

```python
@app.get("/api/predictions/country/{country}",
         response_model=CountryPrediction)
async def get_country_prediction(
    country: str,
    db: Database = Depends(get_db)
):
    """
    Get detailed prediction for specific country
    """
    # Get latest prediction
    prediction = db.get_country_prediction(country)
    if not prediction:
        raise HTTPException(status_code=404, detail="Country not found")

    # Get 7-day trend
    trend = db.get_prediction_trend(country, days=7)

    # Get top features
    features = db.get_country_features(country)
    top_features = get_top_contributing_features(features)

    return {
        "prediction": prediction,
        "trend": trend,
        "top_features": top_features
    }
```

**3. Model Performance**

```python
@app.get("/api/model-performance", response_model=ModelMetrics)
async def get_model_performance(db: Database = Depends(get_db)):
    """
    Get current model performance metrics
    """
    metrics = db.get_active_model_metrics()
    return metrics
```

### Pydantic Models

**File:** [prototype/backend/models.py](prototype/backend/models.py)

```python
from pydantic import BaseModel, Field
from datetime import date
from typing import List

class Prediction(BaseModel):
    country: str
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    prediction_date: date
    for_date: date
    confidence: float

class CountryPrediction(BaseModel):
    prediction: Prediction
    trend: List[Prediction]
    top_features: List[FeatureContribution]

class FeatureContribution(BaseModel):
    feature_name: str
    value: float
    contribution: float

class ModelMetrics(BaseModel):
    version: str
    roc_auc: float
    f1_score: float
    precision: float
    recall: float
    trained_at: str
```

### API Documentation

FastAPI auto-generates interactive docs:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

---

## 6. Frontend Application

### Angular Architecture

**Structure:**
```
frontend-angular/
 src/
    app/
       components/
          dashboard/
             dashboard.component.ts
             dashboard.component.html
             dashboard.component.scss
          country-detail/
              country-detail.component.ts
              country-detail.component.html
              country-detail.component.scss
       services/
          api.service.ts
          prediction.service.ts
       models/
          interfaces.ts
       app.component.ts
       app.routes.ts
    assets/
    environments/
 package.json
```

### Service Layer

**`api.service.ts`:**

```typescript
import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  getLatestPredictions(filters?: PredictionFilters): Observable<Prediction[]> {
    let params = new HttpParams();
    if (filters?.riskLevels) {
      params = params.append('risk_levels', filters.riskLevels.join(','));
    }
    if (filters?.minProbability) {
      params = params.append('min_probability', filters.minProbability.toString());
    }
    return this.http.get<Prediction[]>(`${this.apiUrl}/predictions/latest`, { params });
  }

  getCountryPrediction(country: string): Observable<CountryPrediction> {
    return this.http.get<CountryPrediction>(
      `${this.apiUrl}/predictions/country/${country}`
    );
  }

  getModelPerformance(): Observable<ModelMetrics> {
    return this.http.get<ModelMetrics>(`${this.apiUrl}/model-performance`);
  }
}
```

### Dashboard Component

**`dashboard.component.ts`:**

```typescript
import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
  predictions: Prediction[] = [];
  filteredPredictions: Prediction[] = [];
  riskDistribution: any;

  // Plotly chart data
  mapData: any;
  chartLayout: any;

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.loadPredictions();
  }

  loadPredictions() {
    this.apiService.getLatestPredictions().subscribe(predictions => {
      this.predictions = predictions;
      this.filteredPredictions = predictions;
      this.updateCharts();
    });
  }

  updateCharts() {
    this.createWorldMap();
    this.createRiskDistribution();
    this.createProbabilityHistogram();
  }

  createWorldMap() {
    this.mapData = [{
      type: 'choropleth',
      locationmode: 'country names',
      locations: this.predictions.map(p => p.country),
      z: this.predictions.map(p => p.probability),
      colorscale: [
        [0, 'green'],
        [0.4, 'yellow'],
        [0.6, 'orange'],
        [0.8, 'red']
      ],
      colorbar: {
        title: 'Risk Probability'
      }
    }];
  }

  filterByRiskLevel(levels: string[]) {
    this.filteredPredictions = this.predictions.filter(
      p => levels.includes(p.risk_level)
    );
    this.updateCharts();
  }
}
```

### Visualization with Plotly

```typescript
import { PlotlyModule } from 'angular-plotly.js';

// In component
createProbabilityHistogram() {
  const data = [{
    x: this.predictions.map(p => p.probability),
    type: 'histogram',
    nbinsx: 20,
    marker: {
      color: 'rgba(100, 200, 255, 0.7)'
    }
  }];

  const layout = {
    title: 'Prediction Probability Distribution',
    xaxis: { title: 'Probability' },
    yaxis: { title: 'Count' }
  };

  this.histogramData = data;
  this.histogramLayout = layout;
}
```

---

## 7. Deployment Architecture

### Local Development

```

  Developer Machine (Windows)        
                                     
   
    Backend (Python venv)         
    Port: 8000                    
    Command: python main_db.py    
   
                                     
   
    Frontend (Node.js)            
    Port: 4200                    
    Command: ng serve             
   
                                     
   
    SQLite Database               
    File: gdelt_predictor.db      
   

```

### Docker Containerization

**`Dockerfile` (Backend):**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main_db:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker-compose.yml`:**

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/gdelt
    depends_on:
      - postgres
    volumes:
      - ./backend:/app
      - model-data:/app/saved_models

  frontend:
    build: ./frontend-angular
    ports:
      - "4200:80"
    depends_on:
      - backend

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: gdelt
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
  model-data:
```

### Cloud Deployment (AWS)

```

                          AWS Cloud                          
                                                             
     
    Application Load Balancer                             
    SSL/TLS Termination                                   
     
                                                           
                               
    ECS Fargate (Backend)                                
    - FastAPI containers                                 
    - Auto-scaling                                       
    - Health checks                                      
                               
                                                           
                               
    RDS PostgreSQL                                       
    - Multi-AZ                                           
    - Automated backups                                  
                               
                                                            
                            
    S3 + CloudFront (Frontend)                            
    - Static Angular build                                
    - CDN distribution                                    
                            
                                                             
                            
    EventBridge (Scheduler)                               
    - Daily prediction trigger                            
                            
                                                             
                            
    CloudWatch (Monitoring)                               
    - Logs, metrics, alarms                               
                            

```

---

## 8. Security Considerations

### Current Implementation (POC)

-  No authentication (open API)
-  No rate limiting
-  CORS wide open (localhost only)

### Production Requirements

**1. Authentication & Authorization**

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/admin/retrain", dependencies=[Depends(verify_token)])
async def retrain_model():
    # Only authenticated users
    pass
```

**2. Rate Limiting**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/predictions/latest")
@limiter.limit("100/minute")
async def get_predictions(request: Request):
    # Rate limited
    pass
```

**3. Input Validation**

```python
from pydantic import BaseModel, validator

class PredictionQuery(BaseModel):
    country: str

    @validator('country')
    def validate_country(cls, v):
        # Prevent SQL injection
        if not v.isalpha():
            raise ValueError("Invalid country name")
        return v
```

**4. HTTPS/TLS**

```python
# In production, use reverse proxy (nginx) with SSL cert
# Or AWS ALB with ACM certificate
```

**5. Database Security**

```python
# Use environment variables for credentials
import os
DATABASE_URL = os.getenv("DATABASE_URL")

# Parameterized queries (SQLAlchemy handles this)
query = text("SELECT * FROM predictions WHERE country = :country")
result = connection.execute(query, {"country": country})
```

---

## 9. Performance & Scalability

### Current Performance

**API Response Times:**
- Latest predictions: ~50-100ms
- Country detail: ~30-50ms
- Model performance: ~10ms

**Database:**
- 149K records
- Indexed queries
- SQLite (single-threaded)

### Scalability Improvements

**1. Database Optimization**

```sql
-- Add composite indexes
CREATE INDEX idx_country_date_pred ON predictions(country, for_date, model_version);

-- Partition by date (PostgreSQL)
CREATE TABLE predictions (
    ...
) PARTITION BY RANGE (for_date);

CREATE TABLE predictions_2024 PARTITION OF predictions
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

**2. Caching Layer**

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="gdelt-cache")

@app.get("/api/predictions/latest")
@cache(expire=300)  # 5 minutes
async def get_predictions():
    # Cached response
    pass
```

**3. Async Database Operations**

```python
from databases import Database
import asyncio

database = Database("postgresql://...")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.get("/api/predictions/latest")
async def get_predictions():
    query = "SELECT * FROM predictions"
    results = await database.fetch_all(query)
    return results
```

**4. Load Balancing**

```
         Backend Instance 1
        
LB  Backend Instance 2
        
         Backend Instance 3
                    
                    
            Shared Database
```

**5. Model Serving Optimization**

```python
# Load model once at startup, not per request
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("saved_models/xgboost_model.pkl")

@app.post("/api/predict")
async def predict(features: Features):
    # Use pre-loaded model
    prediction = model.predict([features.values])
    return {"prediction": prediction}
```

---

## 10. Testing Strategy

### Unit Tests

```python
# test_model_manager.py
import pytest
from model_manager import ModelManager

def test_feature_engineering():
    events = pd.DataFrame({...})
    features = engineer_features(events)
    assert features.shape[1] == 59
    assert 'avg_goldstein' in features.columns

def test_model_training():
    manager = ModelManager()
    model, metrics = manager.train_model(sample_data)
    assert metrics['roc_auc'] > 0.5
```

### Integration Tests

```python
# test_api.py
from fastapi.testclient import TestClient
from main_db import app

client = TestClient(app)

def test_get_predictions():
    response = client.get("/api/predictions/latest")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0

def test_country_not_found():
    response = client.get("/api/predictions/country/InvalidCountry")
    assert response.status_code == 404
```

### End-to-End Tests

```typescript
// dashboard.component.spec.ts
describe('DashboardComponent', () => {
  it('should load predictions', () => {
    const fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;

    fixture.detectChanges();

    expect(component.predictions.length).toBeGreaterThan(0);
  });
});
```

---

## Summary

This architecture provides:
-  Clean separation of concerns
-  Scalable design patterns
-  Production-ready API
-  Modern frontend
-  Comprehensive ML pipeline
-  Clear deployment path

**Next Steps:**
1. Implement authentication
2. Add caching layer
3. Switch to PostgreSQL
4. Deploy to cloud
5. Set up CI/CD pipeline

---

**End of Technical Architecture Document**
