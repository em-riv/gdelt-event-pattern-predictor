# Backend Scripts

Utility scripts for data processing, model training, and debugging.

## Organization

This folder contains development and maintenance scripts that aren't part of the main API.

### Data Processing
- `add_2025_data.py` - Add 2025 data to database
- `data_fetcher.py` - GDELT data ingestion (note: also in root as core module)
- `gdelt_live_fetcher.py` - Live GDELT event fetching
- `process_raw_gdelt.py` - Raw data processing pipeline
- `populate_db_simple.py` - Simple database population

### Model Training and Updates
- `train_gru_forecasting.py` - Train Bidirectional GRU model
- `retrain_proper_forecasting.py` - Retraining pipeline
- `retrain_with_better_features.py` - Retraining with improved features
- `generate_predictions_gru.py` - Generate predictions with GRU
- `generate_predictions_improved.py` - Generate improved predictions
- `generate_multi_horizon_predictions.py` - Multi-horizon prediction generation
- `update_production_models.py` - Update deployed models
- `update_xgboost_model.py` - Update XGBoost model
- `update_database_schema.py` - Database schema updates

### Analysis and Debugging
- `analyze_nan_features.py` - Analyze missing values
- `check_columns.py` - Column validation
- `check_conflict_zones.py` - Conflict zone analysis
- `check_db.py` - Database health check
- `check_recent_conflict_intensity.py` - Recent intensity analysis
- `check_status.py` - System status check
- `compare_model_features.py` - Feature comparison between models
- `debug_gru_predictions.py` - GRU prediction debugging
- `debug_ukraine.py` - Specific country debugging
- `fix_2025_features.py` - Fix 2025 feature data
- `fix_predictions.py` - Fix prediction issues

### Testing and Server
- `test_api.py` - API endpoint testing
- `test_api_components.py` - Component testing
- `test_gru_predictions.py` - GRU-specific tests
- `start_server.py` - Server startup (use main_api.py instead)
- `scheduler.py` - Prediction scheduling
- `create_robust_gru_features.py` - Feature creation for GRU

## Usage

Most scripts are run with:
```bash
python scripts/<script_name>.py [arguments]
```

For production, use the main API:
```bash
python main_api.py
```
