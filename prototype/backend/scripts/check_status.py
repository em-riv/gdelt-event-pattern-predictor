from database import get_db

db = get_db()
summary = db.get_data_summary()

print("="*60)
print("DATABASE STATUS")
print("="*60)
print(f"Total Predictions: {summary.get('total_predictions', 0):,}")
print(f"Total Countries:   {summary.get('total_countries', 0)}")
print(f"Latest Prediction: {summary.get('latest_prediction')}")
print(f"Date Range:        {summary.get('date_range')}")

cursor = db.conn.execute('SELECT model_name, model_version, trained_at, metrics FROM models WHERE is_active = 1')
print("\n" + "="*60)
print("ACTIVE MODELS")
print("="*60)
for row in cursor:
    print(f"\nModel: {row[0]} v{row[1]}")
    print(f"Trained: {row[2]}")
    import json
    metrics = json.loads(row[3])
    print(f"Metrics: ROC-AUC={metrics['roc_auc']*100:.2f}%, F1={metrics['f1_score']*100:.2f}%")
