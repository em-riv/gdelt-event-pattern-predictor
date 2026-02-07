"""
Update database schema to support multi-horizon predictions
"""
from database import get_db

print("Updating database schema...")

db = get_db()

# Check if horizon_days column exists
cursor = db.conn.execute("PRAGMA table_info(predictions)")
columns = [row[1] for row in cursor.fetchall()]

if 'horizon_days' not in columns:
    print("Adding horizon_days column...")
    db.conn.execute("ALTER TABLE predictions ADD COLUMN horizon_days INTEGER")
    db.conn.commit()
    print(" Column added")
else:
    print(" horizon_days column already exists")

# Update existing predictions with calculated horizon
print("\nCalculating horizon_days for existing predictions...")
db.conn.execute("""
    UPDATE predictions
    SET horizon_days = CAST((julianday(target_date) - julianday(prediction_date)) AS INTEGER)
    WHERE horizon_days IS NULL
""")
db.conn.commit()

# Show sample
cursor = db.conn.execute("""
    SELECT country, prediction_date, target_date, horizon_days, risk_level
    FROM predictions
    LIMIT 10
""")
print("\nSample predictions:")
for row in cursor.fetchall():
    print(f"  {row[0]:10s} | Pred: {row[1]} | Target: {row[2]} | Horizon: {row[3]:2d} days | Risk: {row[4]}")

print("\n Database schema updated!")
