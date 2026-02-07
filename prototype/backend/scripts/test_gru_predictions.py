"""
Direct test of GRU predictions from database
"""
import sqlite3
import json
from pathlib import Path

def test_gru_predictions():
    print(" Testing GRU predictions from database...")
    
    # Connect to database
    db_path = Path('conflict_predictor.db')
    if not db_path.exists():
        print(" Database file not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get prediction count
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_preds = cursor.fetchone()[0]
    print(f" Total predictions in database: {total_preds}")
    
    # Get latest predictions
    cursor.execute("""
        SELECT country, conflict_probability, risk_level, model_version, created_at 
        FROM predictions 
        ORDER BY conflict_probability DESC 
        LIMIT 10
    """)
    
    predictions = cursor.fetchall()
    
    print(f"\n Top 10 High-Risk Countries (GRU Model):")
    print("=" * 60)
    for i, (country, prob, risk, model, created) in enumerate(predictions, 1):
        print(f"{i:2d}. {country:12s} {prob*100:5.1f}% {risk:8s} {model}")
    
    # Test specific countries
    test_countries = ['USA', 'CHN', 'RUS', 'UKR', 'ISR', 'IRN']
    print(f"\n Specific Country Predictions:")
    print("=" * 50)
    
    for country in test_countries:
        cursor.execute("""
            SELECT conflict_probability, risk_level 
            FROM predictions 
            WHERE country = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (country,))
        
        result = cursor.fetchone()
        if result:
            prob, risk = result
            print(f"{country:3s}: {prob*100:5.1f}% ({risk})")
        else:
            print(f"{country:3s}: No prediction found")
    
    conn.close()
    print(f"\n Database test complete!")

if __name__ == "__main__":
    test_gru_predictions()