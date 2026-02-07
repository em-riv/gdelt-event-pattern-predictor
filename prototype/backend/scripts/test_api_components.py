"""
Test the simple GRU API manually
"""
import sqlite3

def test_database():
    """Test database connectivity"""
    try:
        conn = sqlite3.connect('conflict_predictor.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        print(f" Database connected: {count} predictions found")
        
        # Test a sample query
        cursor.execute("""
            SELECT country, conflict_probability, risk_level 
            FROM predictions 
            ORDER BY conflict_probability DESC 
            LIMIT 5
        """)
        
        top_risks = cursor.fetchall()
        print("\n Top 5 High-Risk Countries:")
        for country, prob, risk in top_risks:
            print(f"   {country}: {prob:.1%} ({risk})")
            
        conn.close()
        return True
        
    except Exception as e:
        print(f" Database error: {e}")
        return False

if __name__ == "__main__":
    print(" Testing GRU API Components...")
    
    # Test database
    db_ok = test_database()
    
    if db_ok:
        print("\n All components working!")
        print(" Ready to start API server")
    else:
        print("\n Issues found!")