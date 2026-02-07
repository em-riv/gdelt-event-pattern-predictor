import sqlite3
import os

# Check all database files and their tables
db_files = [
    'conflict_predictor.db',
    'gdelt_predictor.db'
]

for db_file in db_files:
    if os.path.exists(db_file):
        print(f"\n=== Checking {db_file} ===")
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Check all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Tables: {[t[0] for t in tables]}")
            
            # For each table, show columns
            for table_name in [t[0] for t in tables]:
                cursor.execute(f"PRAGMA table_info({table_name})")
                cols = cursor.fetchall()
                print(f"\n  {table_name} columns: {[c[1] for c in cols]}")
                
                # Count rows
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"  {table_name} rows: {count}")
                
                # If it's gdelt_events, show sample with URLs
                if 'events' in table_name.lower() and count > 0:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                    sample = cursor.fetchall()
                    print(f"  Sample: {sample[:1]}")
            
            conn.close()
        except Exception as e:
            print(f"Error with {db_file}: {e}")
    else:
        print(f"{db_file} does not exist")