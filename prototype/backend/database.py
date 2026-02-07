"""
Database setup and models
SQLite database for storing GDELT data and predictions
"""

import sqlite3
from datetime import datetime, date as date_type
from pathlib import Path
from typing import List, Optional, Dict
import json


class Database:
    """Database manager for GDELT conflict predictor"""

    def __init__(self, db_path: str = "gdelt_predictor.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.initialize_db()

    def initialize_db(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create tables
        self.conn.executescript("""
            -- Country-day features table
            CREATE TABLE IF NOT EXISTS country_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT NOT NULL,
                date DATE NOT NULL,
                num_events INTEGER,
                avg_goldstein REAL,
                quad_class_1 INTEGER,
                quad_class_2 INTEGER,
                quad_class_3 INTEGER,
                quad_class_4 INTEGER,
                is_high_conflict INTEGER,
                features_json TEXT,  -- Store all features as JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(country, date)
            );

            -- Predictions table
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT NOT NULL,
                prediction_date DATE NOT NULL,  -- Date prediction was made
                target_date DATE NOT NULL,      -- Date being predicted
                conflict_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(country, prediction_date, target_date)
            );

            -- Model metadata table
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_path TEXT NOT NULL,
                metrics_json TEXT,  -- ROC-AUC, F1, etc. as JSON
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            );

            -- Raw GDELT events cache (optional - for recent data)
            CREATE TABLE IF NOT EXISTS gdelt_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                date DATE NOT NULL,
                country TEXT,
                event_code TEXT,
                goldstein_scale REAL,
                quad_class INTEGER,
                num_mentions INTEGER,
                raw_data_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_country_features_date
                ON country_features(country, date DESC);
            CREATE INDEX IF NOT EXISTS idx_predictions_country_date
                ON predictions(country, target_date DESC);
            CREATE INDEX IF NOT EXISTS idx_gdelt_events_date
                ON gdelt_events(date DESC, country);
        """)

        self.conn.commit()
        print("[OK] Database initialized")

    def insert_country_features(self, country: str, date: date_type, features: Dict):
        """Insert or update country-day features"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO country_features
            (country, date, num_events, avg_goldstein, quad_class_1, quad_class_2,
             quad_class_3, quad_class_4, is_high_conflict, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            country,
            date,
            features.get('NumEvents_sum', 0),
            features.get('AvgGoldstein_sum', 0),
            features.get('QuadClass_1_sum', 0),
            features.get('QuadClass_2_sum', 0),
            features.get('QuadClass_3_sum', 0),
            features.get('QuadClass_4_sum', 0),
            features.get('IsHighConflict_sum', 0),
            json.dumps(features)
        ))

        self.conn.commit()
        return cursor.lastrowid

    def bulk_insert_features(self, rows: List[tuple]):
        """Bulk insert country features for better performance"""
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO country_features
            (country, date, num_events, avg_goldstein, quad_class_1, quad_class_2,
             quad_class_3, quad_class_4, is_high_conflict, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()

    def get_latest_features(self, country: str, limit: int = 7) -> List[Dict]:
        """Get latest N days of features for a country"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM country_features
            WHERE country = ?
            ORDER BY date DESC
            LIMIT ?
        """, (country, limit))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_all_latest_features(self) -> List[Dict]:
        """Get latest features for all countries (for predictions)"""
        cursor = self.conn.cursor()

        # Get the most recent date for each country
        cursor.execute("""
            SELECT cf.* FROM country_features cf
            INNER JOIN (
                SELECT country, MAX(date) as max_date
                FROM country_features
                GROUP BY country
            ) latest ON cf.country = latest.country AND cf.date = latest.max_date
            ORDER BY cf.country
        """)

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_all_features(self) -> List[Dict]:
        """Get ALL features for all countries (for training)"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM country_features
            ORDER BY country, date
        """)

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def insert_prediction(self, country: str, prediction_date: date_type,
                         target_date: date_type, probability: float,
                         risk_level: str, confidence: float, model_version: str = "1.0"):
        """Insert a new prediction"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO predictions
            (country, prediction_date, target_date, conflict_probability,
             risk_level, confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (country, prediction_date, target_date, probability, risk_level, confidence, model_version))

        self.conn.commit()
        return cursor.lastrowid

    def get_latest_predictions(self) -> List[Dict]:
        """Get latest predictions for all countries"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT p.* FROM predictions p
            INNER JOIN (
                SELECT country, MAX(prediction_date) as max_date
                FROM predictions
                GROUP BY country
            ) latest ON p.country = latest.country AND p.prediction_date = latest.max_date
            ORDER BY p.conflict_probability DESC
        """)

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_country_prediction_history(self, country: str, days: int = 7) -> List[Dict]:
        """Get prediction history for a country"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM predictions
            WHERE country = ?
            ORDER BY prediction_date DESC
            LIMIT ?
        """, (country, days))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def save_model_metadata(self, model_name: str, model_version: str,
                           model_path: str, metrics: Dict):
        """Save model training metadata"""
        cursor = self.conn.cursor()

        # Deactivate previous versions
        cursor.execute("UPDATE models SET is_active = 0 WHERE model_name = ?", (model_name,))

        # Insert new model
        cursor.execute("""
            INSERT INTO models (model_name, model_version, model_path, metrics_json)
            VALUES (?, ?, ?, ?)
        """, (model_name, model_version, model_path, json.dumps(metrics)))

        self.conn.commit()
        return cursor.lastrowid

    def get_active_model(self, model_name: str = "XGBoost") -> Optional[Dict]:
        """Get active model metadata"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM models
            WHERE model_name = ? AND is_active = 1
            ORDER BY trained_at DESC
            LIMIT 1
        """, (model_name,))

        row = cursor.fetchone()
        return dict(row) if row else None

    def get_data_summary(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()

        stats = {}

        # Feature count
        cursor.execute("SELECT COUNT(*) as count FROM country_features")
        stats['total_features'] = cursor.fetchone()['count']

        # Countries count
        cursor.execute("SELECT COUNT(DISTINCT country) as count FROM country_features")
        stats['total_countries'] = cursor.fetchone()['count']

        # Date range
        cursor.execute("SELECT MIN(date) as min_date, MAX(date) as max_date FROM country_features")
        row = cursor.fetchone()
        stats['date_range'] = {'min': row['min_date'], 'max': row['max_date']}

        # Predictions count
        cursor.execute("SELECT COUNT(*) as count FROM predictions")
        stats['total_predictions'] = cursor.fetchone()['count']

        # Latest prediction date
        cursor.execute("SELECT MAX(prediction_date) as latest FROM predictions")
        stats['latest_prediction'] = cursor.fetchone()['latest']

        return stats

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Singleton instance
_db_instance = None

def get_db() -> Database:
    """Get database singleton instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
