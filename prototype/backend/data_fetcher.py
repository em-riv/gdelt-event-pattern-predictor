"""
GDELT Data Fetcher
Fetches and processes GDELT data for conflict prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from database import get_db


class GDELTDataFetcher:
    """Fetches and processes GDELT data"""

    def __init__(self):
        self.db = get_db()
        self.data_path = Path(__file__).parent.parent.parent / "data"

    def load_from_parquet(self, parquet_path: Optional[Path] = None) -> pd.DataFrame:
        """Load GDELT data from parquet file"""
        if parquet_path is None:
            # Try new 2023-2025 file first
            parquet_path = self.data_path / "features_multiresolution" / "country_day" / "country_day_features_2023_2025.parquet"

            # Fallback to old file
            if not parquet_path.exists():
                parquet_path = self.data_path / "features_multiresolution" / "country_day" / "country_day_features.parquet"

        if not parquet_path.exists():
            print(f"  Parquet file not found: {parquet_path}")
            return None

        print(f" Loading data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df['Date'] = pd.to_datetime(df['Date'])

        print(f" Loaded {len(df):,} rows")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Countries: {df['Country'].nunique()}")

        return df

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into features"""
        # Sort by country and date
        df = df.sort_values(['Country', 'Date'])

        # Create target variable (next day high conflict)
        conflict_col = 'IsHighConflict_sum'
        df['NextDay_Conflict'] = df.groupby('Country')[conflict_col].shift(-1)
        df['NextDay_HighConflict'] = (
            df['NextDay_Conflict'] >= df['NextDay_Conflict'].quantile(0.75)
        ).astype(int)

        # Drop NaN targets
        df = df.dropna(subset=['NextDay_Conflict', 'NextDay_HighConflict'])

        return df

    def import_to_database(self, df: pd.DataFrame, batch_size: int = 5000):
        """Import DataFrame to database using bulk inserts"""
        print(f" Importing {len(df):,} rows to database...")

        # Get feature columns
        exclude_cols = ['Country', 'Date', 'NextDay_Conflict', 'NextDay_HighConflict', 'TopRegion']
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        # Prepare bulk insert data
        import json
        rows_to_insert = []

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Prepare features dict - convert numpy types to Python native types
            features = {}
            for col in feature_cols:
                val = row[col]
                if pd.notna(val):
                    features[col] = float(val) if isinstance(val, (np.integer, np.floating)) else val
                else:
                    features[col] = 0.0

            # Add standard fields - ensure native Python types
            features['NumEvents_sum'] = float(row.get('NumEvents_sum', 0))
            features['AvgGoldstein_sum'] = float(row.get('AvgGoldstein_sum', 0))
            features['QuadClass_1_sum'] = float(row.get('QuadClass_1_sum', 0))
            features['QuadClass_2_sum'] = float(row.get('QuadClass_2_sum', 0))
            features['QuadClass_3_sum'] = float(row.get('QuadClass_3_sum', 0))
            features['QuadClass_4_sum'] = float(row.get('QuadClass_4_sum', 0))
            features['IsHighConflict_sum'] = float(row.get('IsHighConflict_sum', 0))

            rows_to_insert.append((
                row['Country'],
                row['Date'].date(),
                features.get('NumEvents_sum', 0),
                features.get('AvgGoldstein_sum', 0),
                features.get('QuadClass_1_sum', 0),
                features.get('QuadClass_2_sum', 0),
                features.get('QuadClass_3_sum', 0),
                features.get('QuadClass_4_sum', 0),
                features.get('IsHighConflict_sum', 0),
                json.dumps(features)
            ))

            # Insert in batches
            if len(rows_to_insert) >= batch_size:
                self.db.bulk_insert_features(rows_to_insert)
                print(f"   Imported {idx + 1:,} / {len(df):,} rows...")
                rows_to_insert = []

        # Insert remaining rows
        if rows_to_insert:
            self.db.bulk_insert_features(rows_to_insert)

        print(f" Imported {len(df):,} rows to database")

    def generate_daily_features(self, country: str, date: date_type) -> Dict:
        """
        Generate features for a specific country-day
        This would fetch from GDELT API in production
        For now, uses sample data
        """
        # In production, this would:
        # 1. Fetch GDELT events for this country-day from API
        # 2. Calculate aggregations (event counts, Goldstein, QuadClass distributions)
        # 3. Return as feature dict

        # Sample implementation
        features = {
            'NumEvents_sum': np.random.randint(50, 300),
            'AvgGoldstein_sum': np.random.uniform(-8, 5),
            'QuadClass_1_sum': np.random.randint(0, 50),
            'QuadClass_2_sum': np.random.randint(0, 50),
            'QuadClass_3_sum': np.random.randint(0, 80),
            'QuadClass_4_sum': np.random.randint(0, 120),
            'IsHighConflict_sum': np.random.randint(0, 50)
        }

        return features

    def update_latest_data(self, countries: Optional[List[str]] = None):
        """
        Update database with latest GDELT data
        In production, this would run daily via scheduler
        """
        if countries is None:
            # Get all countries from existing data
            stats = self.db.get_data_summary()
            print(f"Updating data for all countries...")
            # For now, just generate sample data for test countries
            countries = ['Myanmar', 'Ukraine', 'Sudan', 'China', 'Russia', 'Iran']

        today = date_type.today()
        yesterday = today - timedelta(days=1)

        print(f" Updating features for {yesterday}")

        for country in countries:
            # Generate or fetch features
            features = self.generate_daily_features(country, yesterday)

            # Insert to database
            self.db.insert_country_features(
                country=country,
                date=yesterday,
                features=features
            )

            print(f"    Updated {country}")

        print(f" Data update complete for {len(countries)} countries")


def initialize_database_from_parquet():
    """
    One-time function to populate database from existing parquet file
    Run this once to set up your database
    """
    fetcher = GDELTDataFetcher()

    # Load from parquet
    df = fetcher.load_from_parquet()

    if df is not None:
        # Process features
        df = fetcher.process_features(df)

        # Import to database
        fetcher.import_to_database(df)

        # Show stats
        stats = get_db().get_data_summary()
        print("\n Database Stats:")
        print(f"   Total features: {stats['total_features']:,}")
        print(f"   Countries: {stats['total_countries']}")
        print(f"   Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")
    else:
        print("  No parquet file found - using sample data mode")


if __name__ == "__main__":
    # Run this to populate database from parquet
    print("Initializing database from parquet file...")
    initialize_database_from_parquet()
