"""
GDELT Live Data Fetcher
Downloads and processes live GDELT data to keep database up-to-date
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
import requests
import zipfile
import io
import json
from typing import List, Optional, Dict
from database import get_db


class GDELTLiveFetcher:
    """Fetches live GDELT data and processes it into database format"""

    # GDELT 2.0 Event Database URL pattern
    GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2/"

    def __init__(self):
        self.db = get_db()
        self.cache_dir = Path(__file__).parent / "gdelt_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def get_missing_dates(self) -> List[date_type]:
        """Get list of dates missing from database up to today"""
        stats = self.db.get_data_summary()

        if stats['total_features'] == 0:
            print("  Database is empty")
            return []

        # Get latest date in database
        latest_date = datetime.strptime(stats['date_range']['max'], '%Y-%m-%d').date()
        today = date_type.today()

        # Generate list of missing dates
        missing_dates = []
        current_date = latest_date + timedelta(days=1)

        while current_date <= today:
            missing_dates.append(current_date)
            current_date += timedelta(days=1)

        return missing_dates

    def fetch_gdelt_day(self, target_date: date_type) -> Optional[pd.DataFrame]:
        """
        Fetch GDELT events for a specific day
        GDELT 2.0 publishes data in 15-minute intervals
        """
        print(f" Fetching GDELT data for {target_date}...")

        # GDELT 2.0 file naming: YYYYMMDDHHMMSS.export.CSV.zip
        # We need to fetch all 15-min intervals for the day (96 files)
        date_str = target_date.strftime('%Y%m%d')

        all_events = []

        # Try to fetch data for the full day (every 15 minutes)
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                timestamp = f"{date_str}{hour:02d}{minute:02d}00"
                url = f"{self.GDELT_BASE_URL}{timestamp}.export.CSV.zip"

                try:
                    # Download the zip file
                    response = requests.get(url, timeout=30)

                    if response.status_code == 200:
                        # Extract CSV from zip
                        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                            csv_filename = f"{timestamp}.export.CSV"
                            with z.open(csv_filename) as f:
                                # GDELT 2.0 Event format (no header)
                                df = pd.read_csv(f, sep='\t', header=None, low_memory=False)
                                all_events.append(df)
                                print(f"    {timestamp}: {len(df)} events")
                    else:
                        # File might not exist yet (future date or data not published)
                        continue

                except Exception as e:
                    # Skip missing intervals
                    continue

        if not all_events:
            print(f"     No data found for {target_date}")
            return None

        # Combine all intervals
        df_day = pd.concat(all_events, ignore_index=True)
        print(f"    Total events for {target_date}: {len(df_day):,}")

        # Add column names (GDELT 2.0 format)
        df_day.columns = self._get_gdelt_column_names()

        return df_day

    def _get_gdelt_column_names(self) -> List[str]:
        """GDELT 2.0 Event Database column names"""
        return [
            'GLOBALEVENTID', 'SQLDATE', 'MonthYear', 'Year', 'FractionDate',
            'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
            'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
            'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
            'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
            'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
            'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
            'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
            'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
            'NumArticles', 'AvgTone', 'Actor1Geo_Type', 'Actor1Geo_FullName',
            'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code',
            'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID',
            'Actor2Geo_Type', 'Actor2Geo_FullName', 'Actor2Geo_CountryCode',
            'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code', 'Actor2Geo_Lat',
            'Actor2Geo_Long', 'Actor2Geo_FeatureID', 'ActionGeo_Type',
            'ActionGeo_FullName', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code',
            'ActionGeo_ADM2Code', 'ActionGeo_Lat', 'ActionGeo_Long',
            'ActionGeo_FeatureID', 'DATEADDED', 'SOURCEURL'
        ]

    def process_gdelt_to_features(self, df: pd.DataFrame, target_date: date_type) -> pd.DataFrame:
        """
        Process raw GDELT events into country-day aggregated features
        Matches the format used in our database
        """
        print(f" Processing events into country-day features...")

        # Filter to correct date
        date_int = int(target_date.strftime('%Y%m%d'))
        df = df[df['SQLDATE'] == date_int].copy()

        if len(df) == 0:
            print(f"     No events for date {target_date}")
            return pd.DataFrame()

        # Filter to valid country codes only (3-letter ISO codes, not empty/null)
        df = df[df['Actor1CountryCode'].notna()].copy()
        df = df[df['Actor1CountryCode'].str.len() == 3].copy()
        df = df[df['Actor1CountryCode'] != ''].copy()

        if len(df) == 0:
            print(f"     No valid country codes for date {target_date}")
            return pd.DataFrame()

        # Create derived features
        df['IsCooperation'] = (df['QuadClass'].isin([1, 2])).astype(int)
        df['IsVerbalConflict'] = (df['QuadClass'] == 3).astype(int)
        df['IsHighConflict'] = (df['QuadClass'] == 4).astype(int)

        # Aggregate by country (Actor1CountryCode)
        country_features = df.groupby('Actor1CountryCode').agg({
            'GLOBALEVENTID': 'count',  # EventCount
            'IsCooperation': ['sum', 'mean'],
            'IsVerbalConflict': ['sum', 'mean'],
            'IsHighConflict': ['sum', 'mean'],
            'GoldsteinScale': ['mean', 'std', 'min', 'max'],
            'NumMentions': ['sum', 'mean', 'std'],
            'NumSources': ['sum', 'mean', 'std'],
            'NumArticles': ['sum', 'mean', 'std'],
            'AvgTone': ['mean', 'std', 'min', 'max'],
        }).reset_index()

        # Flatten column names
        country_features.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                    for col in country_features.columns.values]

        # Add QuadClass counts separately (lambda aggregation issue workaround)
        quad_counts = df.groupby('Actor1CountryCode')['QuadClass'].apply(lambda x: pd.Series({
            'QuadClass_1_sum': (x == 1).sum(),
            'QuadClass_2_sum': (x == 2).sum(),
            'QuadClass_3_sum': (x == 3).sum(),
            'QuadClass_4_sum': (x == 4).sum(),
        })).reset_index()

        # Merge QuadClass counts
        country_features = country_features.merge(quad_counts, on='Actor1CountryCode', how='left')

        # Rename to match database format
        rename_map = {
            'Actor1CountryCode': 'Country',
            'GLOBALEVENTID_count': 'EventCount',
            'IsCooperation_sum': 'IsCooperation_sum',
            'IsCooperation_mean': 'IsCooperation_mean',
            'IsVerbalConflict_sum': 'IsVerbalConflict_sum',
            'IsVerbalConflict_mean': 'IsVerbalConflict_mean',
            'IsHighConflict_sum': 'IsHighConflict_sum',
            'IsHighConflict_mean': 'IsHighConflict_mean',
            'GoldsteinScale_mean': 'GoldsteinScale_mean',
            'GoldsteinScale_std': 'GoldsteinScale_std',
            'GoldsteinScale_min': 'GoldsteinScale_min',
            'GoldsteinScale_max': 'GoldsteinScale_max',
        }

        country_features = country_features.rename(columns=rename_map)

        # Add date
        country_features['Date'] = pd.to_datetime(target_date)

        print(f"    Generated features for {len(country_features)} countries")

        return country_features

    def update_database_to_today(self):
        """Main function: fetch missing dates and update database"""
        print("=" * 60)
        print("GDELT Live Data Update")
        print("=" * 60)

        # Get missing dates
        missing_dates = self.get_missing_dates()

        if not missing_dates:
            print(" Database is already up to date!")
            return

        print(f" Found {len(missing_dates)} missing date(s)")
        print(f"   Range: {missing_dates[0]} to {missing_dates[-1]}")

        # Fetch and process each date
        for target_date in missing_dates:
            print(f"\n--- Processing {target_date} ---")

            # Fetch raw GDELT data
            df_raw = self.fetch_gdelt_day(target_date)

            if df_raw is None or len(df_raw) == 0:
                print(f"     Skipping {target_date} (no data)")
                continue

            # Process into features
            df_features = self.process_gdelt_to_features(df_raw, target_date)

            if len(df_features) == 0:
                continue

            # Insert into database
            print(f"    Inserting into database...")
            for idx in range(len(df_features)):
                row = df_features.iloc[idx]

                # Prepare features dict - only numeric columns
                features = {}
                for col in df_features.columns:
                    if col not in ['Country', 'Date']:
                        val = row[col]
                        # Skip if value is not numeric
                        try:
                            if pd.notna(val):
                                features[col] = float(val)
                            else:
                                features[col] = 0.0
                        except (ValueError, TypeError):
                            # Skip non-numeric columns
                            continue

                # Ensure required fields exist
                features['NumEvents_sum'] = features.get('EventCount', 0)
                features['AvgGoldstein_sum'] = features.get('GoldsteinScale_mean', 0)
                features['QuadClass_1_sum'] = features.get('QuadClass_1_sum', 0)
                features['QuadClass_2_sum'] = features.get('QuadClass_2_sum', 0)
                features['QuadClass_3_sum'] = features.get('QuadClass_3_sum', 0)
                features['QuadClass_4_sum'] = features.get('QuadClass_4_sum', 0)
                features['IsHighConflict_sum'] = features.get('IsHighConflict_sum', 0)

                # Insert to database
                self.db.insert_country_features(
                    country=row['Country'],
                    date=target_date,
                    features=features
                )

            print(f"    Inserted {len(df_features)} countries for {target_date}")

        print("\n" + "=" * 60)
        print(" Database update complete!")
        print("=" * 60)

        # Show updated stats
        stats = self.db.get_data_summary()
        print(f"\n Updated Database Stats:")
        print(f"   Total features: {stats['total_features']:,}")
        print(f"   Countries: {stats['total_countries']}")
        print(f"   Date range: {stats['date_range']['min']} to {stats['date_range']['max']}")


if __name__ == "__main__":
    fetcher = GDELTLiveFetcher()
    fetcher.update_database_to_today()
