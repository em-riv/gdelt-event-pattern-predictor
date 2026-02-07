"""
Add 2025 GDELT data to existing 2023-2024 dataset
Processes raw event files and creates same format as existing parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def aggregate_day_file(file_path: Path, expected_date: str) -> pd.DataFrame:
    """
    Aggregate a single day's events to country-day level
    Matches the format of the existing country_day_features.parquet
    """
    try:
        # Load events
        df = pd.read_parquet(file_path)

        # Validate date - ONLY use events from the correct date
        df = df[df['SQLDATE'] == int(expected_date)]

        if len(df) == 0:
            return None

        # Filter for valid countries
        df = df[df['Actor1CountryCode'].notna()].copy()

        if len(df) == 0:
            return None

        # Simple aggregation matching existing format
        agg_dict = {
            'GLOBALEVENTID': 'count',
            'IsCooperation': ['sum', 'mean'],
            'IsVerbalCoopEvent': ['sum', 'mean'],
            'IsMaterialCoopEvent': ['sum', 'mean'],
            'IsConflict': ['sum', 'mean'],
            'IsVerbalConflictEvent': ['sum', 'mean'],
            'IsMaterialConflictEvent': ['sum', 'mean'],
            'GoldsteinScale': ['mean', 'std', 'min', 'max'],
            'NumMentions': ['sum', 'mean', 'std'],
            'NumSources': ['sum', 'mean', 'std'],
            'NumArticles': ['sum', 'mean', 'std'],
            'AvgTone': ['mean', 'std', 'min', 'max'],
            'IsQuad1_VerbalCoop': ['sum', 'mean'],
            'IsQuad2_MaterialCoop': ['sum', 'mean'],
            'IsQuad3_VerbalConflict': ['sum', 'mean'],
            'IsQuad4_MaterialConflict': ['sum', 'mean'],
        }

        # Aggregate by country
        country_day = df.groupby('Actor1CountryCode').agg(agg_dict).reset_index()

        # Flatten multi-level columns
        new_cols = []
        for col in country_day.columns:
            if isinstance(col, tuple):
                if col[1]:
                    new_cols.append(f"{col[0]}_{col[1]}")
                else:
                    new_cols.append(col[0])
            else:
                new_cols.append(col)

        country_day.columns = new_cols

        # Rename to match existing format
        country_day.rename(columns={
            'Actor1CountryCode': 'Country',
            'GLOBALEVENTID_count': 'EventCount',
            'IsVerbalCoopEvent_sum': 'IsVerbalCoop_sum',
            'IsVerbalCoopEvent_mean': 'IsVerbalCoop_mean',
            'IsMaterialCoopEvent_sum': 'IsMaterialCoop_sum',
            'IsMaterialCoopEvent_mean': 'IsMaterialCoop_mean',
            'IsConflict_sum': 'IsHighConflict_sum',
            'IsConflict_mean': 'IsHighConflict_mean',
            'IsVerbalConflictEvent_sum': 'IsVerbalConflict_sum',
            'IsVerbalConflictEvent_mean': 'IsVerbalConflict_mean',
            'IsMaterialConflictEvent_sum': 'IsMaterialConflict_sum',
            'IsMaterialConflictEvent_mean': 'IsMaterialConflict_mean',
        }, inplace=True)

        # Add date
        country_day['Date'] = datetime.strptime(expected_date, '%Y%m%d')

        # Calculate intensity score
        if 'GoldsteinScale_mean' in country_day.columns and 'IsHighConflict_mean' in country_day.columns:
            country_day['IntensityScore_mean'] = (
                -country_day['GoldsteinScale_mean'] * country_day['IsHighConflict_mean']
            )

        return country_day

    except Exception as e:
        print(f"  ERROR: {file_path.name}: {e}")
        return None


def process_2025_data():
    """Process all 2025 data from raw events"""

    base_path = Path(__file__).parent.parent.parent / "archive_2024_12_12" / "features" / "event_level" / "2025"

    print("Processing 2025 GDELT Data...")
    print("=" * 80)

    all_data = []
    processed = 0

    for month in range(1, 13):
        month_path = base_path / f"{month:02d}"

        if not month_path.exists():
            continue

        parquet_files = sorted(month_path.glob("*.parquet"))

        for file_path in parquet_files:
            expected_date = file_path.stem

            # Only process up to today (2025-12-12)
            if int(expected_date) > 20251212:
                break

            country_day = aggregate_day_file(file_path, expected_date)

            if country_day is not None:
                all_data.append(country_day)
                processed += 1

                if processed % 30 == 0:
                    print(f"  Processed {processed} days...")

    print(f"\nProcessed {processed} days of 2025 data")

    if not all_data:
        raise ValueError("No 2025 data was processed!")

    # Combine all 2025 data
    df_2025 = pd.concat(all_data, ignore_index=True)
    df_2025 = df_2025.sort_values(['Country', 'Date']).reset_index(drop=True)

    print(f"\n2025 Data:")
    print(f"  Rows: {len(df_2025):,}")
    print(f"  Countries: {df_2025['Country'].nunique()}")
    print(f"  Date range: {df_2025['Date'].min().date()} to {df_2025['Date'].max().date()}")

    return df_2025


def merge_with_existing(df_2025):
    """Merge 2025 data with existing 2023-2024 data"""

    print("\nLoading existing 2023-2024 data...")
    existing_path = Path(__file__).parent.parent.parent / "data" / "features_multiresolution" / "country_day" / "country_day_features.parquet"

    df_existing = pd.read_parquet(existing_path)

    print(f"  Existing data: {len(df_existing):,} rows from {df_existing['Date'].min().date()} to {df_existing['Date'].max().date()}")

    # Ensure column alignment
    missing_in_2025 = set(df_existing.columns) - set(df_2025.columns)
    missing_in_existing = set(df_2025.columns) - set(df_existing.columns)

    if missing_in_2025:
        print(f"\n  Adding missing columns to 2025 data: {missing_in_2025}")
        for col in missing_in_2025:
            df_2025[col] = np.nan

    if missing_in_existing:
        print(f"\n  Adding missing columns to existing data: {missing_in_existing}")
        for col in missing_in_existing:
            df_existing[col] = np.nan

    # Align column order
    df_2025 = df_2025[df_existing.columns]

    # Convert to CSV and back to reset dtypes - workaround for pandas datetime concat bug
    print("\nMerging datasets...")
    temp_csv = "temp_2025.csv"
    df_2025.to_csv(temp_csv, index=False)
    df_2025_clean = pd.read_csv(temp_csv)
    df_2025_clean['Date'] = pd.to_datetime(df_2025_clean['Date'])

    # Also reload existing to ensure same dtypes
    temp_existing = "temp_existing.csv"
    df_existing.to_csv(temp_existing, index=False)
    df_existing_clean = pd.read_csv(temp_existing)
    df_existing_clean['Date'] = pd.to_datetime(df_existing_clean['Date'])

    df_combined = pd.concat([df_existing_clean, df_2025_clean], ignore_index=True, sort=False)
    df_combined = df_combined.sort_values(['Country', 'Date']).reset_index(drop=True)

    print(f"\nCombined Dataset:")
    print(f"  Total rows: {len(df_combined):,}")
    print(f"  Countries: {df_combined['Country'].nunique()}")
    print(f"  Date range: {df_combined['Date'].min().date()} to {df_combined['Date'].max().date()}")
    print(f"  Columns: {len(df_combined.columns)}")

    return df_combined


if __name__ == "__main__":
    print("2025 GDELT Data Processor")
    print("=" * 80)
    print("Adding 2025 data to existing 2023-2024 dataset\n")

    # Process 2025
    df_2025 = process_2025_data()

    # Merge with existing
    df_combined = merge_with_existing(df_2025)

    # Save combined dataset
    output_path = "../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet"
    print(f"\nSaving to {output_path}...")
    df_combined.to_parquet(output_path, index=False, compression='snappy')

    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Saved! File size: {file_size:.2f} MB")

    print("\n" + "=" * 80)
    print("COMPLETE! Real GDELT data from 2023-01-01 to 2025-12-12")
    print("=" * 80)
