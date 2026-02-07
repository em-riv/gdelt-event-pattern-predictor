"""
Process raw GDELT event-level parquet files into country-day aggregations
Ensures data integrity by only using events from the correct date
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def process_daily_file(file_path: Path, expected_date: str) -> pd.DataFrame:
    """
    Process a single day's event file into country-day aggregations

    Args:
        file_path: Path to the parquet file
        expected_date: Expected date in YYYYMMDD format

    Returns:
        DataFrame with country-day aggregations
    """
    try:
        # Load events
        df = pd.read_parquet(file_path)

        # Validate date - ONLY include events from the expected date
        df = df[df['SQLDATE'] == int(expected_date)]

        if len(df) == 0:
            print(f"  WARNING: No events for {expected_date} in {file_path.name}")
            return None

        # Use Actor1CountryCode as the country
        df = df[df['Actor1CountryCode'].notna()].copy()

        if len(df) == 0:
            return None

        # Create country-day aggregations matching the expected format
        country_day = df.groupby('Actor1CountryCode').agg({
            'GLOBALEVENTID': 'count',  # EventCount

            # Cooperation features
            'IsCooperation': ['sum', 'mean'],
            'IsVerbalCoopEvent': ['sum', 'mean'],
            'IsMaterialCoopEvent': ['sum', 'mean'],

            # Conflict features
            'IsConflict': ['sum', 'mean'],
            'IsVerbalConflictEvent': ['sum', 'mean'],
            'IsMaterialConflictEvent': ['sum', 'mean'],

            # Goldstein scale
            'GoldsteinScale': ['mean', 'std', 'min', 'max'],

            # Media attention
            'NumMentions': ['sum', 'mean', 'std'],
            'NumSources': ['sum', 'mean', 'std'],
            'NumArticles': ['sum', 'mean', 'std'],
            'AvgTone': ['mean', 'std', 'min', 'max'],

            # Quad classes
            'IsQuad1_VerbalCoop': ['sum', 'mean'],
            'IsQuad2_MaterialCoop': ['sum', 'mean'],
            'IsQuad3_VerbalConflict': ['sum', 'mean'],
            'IsQuad4_MaterialConflict': ['sum', 'mean'],

            # Actor types
            'IsStateVsNonState': ['sum', 'mean'],
            'Actor1_IsState': 'mean',
            'Actor1_IsNonState': 'mean'
        }).reset_index()

        # Add QuadClass sums separately
        quad_class_counts = df.groupby('Actor1CountryCode')['QuadClass'].apply(
            lambda x: pd.Series({
                'QuadClass_1_sum': (x == 1).sum(),
                'QuadClass_2_sum': (x == 2).sum(),
                'QuadClass_3_sum': (x == 3).sum(),
                'QuadClass_4_sum': (x == 4).sum(),
            })
        ).reset_index()

        # Merge quad class counts
        country_day = country_day.merge(quad_class_counts, left_on='Actor1CountryCode', right_on='Actor1CountryCode', how='left')

        # Flatten column names
        country_day.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                               for col in country_day.columns.values]

        # Rename to match expected format
        rename_map = {
            'Actor1CountryCode': 'Country',
            'GLOBALEVENTID_count': 'EventCount',
            'IsCooperation_sum': 'IsCooperation_sum',
            'IsCooperation_mean': 'IsCooperation_mean',
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
        }

        country_day.rename(columns=rename_map, inplace=True)

        # Add date
        date_obj = datetime.strptime(expected_date, '%Y%m%d')
        country_day['Date'] = date_obj

        # Calculate intensity score (combination of Goldstein and conflict rate)
        if 'GoldsteinScale_mean' in country_day.columns and 'IsHighConflict_mean' in country_day.columns:
            country_day['IntensityScore_mean'] = (
                -country_day['GoldsteinScale_mean'] * country_day['IsHighConflict_mean']
            )

        return country_day

    except Exception as e:
        print(f"  ERROR processing {file_path.name}: {e}")
        return None


def process_all_files(start_year=2023, end_year=2025):
    """
    Process all daily files from start_year to end_year
    """
    base_path = Path(__file__).parent.parent.parent / "archive_2024_12_12" / "features" / "event_level"

    all_data = []
    total_files = 0
    processed_files = 0

    print(f"Processing GDELT data from {start_year} to {end_year}...")
    print("=" * 80)

    for year in range(start_year, end_year + 1):
        year_path = base_path / str(year)

        if not year_path.exists():
            print(f"Year {year} not found, skipping...")
            continue

        print(f"\nProcessing {year}...")

        for month in range(1, 13):
            month_path = year_path / f"{month:02d}"

            if not month_path.exists():
                continue

            # Get all parquet files in this month
            parquet_files = sorted(month_path.glob("*.parquet"))

            for file_path in parquet_files:
                total_files += 1

                # Extract expected date from filename (YYYYMMDD.parquet)
                expected_date = file_path.stem

                # Process this day's data
                country_day = process_daily_file(file_path, expected_date)

                if country_day is not None and len(country_day) > 0:
                    all_data.append(country_day)
                    processed_files += 1

                    if processed_files % 50 == 0:
                        print(f"  Processed {processed_files}/{total_files} files...")

    print(f"\n{'=' * 80}")
    print(f"Processing complete!")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed_files}")

    if not all_data:
        raise ValueError("No data was processed!")

    # Combine all data
    print("\nCombining all country-day aggregations...")
    final_df = pd.concat(all_data, ignore_index=True)

    # Sort by country and date
    final_df = final_df.sort_values(['Country', 'Date']).reset_index(drop=True)

    print(f"\nFinal dataset:")
    print(f"  Rows: {len(final_df):,}")
    print(f"  Countries: {final_df['Country'].nunique()}")
    print(f"  Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")
    print(f"  Columns: {len(final_df.columns)}")

    return final_df


def save_output(df: pd.DataFrame, output_path: str):
    """Save processed data to parquet"""
    print(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, index=False, compression='snappy')
    print(f"Saved successfully!")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    print("GDELT Raw Data Processor")
    print("=" * 80)
    print("This will process event-level data (2023-2025) into country-day features")
    print("Ensuring data integrity by only using events from the correct date\n")

    # Process all data
    df = process_all_files(start_year=2023, end_year=2025)

    # Save output
    output_path = "../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet"
    save_output(df, output_path)

    print("\n" + "=" * 80)
    print("DONE! Ready to import to database.")
    print("=" * 80)
