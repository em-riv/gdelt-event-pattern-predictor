import pandas as pd

df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
print('Total columns:', len(df.columns))
print('\nColumns in dataset:')

usa = df[df['Country']=='USA'].tail(7)
for col in sorted(df.columns):
    nan_count = usa[col].isna().sum()
    print(f'{col}: {nan_count}/7 NaNs')