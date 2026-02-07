import pandas as pd

df = pd.read_parquet('../../data/features_multiresolution/country_day/country_day_features_2023_2025.parquet')
recent = df[df['Date'] >= '2025-11-01']

print("Recent conflict intensity (Nov-Dec 2025):\n")
for country in ['UKR', 'ISR', 'PSE', 'RUS', 'SYR', 'YEM', 'USA']:
    c_data = recent[recent['Country'] == country]
    if len(c_data) > 0:
        avg = c_data['IsHighConflict_mean'].mean()
        max_val = c_data['IsHighConflict_mean'].max()
        target = (c_data['IsHighConflict_mean'] > 0.20).mean()
        print(f'{country}:')
        print(f'  Avg IsHighConflict_mean: {avg*100:.2f}%')
        print(f'  Max IsHighConflict_mean: {max_val*100:.2f}%')
        print(f'  Days >20% threshold: {target*100:.1f}%\n')
