import sqlite3

conn = sqlite3.connect('gdelt_predictor.db')
cursor = conn.cursor()

countries = ['UKR', 'ISR', 'PSE', 'RUS', 'SYR', 'YEM', 'AFG', 'MMR', 'ETH', 'SDN', 'IRN']
print('Conflict predictions for known conflict zones:\n')

cursor.execute(f"SELECT country, conflict_probability, risk_level, horizon_days FROM predictions WHERE country IN ({','.join(['?']*len(countries))}) ORDER BY country, horizon_days", countries)
results = cursor.fetchall()

current_country = None
for row in results:
    if current_country != row[0]:
        print(f'\n{row[0]}:')
        current_country = row[0]
    print(f'  {row[3]}-day horizon: {row[1]*100:.2f}% ({row[2]})')

print('\n\nTop 15 highest predictions (7-day horizon):')
cursor.execute("SELECT country, conflict_probability, risk_level FROM predictions WHERE horizon_days = 7 ORDER BY conflict_probability DESC LIMIT 15")
for row in cursor.fetchall():
    print(f'{row[0]}: {row[1]*100:.2f}% ({row[2]})')

conn.close()
