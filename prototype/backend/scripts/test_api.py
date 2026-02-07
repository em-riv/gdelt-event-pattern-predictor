"""
Simple backend test to verify API is working with GRU predictions
"""
import requests
import json

try:
    response = requests.get('http://localhost:8000/api/predictions/latest')
    if response.status_code == 200:
        data = response.json()
        print(f" API working! Got {len(data)} predictions")
        
        # Show top 5 predictions
        for i, pred in enumerate(data[:5]):
            print(f"{i+1}. {pred['country']}: {pred['conflict_probability']:.1%} ({pred['risk_level']})")
    else:
        print(f" API error: {response.status_code}")
        print(response.text)
except requests.exceptions.ConnectionError:
    print(" Backend server not running")
except Exception as e:
    print(f" Error: {e}")