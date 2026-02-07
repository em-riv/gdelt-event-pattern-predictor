"""
Robust GDELT Conflict Predictor API with News and Predictions
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from typing import List, Dict, Optional
import uvicorn
import json
from datetime import datetime, timedelta
import random

app = FastAPI(
    title="GDELT Conflict Predictor - Enhanced Edition", 
    version="3.0.0",
    description="Conflict predictions with global news analysis"
)

# Robust CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Sample news data (in production, this would come from GDELT news API)
POSITIVE_NEWS_SAMPLES = [
    {
        "id": 1,
        "title": "Global Peace Index Shows Significant Improvement in 15 Countries",
        "summary": "International cooperation and diplomatic efforts lead to reduced tensions across multiple regions.",
        "country": "Global",
        "sentiment_score": 0.85,
        "date": "2025-12-15",
        "source": "Peace Research Institute",
        "category": "Diplomacy",
        "url": "https://example.com/peace-index"
    },
    {
        "id": 2,
        "title": "Historic Trade Agreement Signed Between Former Adversaries",
        "summary": "Economic partnership expected to create 50,000 jobs and strengthen regional stability.",
        "country": "Multiple",
        "sentiment_score": 0.78,
        "date": "2025-12-14",
        "source": "Economic Times",
        "category": "Economics",
        "url": "https://example.com/trade-agreement"
    },
    {
        "id": 3,
        "title": "Humanitarian Aid Reaches 2 Million People in Crisis Zones",
        "summary": "International relief efforts successfully deliver food, medicine, and shelter to affected populations.",
        "country": "Multiple",
        "sentiment_score": 0.72,
        "date": "2025-12-13",
        "source": "UN News",
        "category": "Humanitarian",
        "url": "https://example.com/aid-delivery"
    },
    {
        "id": 4,
        "title": "Climate Peace Initiative Prevents Resource-Based Conflicts",
        "summary": "Innovative water-sharing agreements resolve disputes between neighboring communities.",
        "country": "East Africa",
        "sentiment_score": 0.80,
        "date": "2025-12-12",
        "source": "Environmental Monitor",
        "category": "Environment",
        "url": "https://example.com/climate-peace"
    },
    {
        "id": 5,
        "title": "Youth Peace Corps Mediates Local Disputes Successfully",
        "summary": "Trained young mediators resolve 200+ community conflicts through dialogue and understanding.",
        "country": "South America",
        "sentiment_score": 0.76,
        "date": "2025-12-11",
        "source": "Youth International",
        "category": "Social",
        "url": "https://example.com/youth-peace"
    },
    {
        "id": 6,
        "title": "Cross-Border Educational Exchange Program Builds Bridges",
        "summary": "Students from 20 countries participate in conflict resolution and cultural understanding programs.",
        "country": "Europe",
        "sentiment_score": 0.74,
        "date": "2025-12-10",
        "source": "Education Today",
        "category": "Education",
        "url": "https://example.com/education-exchange"
    },
    {
        "id": 7,
        "title": "Technology Helps Monitor and Prevent Ethnic Tensions",
        "summary": "AI-powered early warning systems successfully predict and prevent communal violence.",
        "country": "Asia",
        "sentiment_score": 0.77,
        "date": "2025-12-09",
        "source": "Tech for Good",
        "category": "Technology",
        "url": "https://example.com/ai-peace"
    },
    {
        "id": 8,
        "title": "Interfaith Leaders Unite for Regional Stability",
        "summary": "Religious cooperation initiative promotes tolerance and reduces sectarian tensions.",
        "country": "Middle East",
        "sentiment_score": 0.73,
        "date": "2025-12-08",
        "source": "Faith & Peace",
        "category": "Religion",
        "url": "https://example.com/interfaith-unity"
    },
    {
        "id": 9,
        "title": "Economic Development Reduces Migration Pressures",
        "summary": "Job creation programs in conflict-prone areas provide alternatives to displacement.",
        "country": "Central America",
        "sentiment_score": 0.79,
        "date": "2025-12-07",
        "source": "Development Bank",
        "category": "Economics",
        "url": "https://example.com/economic-development"
    },
    {
        "id": 10,
        "title": "Women Peacebuilders Lead Reconciliation Efforts",
        "summary": "Female leaders broker peace agreements in three conflict-affected communities.",
        "country": "Africa",
        "sentiment_score": 0.81,
        "date": "2025-12-06",
        "source": "Women's Peace Network",
        "category": "Social",
        "url": "https://example.com/women-peace"
    }
]

NEGATIVE_NEWS_SAMPLES = [
    {
        "id": 11,
        "title": "Border Tensions Escalate Despite Diplomatic Efforts",
        "summary": "Military buildup continues as negotiations fail to resolve territorial disputes.",
        "country": "Multiple",
        "sentiment_score": -0.82,
        "date": "2025-12-15",
        "source": "Security Monitor",
        "category": "Military",
        "url": "https://example.com/border-tensions"
    },
    {
        "id": 12,
        "title": "Ethnic Violence Displaces Thousands in Remote Region",
        "summary": "Communal clashes force families to flee homes as security forces struggle to respond.",
        "country": "Myanmar",
        "sentiment_score": -0.88,
        "date": "2025-12-14",
        "source": "Crisis Report",
        "category": "Civil Unrest",
        "url": "https://example.com/ethnic-violence"
    },
    {
        "id": 13,
        "title": "Resource Scarcity Fuels Community Conflicts",
        "summary": "Water shortages lead to violent confrontations between farming and herding communities.",
        "country": "Chad",
        "sentiment_score": -0.75,
        "date": "2025-12-13",
        "source": "Conflict Tracker",
        "category": "Resources",
        "url": "https://example.com/resource-conflict"
    },
    {
        "id": 14,
        "title": "Political Opposition Faces Violent Crackdown",
        "summary": "Protesters arrested and injured as government responds with force to demonstrations.",
        "country": "Belarus",
        "sentiment_score": -0.84,
        "date": "2025-12-12",
        "source": "Human Rights Watch",
        "category": "Politics",
        "url": "https://example.com/political-violence"
    },
    {
        "id": 15,
        "title": "Arms Trafficking Network Destabilizes Multiple Countries",
        "summary": "Illegal weapons flow increases violence in already unstable regions.",
        "country": "Sahel Region",
        "sentiment_score": -0.79,
        "date": "2025-12-11",
        "source": "Security Analysis",
        "category": "Security",
        "url": "https://example.com/arms-trafficking"
    },
    {
        "id": 16,
        "title": "Refugee Crisis Overwhelms Border Communities",
        "summary": "Massive population displacement strains resources and creates social tensions.",
        "country": "Jordan",
        "sentiment_score": -0.77,
        "date": "2025-12-10",
        "source": "UNHCR",
        "category": "Migration",
        "url": "https://example.com/refugee-crisis"
    },
    {
        "id": 17,
        "title": "Cyber Attacks Target Critical Infrastructure",
        "summary": "Digital warfare tactics disrupt power grids and communication networks.",
        "country": "Ukraine",
        "sentiment_score": -0.73,
        "date": "2025-12-09",
        "source": "Cyber Security News",
        "category": "Technology",
        "url": "https://example.com/cyber-attacks"
    },
    {
        "id": 18,
        "title": "Drug Cartels Expand Territory Through Violence",
        "summary": "Organized crime groups engage in deadly turf wars affecting civilian populations.",
        "country": "Mexico",
        "sentiment_score": -0.86,
        "date": "2025-12-08",
        "source": "Crime Monitor",
        "category": "Crime",
        "url": "https://example.com/cartel-violence"
    },
    {
        "id": 19,
        "title": "Food Insecurity Triggers Social Unrest",
        "summary": "Rising prices and supply shortages lead to protests and civil disorder.",
        "country": "Sri Lanka",
        "sentiment_score": -0.78,
        "date": "2025-12-07",
        "source": "Economic Observer",
        "category": "Economics",
        "url": "https://example.com/food-crisis"
    },
    {
        "id": 20,
        "title": "Religious Extremists Target Minority Communities",
        "summary": "Hate crimes and targeted violence increase against religious minorities.",
        "country": "Pakistan",
        "sentiment_score": -0.85,
        "date": "2025-12-06",
        "source": "Minority Rights Group",
        "category": "Religion",
        "url": "https://example.com/religious-violence"
    }
]

def get_db_connection():
    """Get database connection with error handling"""
    try:
        conn = sqlite3.connect('conflict_predictor.db')
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "GDELT Conflict Predictor",
        "version": "3.0.0",
        "features": ["conflict_predictions", "global_news", "risk_analysis"],
        "endpoints": {
            "predictions": "/api/predictions/latest",
            "news": "/api/news/positive and /api/news/negative", 
            "dashboard": "/api/dashboard/overview"
        }
    }

@app.get("/api/predictions/latest")
def get_latest_predictions():
    """Get latest conflict predictions"""
    conn = get_db_connection()
    if not conn:
        # Return sample data if DB unavailable
        return [
            {
                "country": "AFG",
                "date": "2025-12-16",
                "prediction_date": "2025-12-15",
                "conflict_probability": 0.968,
                "risk_level": "CRITICAL",
                "confidence": 0.89
            },
            {
                "country": "SDN", 
                "date": "2025-12-16",
                "prediction_date": "2025-12-15",
                "conflict_probability": 0.964,
                "risk_level": "CRITICAL",
                "confidence": 0.87
            },
            {
                "country": "PSE",
                "date": "2025-12-16", 
                "prediction_date": "2025-12-15",
                "conflict_probability": 0.957,
                "risk_level": "CRITICAL",
                "confidence": 0.85
            }
        ]
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT country, prediction_date, target_date, horizon_days,
                   conflict_probability, risk_level, confidence, model_version
            FROM predictions 
            ORDER BY conflict_probability DESC
            LIMIT 50
        """)
        
        rows = cursor.fetchall()
        predictions = []
        
        for row in rows:
            predictions.append({
                "country": row[0],
                "date": row[2],  # target_date
                "prediction_date": row[1],
                "horizon_days": row[3],
                "conflict_probability": float(row[4]),
                "risk_level": row[5],
                "confidence": float(row[6])
            })
        
        conn.close()
        return predictions
        
    except Exception as e:
        print(f"Database query error: {e}")
        conn.close()
        return []

@app.get("/api/news/positive")
def get_positive_news():
    """Get top positive news stories"""
    return sorted(POSITIVE_NEWS_SAMPLES, key=lambda x: x["sentiment_score"], reverse=True)

@app.get("/api/news/negative") 
def get_negative_news():
    """Get top negative/conflict news stories"""
    return sorted(NEGATIVE_NEWS_SAMPLES, key=lambda x: x["sentiment_score"])

@app.get("/api/dashboard/overview")
def get_dashboard_overview():
    """Get comprehensive dashboard data"""
    predictions = get_latest_predictions()
    positive_news = get_positive_news()[:10]
    negative_news = get_negative_news()[:10]
    
    # Calculate summary statistics
    if predictions:
        high_risk_countries = [p for p in predictions if p["conflict_probability"] > 0.8]
        avg_risk = sum(p["conflict_probability"] for p in predictions) / len(predictions)
    else:
        high_risk_countries = []
        avg_risk = 0
    
    return {
        "summary": {
            "total_countries_monitored": len(predictions) if predictions else 223,
            "high_risk_countries": len(high_risk_countries),
            "average_risk_level": round(avg_risk, 3),
            "last_updated": "2025-12-15T18:00:00Z"
        },
        "predictions": predictions[:20],  # Top 20 highest risk
        "positive_news": positive_news,
        "negative_news": negative_news,
        "trends": {
            "risk_direction": "stable",
            "top_risk_factors": ["political_instability", "economic_stress", "social_unrest"]
        }
    }

@app.get("/api/predictions/country/{country}")
def get_country_prediction(country: str):
    """Get prediction for specific country"""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database unavailable", "country": country}
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT country, prediction_date, target_date, horizon_days,
                   conflict_probability, risk_level, confidence, model_version
            FROM predictions 
            WHERE country = ?
            ORDER BY horizon_days
        """, (country.upper(),))
        
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return {"error": "Country not found", "country": country}
        
        predictions = []
        for row in rows:
            predictions.append({
                "country": row[0],
                "date": row[2],
                "prediction_date": row[1], 
                "horizon_days": row[3],
                "conflict_probability": float(row[4]),
                "risk_level": row[5],
                "confidence": float(row[6])
            })
        
        conn.close()
        return {
            "country": country,
            "current_prediction": predictions[0] if predictions else None,
            "predictions": predictions,
            "trend_7day": [p["conflict_probability"] for p in predictions[:3]],
            "top_features": [
                {"feature_name": "GRU_Neural_Network", "contribution": 1.0}
            ],
            "model_info": {
                "name": "GRU",
                "accuracy": 0.74,
                "last_trained": "2025-12-15"
            }
        }
        
    except Exception as e:
        conn.close()
        return {"error": str(e), "country": country}

if __name__ == "__main__":
    print("🚀 Starting Enhanced GDELT Conflict Predictor API...")
    print("📊 Features: Predictions + Global News Analysis")
    print("🔗 Frontend: http://localhost:4200")
    print("🔗 API Docs: http://127.0.0.1:8080/docs")
    print("🔗 Dashboard: http://127.0.0.1:8080/api/dashboard/overview")
    
    try:
        uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
    except KeyboardInterrupt:
        print("🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")