# Quick Push to GitHub

```bash
cd C:\Users\Emman\Documents\AI_dev\GDELT_ConflictPredictor

# Initialize repo
git init
git add .
git commit -m "GDELT Event Pattern Predictor

ML system predicting GDELT reporting patterns (98M events, 229 countries).
FastAPI backend, Angular frontend, trained GRU model (82.57% ROC-AUC).

Includes data pipeline, 4 trained models, REST API, web dashboard, and
5 Jupyter notebooks showing the analysis process.

Key lesson: GDELT predicts what gets reported, not what actually happens.
Circular validation caught by testing against real conflict data (ACLED)."

# Create repo on GitHub (https://github.com/new)
# - Name: gdelt-event-pattern-predictor
# - Public
# - Do NOT init with README

# Then push
git remote add origin https://github.com/YOUR_USERNAME/gdelt-event-pattern-predictor.git
git branch -M main
git push -u origin main
```

That's it. Your project is live.
