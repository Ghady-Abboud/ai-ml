# Geopolitical Tension Forecaster

Real-time dashboard predicting conflict escalation across country pairs using machine learning on news and event data.

## Problem

Geopolitical tensions escalate unpredictably. This system forecasts escalation risk 30 days ahead across multiple severity levels, from diplomatic protests to armed conflict.

## Approach

**Data Sources:**
- GDELT Event Database (300M+ geopolitical events)
- ACLED Conflict Data (ground truth labels)
- News articles via GDELT URLs

**Model:**
- Fine-tuned DistilBERT on news headlines
- XGBoost classifier on combined text embeddings + structured event features
- Predicts escalation level: 0 (stable) → 4 (armed conflict)

**Features:**
- 7-day sliding window of event frequencies by type
- News sentiment and entity mentions
- Historical tension patterns

## Tech Stack

- **ML:** HuggingFace, scikit-learn
- **Backend:** FastAPI
- **Frontend:** React + Recharts
- **Deploy:** Docker, Render/Railway

## MVP Scope

**Country Pairs (8):**
- Russia-Ukraine
- India-Pakistan
- China-Taiwan
- Israel-Iran
- North Korea-South Korea
- Armenia-Azerbaijan
- India-China
- Turkey-Greece

**Deliverables:**
1. Trained model with >70% accuracy on held-out escalations
2. REST API serving predictions
3. Dashboard showing risk scores, trends, and recent events
4. Daily automated data pipeline

## Project Structure
```
├── data/
│   ├── raw/              # GDELT downloads
│   ├── processed/        # Labeled training data
│   └── labels.csv        # Manual escalation labels
├── models/
│   ├── train.py          # Model training pipeline
│   ├── inference.py      # Prediction logic
│   └── weights/          # Saved model checkpoints
├── api/
│   ├── main.py           # FastAPI app
│   ├── routes.py         # Endpoint definitions
│   └── db.py             # Database interactions
├── frontend/
│   ├── src/
│   └── public/
├── scripts/
│   ├── fetch_gdelt.py    # Data collection
│   └── daily_update.py   # Scheduled predictions
├── notebooks/
│   └── eda.ipynb         # Exploratory analysis
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download historical data
python scripts/fetch_gdelt.py --start 2015-01-01 --end 2024-12-01

# Train model
python models/train.py --config config.yaml

# Run API
uvicorn api.main:app --reload

# Run frontend
cd frontend && npm install && npm start
```

## API Endpoints
```
GET  /countries              # List tracked country pairs
GET  /risk/{pair}            # Current risk score + 30-day forecast
GET  /events/{pair}          # Recent events (last 7 days)
POST /predict                # Manual scenario testing
```

## Metrics

- **Precision/Recall** on escalation events (target: >0.7)
- **Early warning time** (avg days before actual escalation)
- **False positive rate** (critical for usability)

## Timeline

- **Week 1-4:** Data collection + labeling
- **Week 5-8:** Model training + evaluation
- **Week 9-10:** API development
- **Week 11-12:** Dashboard + deployment
- **Week 13-16:** Testing + refinement

## Results

_[To be filled with model performance, example predictions, attention visualizations]_

## References

- GDELT Project: https://www.gdeltproject.org/
- ACLED: https://acleddata.com/
