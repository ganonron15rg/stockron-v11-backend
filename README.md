# Stockron Analyzer v11 — Backend (Basic)

Minimal FastAPI server exposing `/analyze` with live data via yfinance.
No OpenAI, no schedulers. Optimized for Render Free plan.

## Endpoints
- GET /healthz
- POST /analyze  { "ticker": "NVDA", "timeframe": "6mo" }

## Local run
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn ai_analyzer_server:app --reload --host 0.0.0.0 --port 8000

## Deploy on Render (via GitHub)
- New → Web Service → Connect GitHub repo with these files
- Build: pip install -r requirements.txt
- Start: uvicorn ai_analyzer_server:app --host 0.0.0.0 --port $PORT
- Open: https://<service>.onrender.com/healthz
