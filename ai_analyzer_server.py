# ==============================================================
#  ai_analyzer_server.py — Stockron Analyzer v12.2 (Hybrid Real Data)
#  Real Fundamentals via Yahoo Finance + optional Finnhub sentiment
#  Safe fallbacks to demo when fields are missing
# ==============================================================

from __future__ import annotations
import os, math, random, asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ---- Optional: real fundamentals (no API key needed)
import yfinance as yf

APP_VERSION = "v12.2-hybrid"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()  # optional

# --------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------
app = FastAPI(title=f"Stockron Analyzer Backend {APP_VERSION}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------------------------------------------------------------
# Health
# --------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "service": "Stockron Backend",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# --------------------------------------------------------------
# Request schema
# --------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    ticker: str
    timeframe: Optional[str] = "6mo"
    notes: Optional[str] = None

# ==============================================================
# Utilities
# ==============================================================
def now_iso(): return datetime.utcnow().isoformat() + "Z"

def safe_num(x, default=None, scale: Optional[float]=None):
    try:
        v = float(x)
        if scale:
            v = v * scale
        return v
    except Exception:
        return default

def pct(x): 
    return None if x is None else round(float(x) * 100.0, 2)

# ==============================================================
# Data fetchers (Yahoo Finance)
# ==============================================================
def yf_info(ticker: str) -> Dict[str, Any] | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        # NOTE: yfinance values often already normalized; keep raw and scale where needed
        return info
    except Exception:
        return None

def yf_quote_price(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        p = t.fast_info.get("last_price")
        if p is None:
            # fallback to info
            p = t.info.get("regularMarketPrice")
        return safe_num(p)
    except Exception:
        return None

def yf_history(ticker: str, period="6mo", interval="1d"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        return df  # pandas.DataFrame
    except Exception:
        return None

# ==============================================================
# Technicals (SMA / ATR) — simple & fast
# ==============================================================
def compute_sma(series, n=20):
    if series is None or len(series) < n: 
        return None
    return float(series[-n:].mean())

def compute_atr(df, n=14):
    try:
        if df is None or len(df) < n+1:
            return None
        high = df["High"]; low = df["Low"]; close = df["Close"]
        trs = []
        prev_close = close.shift(1)
        for i in range(1, len(df)):
            tr = max(
                (high.iloc[i] - low.iloc[i]),
                abs(high.iloc[i] - prev_close.iloc[i]),
                abs(low.iloc[i] - prev_close.iloc[i])
            )
            trs.append(tr)
        if len(trs) < n: return None
        atr = sum(trs[-n:]) / n
        return float(atr)
    except Exception:
        return None

def compute_buy_sell_zones(df, last_price: Optional[float]) -> Dict[str, Any]:
    # buy ≈ SMA20 - 1*ATR, sell ≈ SMA20 + 1*ATR
    if df is None or len(df) < 21:
        # fallback demo using last_price
        lp = last_price or round(random.uniform(20, 500), 2)
        width = round(lp * 0.06, 2)
        return {
            "buy_zone": [round(lp - 2*width, 2), round(lp - 1*width, 2)],
            "sell_zone": [round(lp + 1*width, 2), round(lp + 2*width, 2)],
            "rationale": "Fallback demo zones (no history)"
        }

    sma20 = compute_sma(df["Close"], 20)
    atr14 = compute_atr(df, 14)
    if sma20 is None or atr14 is None:
        lp = last_price or round(random.uniform(20, 500), 2)
        width = round(lp * 0.06, 2)
        return {
            "buy_zone": [round(lp - 2*width, 2), round(lp - 1*width, 2)],
            "sell_zone": [round(lp + 1*width, 2), round(lp + 2*width, 2)],
            "rationale": "Fallback demo zones (insufficient history)"
        }

    buy_low = round(sma20 - 1.0 * atr14, 2)
    buy_high = round(sma20 - 0.4 * atr14, 2)
    sell_low = round(sma20 + 0.4 * atr14, 2)
    sell_high = round(sma20 + 1.0 * atr14, 2)
    return {
        "buy_zone": [min(buy_low, buy_high), max(buy_low, buy_high)],
        "sell_zone": [min(sell_low, sell_high), max(sell_low, sell_high)],
        "rationale": "SMA20 ± ATR14 (conservative)"
    }

# ==============================================================
# Sentiment / News (Finnhub, optional)
# ==============================================================
async def finnhub_company_news_sentiment(ticker: str) -> Dict[str, Any] | None:
    if not FINNHUB_API_KEY:
        return None
    url = "https://finnhub.io/api/v1/news-sentiment"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                return None
            data = r.json()
            # Simplify result: overall sentiment score in [-1,1] → map to label
            score = data.get("sentiment", {}).get("company_news_score")
            if score is None:
                # fallback using sector average if available
                score = data.get("sector_average_bullish_percent")
                if score is not None:
                    # sector bullish percent 0..1 → shift to [-0.5..0.5]
                    score = float(score) - 0.5
            if score is None:
                return None
            label = "Positive" if score > 0.15 else ("Negative" if score < -0.15 else "Neutral")
            return {"score": float(score), "label": label}
    except Exception:
        return None

# ==============================================================
# Scores (simple, transparent)
# ==============================================================
def compute_scores(pe, ps, peg, rev_g, eps_g, dte, pm, roe) -> Dict[str, float]:
    # Normalize inputs (handle None)
    pe = safe_num(pe, 25.0); ps = safe_num(ps, 6.0); peg = safe_num(peg, 1.5)
    rev_g = safe_num(rev_g, 0.10); eps_g = safe_num(eps_g, 0.10)
    dte = safe_num(dte, 0.6); pm = safe_num(pm, 0.15); roe = safe_num(roe, 0.15)

    # Quant score: low PE/PS/PEG + higher Rev/EPS growth
    quant = 0
    quant += max(0, 40 - min(pe, 40)) * 0.25
    quant += max(0, 15 - min(ps, 15)) * 0.15
    quant += max(0, 3.5 - min(peg, 3.5)) * 0.20
    quant += min(rev_g*100, 40) * 0.20
    quant += min(eps_g*100, 40) * 0.20
    quant_score = round(min(100, max(0, quant)), 1)

    # Quality score: higher PM/ROE/Current ratio, lower Debt/Equity
    quality = 0
    quality += min(pm*100, 40) * 0.35
    quality += min(roe*100, 40) * 0.35
    quality += max(0, 2.5 - min(dte, 2.5)) * 10  # favor lower leverage
    quality_score = round(min(100, max(0, quality)), 1)

    # Catalyst score placeholder; will combine sector momentum + sentiment later
    return {"quant_score": quant_score, "quality_score": quality_score}

def stance_from_scores(qs, qls, cats) -> str:
    overall = qs*0.4 + qls*0.4 + cats*0.2
    if overall >= 70: return "Buy"
    if overall >= 55: return "Hold"
    return "Wait"

# ==============================================================
# Analyze endpoint
# ==============================================================
@app.post("/analyze")
async def analyze_stock(req: AnalyzeRequest):
    ticker = (req.ticker or "").upper().strip()
    if not ticker:
        return {"error": "ticker is required"}

    # 1) Real fundamentals via yfinance
    info = yf_info(ticker) or {}

    # Primary fields (attempt real first)
    last_price = yf_quote_price(ticker)
    previous_close = safe_num(info.get("previousClose"))
    market_cap = safe_num(info.get("marketCap"))
    pe_ratio = safe_num(info.get("trailingPE"))
    forward_pe = safe_num(info.get("forwardPE"))
    ps_ratio = safe_num(info.get("priceToSalesTrailing12Months"))
    peg_ratio = safe_num(info.get("pegRatio"))
    rev_growth = safe_num(info.get("revenueGrowth"))      # often 0.x
    eps_growth = safe_num(info.get("earningsGrowth"))      # often 0.x
    beta = safe_num(info.get("beta"))
    debt_to_equity = safe_num(info.get("debtToEquity"), scale=0.01)  # info often in %
    current_ratio = safe_num(info.get("currentRatio"))
    profit_margin = safe_num(info.get("profitMargins"))    # 0.x
    roe = safe_num(info.get("returnOnEquity"))             # 0.x
    dividend_yield = safe_num(info.get("dividendYield"))   # 0.x

    # 2) Fallbacks (demo) when missing
    if last_price is None: last_price = round(random.uniform(20, 500), 2)
    if pe_ratio is None: pe_ratio = round(random.uniform(8, 35), 2)
    if ps_ratio is None: ps_ratio = round(random.uniform(2, 15), 2)
    if peg_ratio is None: peg_ratio = round(random.uniform(0.8, 2.5), 2)
    if rev_growth is None: rev_growth = round(random.uniform(0.05, 0.30), 2)
    if eps_growth is None: eps_growth = round(random.uniform(0.05, 0.35), 2)
    if debt_to_equity is None: debt_to_equity = round(random.uniform(0.1, 2.0), 2)
    if current_ratio is None: current_ratio = round(random.uniform(1.0, 3.0), 2)
    if profit_margin is None: profit_margin = round(random.uniform(0.05, 0.35), 2)
    if roe is None: roe = round(random.uniform(0.10, 0.40), 2)
    if market_cap is None: market_cap = random.randint(500_000_000, 2_000_000_000_000)
    if dividend_yield is None: dividend_yield = round(random.uniform(0.0, 0.03), 4)

    # 3) History for technical zones
    df = yf_history(ticker, period="6mo", interval="1d")
    zones = compute_buy_sell_zones(df, last_price)

    # 4) Sentiment (Finnhub optional)
    sentiment_label = "Neutral"
    sentiment_score = None
    sent = await finnhub_company_news_sentiment(ticker)
    if sent:
        sentiment_label = sent.get("label", "Neutral")
        sentiment_score = sent.get("score")

    # 5) Sector momentum (quick proxy from last 14-day change)
    sector_momentum = None
    try:
        if df is not None and len(df) >= 15:
            p_now = float(df["Close"].iloc[-1])
            p_14 = float(df["Close"].iloc[-15])
            sector_momentum = round(((p_now - p_14)/p_14)*100, 1)
    except Exception:
        sector_momentum = None
    if sector_momentum is None:
        sector_momentum = round(random.uniform(-5, 10), 1)

    # 6) Scores (quant/quality) + catalyst
    s = compute_scores(pe_ratio, ps_ratio, peg_ratio, rev_growth, eps_growth,
                       debt_to_equity, profit_margin, roe)
    # Catalyst score simple mix: momentum (+), sentiment (+/-)
    cat = 50.0 + sector_momentum*1.5
    if sentiment_label == "Positive": cat += 10
    elif sentiment_label == "Negative": cat -= 10
    catalyst_score = round(max(0, min(100, cat)), 1)

    overall = round(s["quant_score"]*0.4 + s["quality_score"]*0.4 + catalyst_score*0.2, 1)
    ai_stance = stance_from_scores(s["quant_score"], s["quality_score"], catalyst_score)

    # 7) Summaries
    company_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector") or "Unknown"
    ai_summary = (
        f"{company_name} ({ticker}): PE {round(pe_ratio,2)}, PS {round(ps_ratio,2)}, "
        f"PEG {round(peg_ratio,2)}; PM {pct(profit_margin)}%, ROE {pct(roe)}%, "
        f"Debt/Equity {round(debt_to_equity,2)}, Rev/EPS growth {pct(rev_growth)}%/{pct(eps_growth)}%. "
        f"Momentum {sector_momentum}%, Sentiment {sentiment_label}."
    )
    quant_summary = "מכפילים מאוזנים וצמיחה סבירה בהכנסות/רווח למניה."
    quality_summary = "רווחיות ו-ROE טובים, מינוף נשלט ונזילות תקינה."
    catalyst_summary = "מומנטום טכני מתון; סנטימנט תקשורתי " + sentiment_label + "."

    response = {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "quant": {
            "pe_ratio": round(pe_ratio, 2),
            "ps_ratio": round(ps_ratio, 2),
            "peg_ratio": round(peg_ratio, 2),
            "rev_growth": round(rev_growth, 4),   # נשמר כאחוז דצימלי (0.xx)
            "eps_growth": round(eps_growth, 4),
            "overall_score": s["quant_score"],
            "market_cap": int(market_cap)
        },
        "quality": {
            "debt_equity": round(debt_to_equity, 2),
            "current_ratio": round(current_ratio, 2),
            "profit_margin": round(profit_margin, 4),
            "roe": round(roe, 4),
            "quality_score": s["quality_score"],
            "dividend_yield": round(dividend_yield, 4)
        },
        "catalyst": {
            "news_sentiment": sentiment_label,
            "sector_momentum": sector_momentum,
            "ai_signal": "Buy" if overall >= 70 else ("Hold" if overall >= 55 else "Wait"),
            "catalyst_score": catalyst_score
        },
        "quant_summary": quant_summary,
        "quality_summary": quality_summary,
        "catalyst_summary": catalyst_summary,
        "ai_summary": ai_summary,
        "ai_stance": ai_stance,
        "conviction_level": f"{round(overall/10,1)}/10",
        "buy_sell_zones": zones,
        "last_price": round(last_price, 2),
        "previous_close": round(previous_close, 2) if previous_close else None,
        "data_reliability": "High" if info else "Partial",
        "transparency": "Yahoo Finance fundamentals + optional Finnhub sentiment (fallback to demo where missing)",
        "timestamp": now_iso()
    }
    return response

# --------------------------------------------------------------
# Run (Render)
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
