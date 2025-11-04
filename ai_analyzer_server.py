import math
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf

APP_VERSION = "v11-basic-1.0.0"

app = FastAPI(title="Stockron Analyzer Backend (Basic)", version=APP_VERSION, docs_url="/docs", redoc_url="/redoc")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1/window, adjust=False).mean()
    rd = dn.ewm(alpha=1/window, adjust=False).mean()
    rs = ru / (rd.replace(0, 1e-9))
    return 100.0 - (100.0 / (1.0 + rs))

def true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def atr(h, l, c, window: int = 14):
    return true_range(h,l,c).rolling(window=window, min_periods=max(2, window//2)).mean()

def safe(v, fb=None):
    try:
        if v is None: return fb
        if isinstance(v, (int,float)) and math.isfinite(v): return float(v)
        if isinstance(v, str):
            x = float(v.replace(',','').replace('%','').strip())
            if math.isfinite(x): return x
        return fb
    except Exception:
        return fb

class AnalyzeIn(BaseModel):
    ticker: str
    timeframe: Optional[str] = Field(default="6mo")

class AnalyzeOut(BaseModel):
    ticker: str
    ai_stance: str
    scores: Dict[str, float]
    summaries: Dict[str, str]
    metrics: Dict[str, Optional[float]]
    buy_sell: Dict[str, Any]

def yf_period(tf: str) -> str:
    return {"1mo":"1mo","3mo":"3mo","6mo":"6mo","1y":"1y","5y":"5y","max":"max"}.get(tf,"6mo")

def fetch_prices(ticker: str, timeframe: str) -> pd.DataFrame:
    df = yf.download(ticker, period=yf_period(timeframe), interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise HTTPException(404, f"No price data for {ticker}")
    return df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})

def fetch_fundamentals(ticker: str) -> Dict[str, Optional[float]]:
    t = yf.Ticker(ticker)
    out: Dict[str, Optional[float]] = {}
    try:
        fi = t.fast_info
        out["price"] = safe(getattr(fi, "last_price", None))
        out["market_cap"] = safe(getattr(fi, "market_cap", None))
        out["pe"] = safe(getattr(fi, "trailing_pe", None))
        out["forward_pe"] = safe(getattr(fi, "forward_pe", None))
        out["beta"] = safe(getattr(fi, "beta", None))
    except Exception:
        pass
    try:
        inc = t.get_income_stmt()
        if isinstance(inc, pd.DataFrame) and not inc.empty and "BasicEPS" in inc.index:
            eps = inc.loc["BasicEPS"].dropna()
            if len(eps) >= 2:
                last, prev = float(eps.iloc[0]), float(eps.iloc[1])
                if prev != 0:
                    out["eps_growth"] = (last - prev) / abs(prev) * 100.0
    except Exception:
        pass
    try:
        bs = t.get_balance_sheet()
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            liab = bs.loc["TotalLiabilitiesNetMinorityInterest"].dropna().iloc[0]
            eq = bs.loc["TotalEquityGrossMinorityInterest"].dropna().iloc[0]
            if eq and eq != 0:
                out["debt_equity"] = float(liab)/float(eq)
    except Exception:
        pass
    return out

def derive_peg(pe: Optional[float], eps_growth_pct: Optional[float]) -> Optional[float]:
    if pe is None or eps_growth_pct is None: return None
    g = eps_growth_pct/100.0
    if not g or not math.isfinite(g): return None
    return pe / g

def compute_zones(df: pd.DataFrame):
    c,h,l = df["close"], df["high"], df["low"]
    s50 = sma(c,50).iloc[-1]
    a14 = atr(h,l,c,14).iloc[-1]
    last_price = float(c.iloc[-1])
    prev_high = h.tail(60).max()
    vol_ratio = a14 / max(last_price, 1e-9)
    delta = min(0.08, max(0.03, vol_ratio*2.0))
    eps = min(0.05, max(0.02, vol_ratio*1.5))
    buy = (float(s50*(1-delta)), float(s50))
    sell = (float(prev_high*(1-eps)), float(prev_high))
    return buy, sell, "Dynamic zones based on SMA50 & ATR."

def scores_from(metrics: Dict[str, Optional[float]]):
    pe, growth, debt, beta = safe(metrics.get("pe")), safe(metrics.get("eps_growth")), safe(metrics.get("debt_equity")), safe(metrics.get("beta"))
    quant = 50.0 + (12 if (pe is not None and pe<15) else 0) - (10 if (pe is not None and pe>60) else 0) + (15 if (growth and growth>20) else 0) - (10 if (growth is not None and growth<0) else 0)
    quality = 50.0 + (15 if (debt is not None and debt<0.5) else 0) - (10 if (debt is not None and debt>2.0) else 0)
    catalyst = 50.0 + (5 if (growth and growth>10) else 0) - (5 if (beta and beta>1.5) else 0)
    clamp = lambda x: float(max(0, min(100, round(x))))
    return {"quant":clamp(quant), "quality":clamp(quality), "catalyst":clamp(catalyst)}

def stance(scores):
    w = 0.4*scores["quant"] + 0.4*scores["quality"] + 0.2*scores["catalyst"]
    return "Buy" if w>=70 else ("Hold" if w>=55 else "Wait")

@app.get("/healthz")
def healthz():
    return {"name":"Stockron Analyzer Backend (Basic)","status":"ok","version":APP_VERSION}

class _Out(BaseModel):
    ticker: str
    ai_stance: str
    scores: Dict[str, float]
    summaries: Dict[str, str]
    metrics: Dict[str, Optional[float]]
    buy_sell: Dict[str, Any]

@app.post("/analyze", response_model=_Out)
def analyze(inp: AnalyzeIn):
    t = (inp.ticker or "").upper().strip()
    if not t: raise HTTPException(400, "ticker required")
    df = fetch_prices(t, inp.timeframe)
    price = float(df["close"].iloc[-1])
    rsi14 = float(rsi(df["close"]).iloc[-1])
    s50 = float(sma(df["close"],50).iloc[-1])
    buy, sell, rationale = compute_zones(df)
    f = fetch_fundamentals(t)
    metrics = {
        "price": price, "pe": f.get("pe"), "forward_pe": f.get("forward_pe"),
        "ps": f.get("ps"), "peg": derive_peg(f.get("pe"), f.get("eps_growth")),
        "eps_growth": f.get("eps_growth"), "debt_equity": f.get("debt_equity"),
        "beta": f.get("beta"), "rsi14": rsi14, "sma50": s50
    }
    scores = scores_from(metrics)
    st = stance(scores)
    sums = {
        "quant_summary": "מכפילים מול צמיחה — איזון סביר.",
        "quality_summary": "מינוף נמוך עד בינוני מועדף.",
        "catalyst_summary": "מומנטום וצמיחה יכולים לשמש כקטליזטור."
    }
    return _Out(
        ticker=t, ai_stance=st, scores=scores, summaries=sums, metrics=metrics,
        buy_sell={"buy_zone":[round(buy[0],4),round(buy[1],4)], "sell_zone":[round(sell[0],4),round(sell[1],4)], "rationale": rationale}
    )
