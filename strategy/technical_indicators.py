"""
strategy/technical_indicators.py

Technical indicator engine for NEPSE price data.
All indicators computed from scratch using pandas — no TA-Lib dependency.
Tuned for NEPSE's low-liquidity conditions (short lookback periods).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from db.models import get_engine, StockPrice
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


# ── Helper functions ──────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (EMA-based).
    Returns values 0–100. NaN for first `period` rows.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — measures volatility.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    Uses Wilder's EMA smoothing.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_bollinger(series: pd.Series, period: int = 20,
                      num_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """
    Bollinger Bands: middle=SMA(period), upper=middle+num_std*σ, lower=middle-num_std*σ
    Returns (upper_band, lower_band).
    """
    sma = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)   # population std, consistent with most TA platforms
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume — cumulative volume flow.
    +volume when close > prev_close, -volume when close < prev_close.
    """
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


# ── Main indicator engine ─────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Add all technical indicators to a price DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        symbol: Stock symbol (used for logging only)

    Returns:
        df with new indicator columns added in-place.
    """
    if df.empty or len(df) < 20:
        logger.warning("Insufficient data for %s (rows=%d)", symbol, len(df))
        return df

    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Trend ──────────────────────────────────────────────────────────────────
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_13"] = df["close"].ewm(span=13, adjust=False).mean()

    # ── Momentum ───────────────────────────────────────────────────────────────
    df["rsi_14"]       = compute_rsi(df["close"], 14)
    df["macd"]         = (df["close"].ewm(span=12, adjust=False).mean()
                          - df["close"].ewm(span=26, adjust=False).mean())
    df["macd_signal"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]

    # ── Volatility ─────────────────────────────────────────────────────────────
    df["atr_14"]   = compute_atr(df, 14)
    df["bb_upper"], df["bb_lower"] = compute_bollinger(df["close"], 20, 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["sma_20"].replace(0, np.nan)

    # ── Volume ─────────────────────────────────────────────────────────────────
    df["vol_sma_20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"]   = df["volume"] / df["vol_sma_20"].replace(0, np.nan)
    df["obv"]         = compute_obv(df)

    # ── NEPSE-specific: 52-week levels ────────────────────────────────────────
    df["week52_high"]   = df["close"].rolling(252).max()
    df["week52_low"]    = df["close"].rolling(252).min()
    df["pct_from_high"] = (
        (df["close"] - df["week52_high"]) / df["week52_high"].replace(0, np.nan) * 100
    )

    # ── Short-term returns ─────────────────────────────────────────────────────
    df["ret_5d"]  = df["close"].pct_change(5) * 100
    df["ret_10d"] = df["close"].pct_change(10) * 100
    df["ret_20d"] = df["close"].pct_change(20) * 100

    logger.debug("Indicators computed for %s (%d rows)", symbol, len(df))
    return df


# ── Database helpers ──────────────────────────────────────────────────────────

def load_price_history(symbol: str, days: int = 500) -> pd.DataFrame:
    """
    Load OHLCV price history for a symbol from PostgreSQL.
    Returns DataFrame sorted by date ascending. Empty DF if no data.
    """
    try:
        engine = get_engine()
        with Session(engine) as session:
            rows = (
                session.query(StockPrice)
                .filter(StockPrice.symbol == symbol.upper())
                .order_by(StockPrice.date.desc())
                .limit(days)
                .all()
            )
        if not rows:
            logger.warning("No price data found for %s", symbol)
            return pd.DataFrame()

        records = [
            {
                "date":   r.date,
                "open":   float(r.open or r.ltp or 0),
                "high":   float(r.high or r.ltp or 0),
                "low":    float(r.low or r.ltp or 0),
                "close":  float(r.close or r.ltp or 0),
                "volume": float(r.volume or 0),
            }
            for r in rows
        ]
        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error("DB error loading %s: %s", symbol, e)
        return pd.DataFrame()


def get_indicators_for_symbol(symbol: str, days: int = 500) -> pd.DataFrame:
    """Load price history and compute all indicators. Returns ready-to-use DataFrame."""
    df = load_price_history(symbol, days)
    if df.empty:
        return df
    return compute_indicators(df, symbol)


def get_latest_indicators(symbol: str) -> dict:
    """
    Return the most recent row of indicators as a dict.
    Useful for signal combiner: just needs current values.
    """
    df = get_indicators_for_symbol(symbol)
    if df.empty:
        return {}
    latest = df.iloc[-1].to_dict()
    # Convert numpy types to Python native for JSON serialisation
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in latest.items()}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NABIL"
    print(f"Computing indicators for {symbol}...")
    df = get_indicators_for_symbol(symbol)
    if df.empty:
        print(f"No data for {symbol}")
    else:
        cols = ["date", "close", "sma_20", "sma_50", "rsi_14", "macd",
                "bb_upper", "bb_lower", "atr_14", "vol_ratio", "pct_from_high"]
        available = [c for c in cols if c in df.columns]
        print(df[available].tail(10).to_string(index=False))
        latest = get_latest_indicators(symbol)
        print(f"\nLatest RSI: {latest.get('rsi_14', 'N/A'):.1f}")
        print(f"Latest vol_ratio: {latest.get('vol_ratio', 'N/A'):.2f}")
