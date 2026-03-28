"""
strategy/regime_detector.py

NEPSE market regime detector.
Classifies current market conditions using NEPSE index price history.
Used to select the appropriate trading strategy and signal weightings.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategy.technical_indicators import (
    compute_indicators, compute_rsi, load_price_history
)
from db.models import get_engine, StockPrice, MarketSnapshot
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

# ── Regime thresholds (tuned for NEPSE) ──────────────────────────────────────

THRESHOLDS = {
    "bull_return_pct":    3.0,    # 20-day return > 3% → bullish
    "bear_return_pct":   -3.0,    # 20-day return < -3% → bearish
    "sideways_range":     2.0,    # abs(20-day return) < 2% → sideways
    "rsi_bull":          50.0,
    "rsi_bear":          50.0,
    "rsi_overbought":    70.0,
    "rsi_oversold":      30.0,
    "rsi_deep_oversold": 25.0,
    "bb_sideways":        0.08,   # BB width < 0.08 → compressed / sideways
    "vol_spike":          3.0,    # volume > 3× avg → capitulation signal
    "sma_cross_lookback": 30,     # days to look back for prior BEAR regime
}


# ── NEPSE index data loader ───────────────────────────────────────────────────

def load_nepse_index(days: int = 300) -> pd.DataFrame:
    """
    Load NEPSE index history from market_snapshots table.
    Falls back to synthesising from average stock prices if missing.
    Returns DataFrame with columns: date, close, volume (proxy).
    """
    try:
        engine = get_engine()
        with Session(engine) as session:
            rows = (
                session.query(MarketSnapshot)
                .order_by(MarketSnapshot.date.desc())
                .limit(days)
                .all()
            )
        if rows:
            records = [
                {
                    "date":   r.date,
                    "close":  float(r.nepse_index or 0),
                    "open":   float(r.nepse_index or 0),
                    "high":   float(r.nepse_index or 0),
                    "low":    float(r.nepse_index or 0),
                    "volume": float(r.turnover_npr or 0) / 1_000_000,  # in millions
                }
                for r in rows
                if r.nepse_index and r.nepse_index > 0
            ]
            if records:
                return pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.warning("Could not load NEPSE index from DB: %s", e)

    # Fallback: empty DataFrame with schema
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA20, SMA50, RSI, BB width to the index DataFrame."""
    if df.empty or len(df) < 21:
        return df
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["sma_20"]  = df["close"].rolling(20).mean()
    df["sma_50"]  = df["close"].rolling(50).mean()
    df["rsi_14"]  = compute_rsi(df["close"], 14)
    df["ret_20d"] = df["close"].pct_change(20) * 100
    df["ret_5d"]  = df["close"].pct_change(5) * 100
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"]  = df["volume"] / df["vol_sma_20"].replace(0, np.nan)
    # Bollinger width
    std = df["close"].rolling(20).std(ddof=0)
    df["bb_width"] = (4 * std) / df["sma_20"].replace(0, np.nan)
    return df


# ── Regime detection ──────────────────────────────────────────────────────────

def detect_regime(df_nepse_index: Optional[pd.DataFrame] = None) -> dict:
    """
    Classify NEPSE's current market regime using the last 60 trading days.

    Args:
        df_nepse_index: Optional pre-loaded DataFrame. Loads from DB if None.

    Returns:
        {
            "regime": "BULL" | "BEAR" | "SIDEWAYS" | "RECOVERY",
            "sub_regime": "EARLY_BULL" | "LATE_BULL" | "CAPITULATION" |
                          "CONSOLIDATION" | "NONE",
            "confidence": 0.0–1.0,
            "signals_used": [...],
            "description": str,
            "key_metrics": {...},
        }
    """
    if df_nepse_index is None:
        df_nepse_index = load_nepse_index(300)

    df = _add_indicators(df_nepse_index)

    # Fallback result when insufficient data
    no_data = {
        "regime": "SIDEWAYS",
        "sub_regime": "NONE",
        "confidence": 0.0,
        "signals_used": [],
        "description": "Insufficient NEPSE index data to classify regime.",
        "key_metrics": {},
    }

    if df.empty or len(df) < 55:
        logger.warning("Regime detection: insufficient data (%d rows)", len(df))
        return no_data

    recent = df.tail(60).copy()
    latest = df.iloc[-1]

    # ── Extract key metrics ───────────────────────────────────────────────────
    sma20     = float(latest.get("sma_20", 0) or 0)
    sma50     = float(latest.get("sma_50", 0) or 0)
    price     = float(latest["close"])
    rsi       = float(latest.get("rsi_14", 50) or 50)
    ret20     = float(latest.get("ret_20d", 0) or 0)
    ret5      = float(latest.get("ret_5d", 0) or 0)
    bb_width  = float(latest.get("bb_width", 0.1) or 0.1)
    vol_ratio = float(latest.get("vol_ratio", 1.0) or 1.0)

    key_metrics = {
        "price": price, "sma_20": sma20, "sma_50": sma50,
        "rsi_14": rsi, "ret_20d": ret20, "ret_5d": ret5,
        "bb_width": bb_width, "vol_ratio": vol_ratio,
    }

    # ── Signal scoring ────────────────────────────────────────────────────────
    signals_used = []
    bull_signals  = 0
    bear_signals  = 0
    side_signals  = 0
    total_weight  = 0

    def add_signal(name: str, direction: str, weight: float = 1.0) -> None:
        nonlocal bull_signals, bear_signals, side_signals, total_weight
        signals_used.append(f"{name}: {direction}")
        total_weight += weight
        if direction == "BULL":
            bull_signals += weight
        elif direction == "BEAR":
            bear_signals += weight
        else:
            side_signals += weight

    # SMA cross (weight 2)
    if sma20 > 0 and sma50 > 0:
        add_signal("SMA20>SMA50", "BULL" if sma20 > sma50 else "BEAR", 2.0)
        add_signal("price vs SMA50", "BULL" if price > sma50 else "BEAR", 1.5)

    # RSI (weight 1.5)
    if rsi > THRESHOLDS["rsi_bull"]:
        add_signal("RSI>50", "BULL", 1.5)
    else:
        add_signal("RSI<50", "BEAR", 1.5)

    # 20-day return (weight 2)
    if ret20 > THRESHOLDS["bull_return_pct"]:
        add_signal("20d-return>+3%", "BULL", 2.0)
    elif ret20 < THRESHOLDS["bear_return_pct"]:
        add_signal("20d-return<-3%", "BEAR", 2.0)
    elif abs(ret20) < THRESHOLDS["sideways_range"]:
        add_signal("20d-return flat", "SIDE", 1.5)

    # Bollinger width (weight 1)
    if bb_width < THRESHOLDS["bb_sideways"]:
        add_signal("BB compressed", "SIDE", 1.0)
    else:
        add_signal("BB expanding", "BULL" if ret20 > 0 else "BEAR", 0.5)

    # ── Primary regime classification ────────────────────────────────────────
    if total_weight == 0:
        return no_data

    bull_score = bull_signals / total_weight
    bear_score = bear_signals / total_weight
    side_score = side_signals / total_weight

    # Check for RECOVERY: was BEAR in last 30 days, now SMA20 crossing SMA50
    was_bear = False
    if len(df) >= 30 and sma20 > 0 and sma50 > 0:
        prior = df.iloc[-30]
        prior_sma20 = float(prior.get("sma_20") or 0)
        prior_sma50 = float(prior.get("sma_50") or 0)
        if prior_sma20 > 0 and prior_sma50 > 0:
            was_bear = prior_sma20 < prior_sma50

    if was_bear and sma20 > sma50 and rsi > 45:
        regime = "RECOVERY"
        confidence = min(0.9, 0.5 + (sma20 - sma50) / max(sma50, 1) * 5)
    elif bull_score >= 0.55:
        regime = "BULL"
        confidence = float(bull_score)
    elif bear_score >= 0.55:
        regime = "BEAR"
        confidence = float(bear_score)
    else:
        regime = "SIDEWAYS"
        confidence = max(side_score, 0.4)

    confidence = round(min(1.0, confidence), 2)

    # ── Sub-regime ────────────────────────────────────────────────────────────
    sub_regime = "NONE"
    if regime == "BULL":
        # Was it recently BEAR? → EARLY_BULL
        if was_bear:
            sub_regime = "EARLY_BULL"
        # Late bull: RSI overbought + declining volume despite rising price
        elif rsi > THRESHOLDS["rsi_overbought"] and vol_ratio < 0.9 and ret5 > 0:
            sub_regime = "LATE_BULL"
        else:
            sub_regime = "EARLY_BULL" if ret20 < 8 else "LATE_BULL"
    elif regime == "BEAR":
        # Capitulation: deep oversold + huge volume spike
        if rsi < THRESHOLDS["rsi_deep_oversold"] and vol_ratio > THRESHOLDS["vol_spike"]:
            sub_regime = "CAPITULATION"
        else:
            sub_regime = "CAPITULATION" if rsi < 28 else "NONE"
    elif regime == "SIDEWAYS":
        # Inside BB for 5+ days
        if bb_width < THRESHOLDS["bb_sideways"]:
            sub_regime = "CONSOLIDATION"

    # ── Plain English description ─────────────────────────────────────────────
    desc_map = {
        ("BULL",     "EARLY_BULL"):    f"Early bull run — SMA20 crossed above SMA50. RSI {rsi:.0f}, 20d return {ret20:+.1f}%. Building momentum.",
        ("BULL",     "LATE_BULL"):     f"Late bull stage — RSI {rsi:.0f} (overbought), volume declining. Reduce exposure, protect gains.",
        ("BEAR",     "CAPITULATION"):  f"Bear market capitulation — RSI {rsi:.0f} (deep oversold), vol spike {vol_ratio:.1f}×. Historical buy signal for blue chips.",
        ("BEAR",     "NONE"):          f"Bear market — SMA20 < SMA50, RSI {rsi:.0f}, 20d return {ret20:+.1f}%. Capital preservation mode.",
        ("SIDEWAYS", "CONSOLIDATION"): f"Tight consolidation — BB width {bb_width:.3f}, RSI {rsi:.0f}. Await breakout signal.",
        ("SIDEWAYS", "NONE"):          f"Sideways market — 20d return {ret20:+.1f}%, RSI {rsi:.0f}. Range-bound. Mean reversion plays only.",
        ("RECOVERY", "NONE"):          f"Recovery phase — SMA20 crossing above SMA50 after bear. RSI {rsi:.0f}, 20d return {ret20:+.1f}%. Cautious re-entry.",
    }
    description = desc_map.get(
        (regime, sub_regime),
        f"{regime} ({sub_regime}) — RSI {rsi:.0f}, 20d return {ret20:+.1f}%, confidence {confidence:.0%}"
    )

    return {
        "regime":       regime,
        "sub_regime":   sub_regime,
        "confidence":   confidence,
        "signals_used": signals_used,
        "description":  description,
        "key_metrics":  key_metrics,
    }


def get_regime_history(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling regime classification for a historical price DataFrame.
    Returns DataFrame with columns: date, regime, sub_regime, confidence.
    Useful for backtesting and visualisation.
    """
    df = _add_indicators(df)
    results = []
    for i in range(window, len(df)):
        slice_df = df.iloc[:i + 1]
        result = detect_regime(slice_df)
        results.append({
            "date":       df.iloc[i]["date"],
            "regime":     result["regime"],
            "sub_regime": result["sub_regime"],
            "confidence": result["confidence"],
        })
    return pd.DataFrame(results)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading NEPSE index data...")
    df = load_nepse_index(300)
    if df.empty:
        print("No NEPSE index data in database. Using synthetic demo data.")
        # Create synthetic data for demo
        import datetime
        dates = pd.date_range(end=datetime.date.today(), periods=200, freq="B")
        np.random.seed(42)
        prices = 2000 + np.cumsum(np.random.randn(200) * 15)
        volumes = np.random.exponential(1e6, 200)
        df = pd.DataFrame({"date": dates.date, "open": prices * 0.99,
                           "high": prices * 1.01, "low": prices * 0.98,
                           "close": prices, "volume": volumes})

    result = detect_regime(df)
    print(f"\n{'='*55}")
    print(f"  NEPSE Regime Detection")
    print(f"{'='*55}")
    print(f"  Regime:     {result['regime']} ({result['sub_regime']})")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Description: {result['description']}")
    print(f"\n  Key Metrics:")
    for k, v in result["key_metrics"].items():
        print(f"    {k:15s}: {v:.2f}")
    print(f"\n  Signals Used:")
    for s in result["signals_used"]:
        print(f"    • {s}")
