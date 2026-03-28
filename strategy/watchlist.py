"""
strategy/watchlist.py
Daily NEPSE watchlist generator — scores and ranks symbols using technical +
MiroFish + regime signals, then formats output for terminal and Telegram.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights (sum to 100)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "mirofish_sentiment":  25,   # MiroFish composite score
    "technical_momentum":  25,   # RSI / MACD / trend
    "sector_rank":         15,   # sector rotation rank (inverted)
    "volume_surge":        15,   # vol_ratio vs 20d avg
    "price_action":        10,   # distance from 52w high/low
    "regime_alignment":    10,   # does the regime support a BUY?
}

# Tier boundaries
TIER_A = 75   # Strong conviction
TIER_B = 55   # Moderate conviction
TIER_C = 40   # Watchlist only


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class WatchlistEntry:
    symbol: str
    score: float                      # 0-100
    tier: str                         # A / B / C / AVOID
    action: str                       # BUY / WATCH / AVOID
    regime: str = "UNKNOWN"
    composite_signal: float = 0.0
    mirofish_score: float = 0.0
    rsi: float = 50.0
    macd_hist: float = 0.0
    vol_ratio: float = 1.0
    sector: str = "unknown"
    sector_rank: int = 99
    ltp: float = 0.0
    week52_high: float = 0.0
    week52_low: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _score_mirofish(mf_score: float) -> float:
    """Map MiroFish score (-1..+1) → 0..25 pts."""
    # Normalise: [-1,+1] → [0,1] then scale
    normalised = (mf_score + 1.0) / 2.0
    # Apply non-linear boost for high-conviction scores
    if normalised > 0.7:
        normalised = 0.7 + (normalised - 0.7) * 1.5
    return min(SCORE_WEIGHTS["mirofish_sentiment"], normalised * SCORE_WEIGHTS["mirofish_sentiment"])


def _score_technical(rsi: float, macd_hist: float, atr: float,
                     ret_5d: float, ret_20d: float, sma20: float,
                     sma50: float, price: float) -> tuple[float, list[str]]:
    """Technical momentum score 0..25 pts."""
    pts = 0.0
    reasons: list[str] = []

    # RSI component (0..8 pts)
    if 45 <= rsi <= 65:
        pts += 8
        reasons.append(f"RSI healthy ({rsi:.1f})")
    elif 35 <= rsi < 45:
        pts += 5
        reasons.append(f"RSI recovering ({rsi:.1f})")
    elif 65 < rsi <= 72:
        pts += 4          # still ok but getting stretched
    elif rsi < 35:
        pts += 3          # oversold bounce potential
        reasons.append(f"RSI oversold ({rsi:.1f})")
    else:
        pts += 1          # overbought >72

    # MACD histogram (0..7 pts)
    if atr > 0 and macd_hist / atr > 0.05:
        pts += 7
        reasons.append("MACD bullish cross")
    elif atr > 0 and macd_hist / atr > 0.0:
        pts += 4
    elif atr > 0 and macd_hist / atr > -0.05:
        pts += 2

    # Trend alignment (0..5 pts)
    if price > sma20 > sma50:
        pts += 5
        reasons.append("Price>SMA20>SMA50")
    elif price > sma50:
        pts += 3
    elif price > sma20:
        pts += 2

    # Short-term momentum (0..5 pts)
    if ret_5d > 3.0:
        pts += 5
        reasons.append(f"5d return +{ret_5d:.1f}%")
    elif ret_5d > 1.0:
        pts += 3
    elif ret_5d > 0:
        pts += 1

    return min(SCORE_WEIGHTS["technical_momentum"], pts), reasons


def _score_sector(sector_rank: int, total_sectors: int = 8) -> float:
    """Sector rotation rank → 0..15 pts (rank 1 = best)."""
    if sector_rank <= 0 or total_sectors <= 0:
        return SCORE_WEIGHTS["sector_rank"] / 2  # neutral
    normalised = 1.0 - (sector_rank - 1) / total_sectors
    return normalised * SCORE_WEIGHTS["sector_rank"]


def _score_volume(vol_ratio: float) -> tuple[float, list[str]]:
    """Volume surge → 0..15 pts."""
    reasons: list[str] = []
    if vol_ratio >= 3.0:
        pts = 15.0
        reasons.append(f"Volume surge {vol_ratio:.1f}×")
    elif vol_ratio >= 2.0:
        pts = 11.0
        reasons.append(f"Above-avg volume {vol_ratio:.1f}×")
    elif vol_ratio >= 1.3:
        pts = 7.0
    elif vol_ratio >= 0.8:
        pts = 4.0
    else:
        pts = 1.0          # low/no volume
    return min(SCORE_WEIGHTS["volume_surge"], pts), reasons


def _score_price_action(price: float, week52_high: float,
                        week52_low: float) -> tuple[float, list[str]]:
    """Price position relative to 52-week range → 0..10 pts."""
    reasons: list[str] = []
    if week52_high <= week52_low or week52_high <= 0:
        return SCORE_WEIGHTS["price_action"] / 2, reasons

    range_pct = (price - week52_low) / (week52_high - week52_low)

    # Sweet spot: 30–65% of 52w range (not overbought, not in free-fall)
    if 0.30 <= range_pct <= 0.65:
        pts = 10.0
        reasons.append(f"Price at {range_pct*100:.0f}% of 52w range")
    elif 0.15 <= range_pct < 0.30:
        pts = 7.0          # near lows — recovery potential
        reasons.append(f"Near 52w low ({range_pct*100:.0f}%)")
    elif 0.65 < range_pct <= 0.85:
        pts = 5.0
    elif range_pct > 0.85:
        pts = 2.0          # near 52w high — stretched
    else:
        pts = 3.0          # deep value territory
    return min(SCORE_WEIGHTS["price_action"], pts), reasons


def _score_regime(regime: str, composite_signal: float) -> tuple[float, list[str]]:
    """Regime alignment → 0..10 pts."""
    reasons: list[str] = []
    base = 0.0

    if regime in ("BULL", "EARLY_BULL"):
        base = 10.0
        reasons.append(f"Regime: {regime}")
    elif regime == "RECOVERY":
        base = 8.0
        reasons.append("Recovery regime")
    elif regime == "SIDEWAYS":
        base = 5.0
    elif regime in ("BEAR", "CAPITULATION"):
        base = 2.0         # very selective in bear
    else:
        base = 4.0

    # Modulate by composite signal strength
    if composite_signal > 0.5:
        base = min(10.0, base * 1.2)
    elif composite_signal < 0:
        base = base * 0.5

    return min(SCORE_WEIGHTS["regime_alignment"], base), reasons


# ---------------------------------------------------------------------------
# Main watchlist builder
# ---------------------------------------------------------------------------

def build_watchlist(
    symbols: list[str],
    mirofish_signals: dict,        # {symbol: {score, conviction, action, ...}}
    sector_rotation: dict,         # output of get_rotation_signal()
    regime: str = "SIDEWAYS",
    top_n: int = 10,
) -> list[WatchlistEntry]:
    """
    Score every symbol and return top_n ranked entries.

    Parameters
    ----------
    symbols          : list of NEPSE symbols to evaluate
    mirofish_signals : dict keyed by symbol with MiroFish outputs
    sector_rotation  : sector rotation dict from sector_rotation.get_rotation_signal()
    regime           : current market regime string
    top_n            : how many entries to return
    """
    from strategy.technical_indicators import get_latest_indicators  # lazy import

    # Build sector → rank map
    sector_rank_map: dict[str, int] = {}
    if sector_rotation and "ranked_sectors" in sector_rotation:
        for i, sec_info in enumerate(sector_rotation["ranked_sectors"], start=1):
            sec_name = sec_info.get("sector", "")
            sector_rank_map[sec_name] = i

    entries: list[WatchlistEntry] = []

    for symbol in symbols:
        try:
            indicators = get_latest_indicators(symbol)
        except Exception as exc:
            logger.warning("Could not load indicators for %s: %s", symbol, exc)
            indicators = {}

        if not indicators:
            continue

        mf_data = mirofish_signals.get(symbol, {})
        mf_score: float = mf_data.get("score", 0.0)
        composite: float = mf_data.get("composite_signal", mf_score)
        action_raw: str = mf_data.get("action", "HOLD")

        # --- individual component scores ---
        pts_mf = _score_mirofish(mf_score)

        pts_tech, tech_reasons = _score_technical(
            rsi=indicators.get("rsi14", 50.0),
            macd_hist=indicators.get("macd_hist", 0.0),
            atr=indicators.get("atr14", 1.0) or 1.0,
            ret_5d=indicators.get("ret_5d", 0.0),
            ret_20d=indicators.get("ret_20d", 0.0),
            sma20=indicators.get("sma20", indicators.get("close", 100)),
            sma50=indicators.get("sma50", indicators.get("close", 100)),
            price=indicators.get("close", 100.0),
        )

        # Find sector for this symbol
        symbol_sector = "unknown"
        try:
            from strategy.sector_rotation import NEPSE_SECTORS
            for sec_name, sec_symbols in NEPSE_SECTORS.items():
                if symbol in sec_symbols:
                    symbol_sector = sec_name
                    break
        except ImportError:
            pass

        sec_rank = sector_rank_map.get(symbol_sector, 5)
        pts_sector = _score_sector(sec_rank)

        pts_vol, vol_reasons = _score_volume(indicators.get("vol_ratio", 1.0))

        pts_price, price_reasons = _score_price_action(
            price=indicators.get("close", 0.0),
            week52_high=indicators.get("week52_high", 0.0),
            week52_low=indicators.get("week52_low", 0.0),
        )

        pts_regime, regime_reasons = _score_regime(regime, composite)

        total_score = pts_mf + pts_tech + pts_sector + pts_vol + pts_price + pts_regime

        # Tier assignment
        if total_score >= TIER_A:
            tier = "A"
            action = "BUY"
        elif total_score >= TIER_B:
            tier = "B"
            action = "WATCH"
        elif total_score >= TIER_C:
            tier = "C"
            action = "WATCH"
        else:
            tier = "AVOID"
            action = "AVOID"

        # Override if MiroFish says SELL strongly
        if mf_score < -0.4 and action_raw == "SELL":
            tier = "AVOID"
            action = "AVOID"

        all_reasons = tech_reasons + vol_reasons + price_reasons + regime_reasons
        if mf_score > 0.5:
            all_reasons.insert(0, f"MiroFish bullish ({mf_score:.2f})")

        warnings: list[str] = []
        if indicators.get("rsi14", 50) > 72:
            warnings.append("RSI overbought")
        if indicators.get("vol_ratio", 1.0) < 0.3:
            warnings.append("Very low volume")

        entry = WatchlistEntry(
            symbol=symbol,
            score=round(total_score, 2),
            tier=tier,
            action=action,
            regime=regime,
            composite_signal=round(composite, 3),
            mirofish_score=round(mf_score, 3),
            rsi=round(indicators.get("rsi14", 50.0), 1),
            macd_hist=round(indicators.get("macd_hist", 0.0), 4),
            vol_ratio=round(indicators.get("vol_ratio", 1.0), 2),
            sector=symbol_sector,
            sector_rank=sec_rank,
            ltp=round(indicators.get("close", 0.0), 2),
            week52_high=round(indicators.get("week52_high", 0.0), 2),
            week52_low=round(indicators.get("week52_low", 0.0), 2),
            reasons=all_reasons[:5],
            warnings=warnings,
        )
        entries.append(entry)

    # Sort descending by score
    entries.sort(key=lambda e: e.score, reverse=True)
    return entries[:top_n]


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_terminal(entries: list[WatchlistEntry], regime: str,
                    today: Optional[date] = None) -> str:
    """Return a rich-friendly plain-text watchlist string."""
    today = today or date.today()
    lines = [
        f"\n{'='*62}",
        f"  📊  NEPSE DAILY WATCHLIST  —  {today}  —  Regime: {regime}",
        f"{'='*62}",
        f"{'#':>2}  {'Symbol':<8}  {'Score':>5}  {'Tier':>4}  {'RSI':>5}  "
        f"{'Vol×':>5}  {'MF':>6}  {'Action':<7}",
        f"{'-'*62}",
    ]
    for i, e in enumerate(entries, start=1):
        tier_icon = {"A": "🟢", "B": "🟡", "C": "🔵", "AVOID": "🔴"}.get(e.tier, "⚪")
        lines.append(
            f"{i:>2}  {e.symbol:<8}  {e.score:>5.1f}  {tier_icon}{e.tier:<3}  "
            f"{e.rsi:>5.1f}  {e.vol_ratio:>5.1f}  {e.mirofish_score:>+6.2f}  {e.action:<7}"
        )
        if e.reasons:
            lines.append(f"    → {', '.join(e.reasons[:3])}")
        if e.warnings:
            lines.append(f"    ⚠  {', '.join(e.warnings)}")

    lines.append(f"{'='*62}\n")
    return "\n".join(lines)


def format_telegram(entries: list[WatchlistEntry], regime: str,
                    today: Optional[date] = None) -> str:
    """Return Telegram-formatted markdown (MarkdownV2 safe) watchlist."""
    today = today or date.today()
    tier_emoji = {"A": "🟢", "B": "🟡", "C": "🔵", "AVOID": "🔴"}

    lines = [
        f"📊 *NEPSE Watchlist* — {today}",
        f"📈 Regime: `{regime}`",
        "",
    ]

    tier_a = [e for e in entries if e.tier == "A"]
    tier_b = [e for e in entries if e.tier == "B"]

    if tier_a:
        lines.append("*🟢 Tier A — High Conviction BUY*")
        for e in tier_a:
            reason_str = " | ".join(e.reasons[:2]) if e.reasons else ""
            lines.append(
                f"  • *{e.symbol}* — Score: `{e.score:.0f}` | RSI: `{e.rsi:.0f}` | "
                f"Vol: `{e.vol_ratio:.1f}×` | MF: `{e.mirofish_score:+.2f}`"
            )
            if reason_str:
                lines.append(f"    _{reason_str}_")
        lines.append("")

    if tier_b:
        lines.append("*🟡 Tier B — Watch*")
        for e in tier_b:
            lines.append(
                f"  • *{e.symbol}* — Score: `{e.score:.0f}` | RSI: `{e.rsi:.0f}` | "
                f"Vol: `{e.vol_ratio:.1f}×`"
            )
        lines.append("")

    avoid = [e for e in entries if e.tier == "AVOID"]
    if avoid:
        lines.append("*🔴 Avoid*: " + ", ".join(e.symbol for e in avoid))
        lines.append("")

    lines.append(f"_Generated {datetime.now().strftime('%H:%M NST')}_")
    return "\n".join(lines)


def send_telegram(message: str, bot_token: Optional[str] = None,
                  chat_id: Optional[str] = None) -> bool:
    """Send message to Telegram chat. Returns True on success."""
    import urllib.request
    import urllib.parse

    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id or "your_" in bot_token:
        logger.info("Telegram not configured — skipping send")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_watchlist(entries: list[WatchlistEntry], output_dir: str = "data/processed") -> str:
    """Persist watchlist to JSON. Returns path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    today_str = date.today().isoformat()
    path = Path(output_dir) / f"watchlist_{today_str}.json"

    data = [
        {
            "symbol": e.symbol,
            "score": e.score,
            "tier": e.tier,
            "action": e.action,
            "regime": e.regime,
            "composite_signal": e.composite_signal,
            "mirofish_score": e.mirofish_score,
            "rsi": e.rsi,
            "macd_hist": e.macd_hist,
            "vol_ratio": e.vol_ratio,
            "sector": e.sector,
            "sector_rank": e.sector_rank,
            "ltp": e.ltp,
            "week52_high": e.week52_high,
            "week52_low": e.week52_low,
            "reasons": e.reasons,
            "warnings": e.warnings,
        }
        for e in entries
    ]

    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"date": today_str, "watchlist": data}, fh, indent=2, ensure_ascii=False)

    logger.info("Watchlist saved → %s", path)
    return str(path)


def load_latest_watchlist(data_dir: str = "data/processed") -> list[dict]:
    """Load the most recent saved watchlist."""
    p = Path(data_dir)
    files = sorted(p.glob("watchlist_*.json"), reverse=True)
    if not files:
        return []
    with open(files[0], encoding="utf-8") as fh:
        return json.load(fh).get("watchlist", [])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_daily_watchlist(
    symbols: list[str],
    mirofish_signals: dict,
    sector_rotation: dict,
    regime: str = "SIDEWAYS",
    top_n: int = 10,
    send_to_telegram: bool = True,
    save: bool = True,
) -> list[WatchlistEntry]:
    """
    Full pipeline: score → rank → format → (optionally) Telegram + save.
    Returns ranked WatchlistEntry list.
    """
    entries = build_watchlist(symbols, mirofish_signals, sector_rotation,
                               regime=regime, top_n=top_n)

    terminal_output = format_terminal(entries, regime)
    print(terminal_output)

    if save:
        save_watchlist(entries)

    if send_to_telegram:
        tg_msg = format_telegram(entries, regime)
        sent = send_telegram(tg_msg)
        if sent:
            logger.info("Watchlist sent to Telegram")

    return entries
