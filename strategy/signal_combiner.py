"""
strategy/signal_combiner.py

Hybrid signal combiner — merges MiroFish simulation signals with technical
indicators and sector rotation into a single actionable NEPSE trade signal.

Weighting:
  BULL regime:    MiroFish 40%, Technical 35%, Sector 25%
  BEAR regime:    MiroFish 50%, Technical 30%, Sector 20%
  SIDEWAYS:       MiroFish 25%, Technical 50%, Sector 25%
  RECOVERY:       MiroFish 35%, Technical 40%, Sector 25%
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# ── Weighting by regime ───────────────────────────────────────────────────────

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "BULL":     {"mirofish": 0.40, "technical": 0.35, "sector": 0.25},
    "BEAR":     {"mirofish": 0.50, "technical": 0.30, "sector": 0.20},
    "SIDEWAYS": {"mirofish": 0.25, "technical": 0.50, "sector": 0.25},
    "RECOVERY": {"mirofish": 0.35, "technical": 0.40, "sector": 0.25},
}

# ── RSI-based technical score ─────────────────────────────────────────────────

def _technical_score_from_indicators(technical_data: dict) -> float:
    """
    Convert raw indicator values into a -1.0 to +1.0 technical score.
    Uses RSI, MACD histogram, Bollinger position, and volume ratio.
    """
    score = 0.0
    weight_total = 0.0

    # RSI contribution (weight 0.35)
    rsi = technical_data.get("rsi_14")
    if rsi is not None:
        # RSI 50 → 0, RSI 70 → +0.7, RSI 30 → -0.7
        rsi_score = (float(rsi) - 50) / 50.0
        rsi_score = max(-1.0, min(1.0, rsi_score * 1.4))
        score += 0.35 * rsi_score
        weight_total += 0.35

    # MACD histogram (weight 0.25)
    macd_hist = technical_data.get("macd_hist")
    atr       = technical_data.get("atr_14") or 1.0
    if macd_hist is not None and atr:
        # Normalise by ATR to make it comparable across stocks
        macd_norm = float(macd_hist) / float(atr)
        macd_score = max(-1.0, min(1.0, macd_norm * 0.5))
        score += 0.25 * macd_score
        weight_total += 0.25

    # Bollinger position (weight 0.20): is price above/below midpoint?
    bb_upper = technical_data.get("bb_upper")
    bb_lower = technical_data.get("bb_lower")
    close    = technical_data.get("close")
    if bb_upper and bb_lower and close:
        bb_range = float(bb_upper) - float(bb_lower)
        if bb_range > 0:
            bb_pos = (float(close) - float(bb_lower)) / bb_range  # 0=lower, 1=upper
            bb_score = (bb_pos - 0.5) * 2  # -1 to +1
            score += 0.20 * bb_score
            weight_total += 0.20

    # Volume ratio (weight 0.20): rising volume confirms direction
    vol_ratio = technical_data.get("vol_ratio")
    ret_5d    = technical_data.get("ret_5d", 0) or 0
    if vol_ratio is not None:
        vol_boost = min(1.0, max(-0.5, (float(vol_ratio) - 1.0)))
        # Volume confirms direction of recent price move
        direction = 1.0 if float(ret_5d) >= 0 else -1.0
        vol_score = vol_boost * direction
        score += 0.20 * vol_score
        weight_total += 0.20

    if weight_total == 0:
        return 0.0
    return round(score / weight_total, 3)


def _sector_score_from_rotation(sector_rotation: dict, regime: dict) -> float:
    """
    Extract a -1.0 to +1.0 sector score from the rotation dict.
    Uses the top sector combined_score as a proxy for market breadth.
    """
    rankings = sector_rotation.get("sector_rankings", [])
    if not rankings:
        return 0.0
    # Average of top 3 sector scores → market breadth signal
    top3 = sorted(rankings, key=lambda x: x.get("combined_score", 0), reverse=True)[:3]
    avg_top3 = sum(r.get("combined_score", 0) for r in top3) / len(top3)
    return round(float(avg_top3), 3)


# ── Main combiner ─────────────────────────────────────────────────────────────

def combine_signals(
    mirofish_signal:  dict,
    technical_data:   dict,
    regime:           dict,
    sector_rotation:  dict,
) -> dict:
    """
    Combine MiroFish, technical, and sector signals into a final NEPSE
    trade recommendation.

    Args:
        mirofish_signal:  from pipeline/signal_extractor.py
        technical_data:   from strategy/technical_indicators.get_latest_indicators()
        regime:           from strategy/regime_detector.detect_regime()
        sector_rotation:  from strategy/sector_rotation.get_rotation_signal()

    Returns:
        Full composite signal dict with action, conviction, position sizing.
    """
    today_str = date.today().isoformat()

    # ── Extract raw sub-scores ────────────────────────────────────────────────
    # MiroFish score: already -1.0 to +1.0
    mf_score      = float(mirofish_signal.get("bull_bear_score", 0.0) or 0.0)
    mf_confidence = float(mirofish_signal.get("confidence_pct", 50.0) or 50.0) / 100.0

    # Technical score: derived from indicators
    tech_score = _technical_score_from_indicators(technical_data)

    # Sector score: from rotation model
    sector_score = _sector_score_from_rotation(sector_rotation, regime)

    # ── Get weights for current regime ───────────────────────────────────────
    regime_name = regime.get("regime", "SIDEWAYS")
    weights = REGIME_WEIGHTS.get(regime_name, REGIME_WEIGHTS["SIDEWAYS"])

    mf_w   = weights["mirofish"]
    tech_w = weights["technical"]
    sect_w = weights["sector"]

    # ── Compute weighted composite ────────────────────────────────────────────
    composite = (mf_w * mf_score) + (tech_w * tech_score) + (sect_w * sector_score)
    composite = round(float(composite), 3)

    mf_contrib   = round(mf_w * mf_score, 3)
    tech_contrib = round(tech_w * tech_score, 3)
    sect_contrib = round(sect_w * sector_score, 3)

    # ── Gate logic: handle conflicts ──────────────────────────────────────────
    rsi         = float(technical_data.get("rsi_14") or 50)
    sub_regime  = regime.get("sub_regime", "NONE")
    override    = None

    # Gate 1: MiroFish bullish but RSI overbought → wait for pullback
    if mf_score > 0.3 and rsi > 75:
        override = ("HOLD", "LOW",
                    "MiroFish bullish but RSI overbought (%.0f) — "
                    "wait for pullback before entry." % rsi)

    # Gate 2: MiroFish bearish but RSI deeply oversold → potential bounce
    elif mf_score < -0.3 and rsi < 25:
        override = ("HOLD", "LOW",
                    "MiroFish bearish but RSI deeply oversold (%.0f) — "
                    "potential bounce. Avoid new shorts." % rsi)

    # Gate 3: Bear regime but MiroFish very bullish + capitulation signal → BUY
    elif (regime_name == "BEAR" and sub_regime == "CAPITULATION"
          and mf_score > 0.70 and mf_confidence > 0.65):
        override = ("BUY", "MEDIUM",
                    "Bear-market capitulation with strong MiroFish bullish signal "
                    "(%.2f) — defensive blue-chip accumulation." % mf_score)

    # Gate 4: Late bull + declining volume → reduce
    elif sub_regime == "LATE_BULL" and composite > 0.4:
        composite = composite * 0.75  # dampen signal in late bull
        override = None  # don't override, just reduce

    # ── Determine action from composite ──────────────────────────────────────
    if override:
        action, raw_conviction, gate_reason = override
        reasoning = gate_reason
    else:
        gate_reason = None
        if composite >= 0.45:
            action = "BUY"
        elif composite <= -0.45:
            action = "SELL"
        elif composite >= 0.15:
            action = "BUY"    # weak buy
        elif composite <= -0.15:
            action = "REDUCE"
        else:
            action = "HOLD"
        raw_conviction = None

    # ── Conviction from composite magnitude and agreement ─────────────────────
    if raw_conviction is None:
        # Check agreement between signals
        signs = [
            1 if mf_score > 0.1 else (-1 if mf_score < -0.1 else 0),
            1 if tech_score > 0.1 else (-1 if tech_score < -0.1 else 0),
            1 if sector_score > 0.1 else (-1 if sector_score < -0.1 else 0),
        ]
        agreement = abs(sum(signs))  # 0=conflict, 1=partial, 2=strong, 3=full
        abs_composite = abs(composite)

        if abs_composite >= 0.55 and agreement >= 2:
            conviction = "HIGH"
        elif abs_composite >= 0.30 and agreement >= 1:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"
    else:
        conviction = raw_conviction

    # ── Position sizing (Kelly-based) ─────────────────────────────────────────
    kelly_fractions = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}
    full_kelly_pct  = {"BULL": 0.18, "BEAR": 0.06, "SIDEWAYS": 0.10, "RECOVERY": 0.12}
    base_size = full_kelly_pct.get(regime_name, 0.10)
    position_size_pct = base_size * kelly_fractions.get(conviction, 0.25)

    # Override action → size 0 or minimal
    if action == "HOLD" and override:
        position_size_pct = 0.0
    elif action == "HOLD":
        position_size_pct = 0.0

    position_size_pct = round(position_size_pct, 3)

    # ── Best/worst sectors ────────────────────────────────────────────────────
    rankings = sector_rotation.get("sector_rankings", [])
    top_sector   = rankings[0]["display_name"] if rankings else "Unknown"
    avoid_sector = rankings[-1]["display_name"] if rankings else "Unknown"
    top_sector_key   = rankings[0].get("sector", "") if rankings else ""
    avoid_sector_key = rankings[-1].get("sector", "") if rankings else ""

    # ── Entry/exit conditions ─────────────────────────────────────────────────
    entry_conditions = _build_entry_conditions(composite, mf_score, rsi, regime_name)
    exit_conditions  = _build_exit_conditions(regime_name)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    if not override:
        reasoning = (
            f"Composite score {composite:+.3f} in {regime_name} regime "
            f"(MiroFish {mf_score:+.2f}, Technical {tech_score:+.2f}, "
            f"Sector {sector_score:+.2f}). "
            f"Sector leader: {top_sector}. "
            f"{'All signals agree — high conviction.' if conviction == 'HIGH' else 'Signals partially mixed — medium conviction.' if conviction == 'MEDIUM' else 'Conflicting signals — reduce size.'}"
        )

    return {
        "date":             today_str,
        "composite_score":  composite,
        "action":           action,
        "conviction":       conviction,
        "regime":           regime_name,
        "sub_regime":       sub_regime,
        "top_sector":       top_sector,
        "top_sector_key":   top_sector_key,
        "avoid_sector":     avoid_sector,
        "avoid_sector_key": avoid_sector_key,
        "entry_conditions": entry_conditions,
        "exit_conditions":  exit_conditions,
        "position_size_pct": position_size_pct,
        "signal_breakdown": {
            "mirofish_score":       mf_score,
            "mirofish_confidence":  mf_confidence,
            "technical_score":      tech_score,
            "sector_score":         sector_score,
            "mirofish_contribution": mf_contrib,
            "technical_contribution": tech_contrib,
            "sector_contribution":  sect_contrib,
        },
        "weights_used":     weights,
        "gate_triggered":   gate_reason,
        "reasoning":        reasoning,
    }


def _build_entry_conditions(composite: float, mf_score: float,
                             rsi: float, regime: str) -> list[str]:
    conditions = []
    if composite > 0:
        if mf_score < 0.5:
            conditions.append(f"Wait for MiroFish score > 0.5 (currently {mf_score:.2f})")
        if rsi > 70:
            conditions.append(f"Wait for RSI to pull back below 68 (currently {rsi:.0f})")
        if rsi < 45 and regime == "BULL":
            conditions.append("RSI dip to 45–55 range provides better entry")
        conditions.append("Volume ratio > 1.2 on entry day preferred")
    return conditions if conditions else ["All entry conditions met — ready to act"]


def _build_exit_conditions(regime: str) -> list[str]:
    base = [
        "Stop loss: -8% from entry (hard stop, no exceptions)",
        "Trailing stop: 5% below highest close since entry",
    ]
    if regime == "BULL":
        base += ["Profit target: +18% (take 60% off, let 40% ride)",
                 "RSI crosses above 78 (overbought exit)"]
    elif regime == "BEAR":
        base += ["Profit target: SMA20 level (small wins in bear market)",
                 "Exit if regime does NOT shift to RECOVERY within 10 days"]
    elif regime == "SIDEWAYS":
        base += ["Profit target: SMA20 (mean reversion complete)",
                 "Exit after 12 trading days regardless"]
    return base


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo with synthetic signals
    demo_mirofish = {"bull_bear_score": 0.62, "confidence_pct": 74.0,
                     "sector_signals": {"hydropower": 0.84, "banking": 0.71}}
    demo_technical = {"rsi_14": 54, "macd_hist": 2.5, "atr_14": 12.0,
                      "bb_upper": 340, "bb_lower": 295, "close": 320,
                      "vol_ratio": 1.4, "ret_5d": 2.3}
    demo_regime = {"regime": "BULL", "sub_regime": "EARLY_BULL",
                   "confidence": 0.82, "key_metrics": {}}
    demo_rotation = {"sector_rankings": [
        {"sector": "hydropower", "display_name": "Hydropower", "combined_score": 0.76},
        {"sector": "banking", "display_name": "Commercial Banks", "combined_score": 0.55},
        {"sector": "manufacturing", "display_name": "Manufacturing", "combined_score": -0.3},
    ]}

    result = combine_signals(demo_mirofish, demo_technical, demo_regime, demo_rotation)
    print(f"Composite Score: {result['composite_score']:+.3f}")
    print(f"Action:          {result['action']}")
    print(f"Conviction:      {result['conviction']}")
    print(f"Position Size:   {result['position_size_pct']:.1%}")
    print(f"Top Sector:      {result['top_sector']}")
    print(f"Reasoning:       {result['reasoning']}")
