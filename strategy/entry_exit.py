"""
strategy/entry_exit.py

Entry and exit conditions for three NEPSE trading strategy modes.
Active strategy selected based on detected market regime.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Strategy definitions ──────────────────────────────────────────────────────

STRATEGIES: dict[str, dict] = {
    "momentum_bull": {
        "active_in_regimes": ["BULL", "RECOVERY"],
        "active_in_sub_regimes": ["EARLY_BULL", "NONE"],
        "description": "Ride sector momentum — buy breakouts with MiroFish confirmation",
        "entry_conditions": [
            "MiroFish bull_bear_score > +0.55",
            "Composite signal score > +0.50",
            "Stock price > SMA50",
            "RSI between 50–68 (not overbought)",
            "volume_ratio > 1.3 on entry day (volume confirmation)",
            "Sector is in top-2 by rotation momentum",
            "No book close within next 5 trading days",
        ],
        "entry_score_thresholds": {
            "mirofish_min":   0.55,
            "composite_min":  0.50,
            "rsi_min":        50.0,
            "rsi_max":        68.0,
            "vol_ratio_min":  1.30,
            "price_vs_sma50": True,    # price must be above SMA50
        },
        "exit_conditions": [
            "Stop loss: -8% from entry price (hard stop, no exceptions)",
            "Profit target: +18% (take 60% off, let 40% ride with trailing stop)",
            "Trailing stop: 5% below highest close since entry",
            "MiroFish turns BEARISH (score < -0.3) for 2 consecutive signals",
            "RSI crosses above 78 (overbought exit)",
            "Regime shifts to BEAR or SIDEWAYS",
        ],
        "exit_thresholds": {
            "stop_loss_pct":    -0.08,
            "profit_target_pct": 0.18,
            "trailing_stop_pct": 0.05,
            "rsi_exit_high":    78.0,
            "mirofish_exit":   -0.30,
        },
        "position_size": "Full Kelly (up to 18% per position)",
        "max_position_pct": 0.18,
        "kelly_fraction":   1.0,
        "max_hold_days":    30,
    },

    "mean_reversion_sideways": {
        "active_in_regimes": ["SIDEWAYS"],
        "active_in_sub_regimes": ["CONSOLIDATION", "NONE"],
        "description": "Buy oversold quality stocks — wait for bounce to SMA20",
        "entry_conditions": [
            "RSI < 35 (oversold)",
            "Stock within 5% of Bollinger lower band",
            "MiroFish score > 0 (not actively bearish)",
            "Strong fundamentals: P/E < sector average, EPS positive",
            "volume_ratio < 0.8 (quiet selling — not capitulation)",
            "Price above 52-week low by at least 10% (not in freefall)",
        ],
        "entry_score_thresholds": {
            "rsi_max":          35.0,
            "mirofish_min":     0.0,
            "vol_ratio_max":    0.80,
            "pct_from_52w_low": 0.10,
            "price_vs_bb_lower_pct": 0.05,
        },
        "exit_conditions": [
            "Stop loss: -6% from entry (tighter stop for ranging markets)",
            "Profit target: SMA20 price level (mean reversion complete)",
            "RSI crosses above 60",
            "Maximum hold: 12 trading days (exit regardless of price)",
        ],
        "exit_thresholds": {
            "stop_loss_pct":     -0.06,
            "rsi_exit_high":     60.0,
        },
        "position_size": "Half Kelly (up to 10% per position)",
        "max_position_pct": 0.10,
        "kelly_fraction":   0.50,
        "max_hold_days":    12,
    },

    "defensive_bear": {
        "active_in_regimes": ["BEAR"],
        "active_in_sub_regimes": ["CAPITULATION", "NONE"],
        "description": "Protect capital — only buy deep value during capitulation",
        "entry_conditions": [
            "MiroFish score > +0.70 (very high conviction required in bear market)",
            "RSI < 28 (deep oversold)",
            "Sub-regime is CAPITULATION (panic selling = buy signal)",
            "Volume spike > 3× average (institutional accumulation signal)",
            "Blue-chip only: NABIL, EBL, NHPC, NLIC, NLICL, HBL",
            "Stock historically recovered from similar drawdowns",
        ],
        "entry_score_thresholds": {
            "mirofish_min":  0.70,
            "rsi_max":       28.0,
            "vol_ratio_min": 3.0,
            "blue_chip_only": ["NABIL", "EBL", "NHPC", "NLIC", "NLICL", "HBL",
                               "SBI", "ADBL", "CZBIL", "UPPER", "BPCL"],
        },
        "exit_conditions": [
            "Stop loss: -5% from entry (very tight — bear markets are unforgiving)",
            "Profit target: SMA20 level (small win, preserve capital)",
            "Exit if regime does NOT shift to RECOVERY within 10 days",
        ],
        "exit_thresholds": {
            "stop_loss_pct":    -0.05,
            "max_hold_days":    10,
        },
        "position_size": "Quarter Kelly (max 6% per position)",
        "max_position_pct": 0.06,
        "kelly_fraction":   0.25,
        "max_hold_days":    10,
    },
}

# Fallback strategy when regime is unclear
DEFAULT_STRATEGY = "mean_reversion_sideways"


# ── Strategy selector ─────────────────────────────────────────────────────────

# Days to apply transition discount after regime change
TRANSITION_DAYS = 3
TRANSITION_SIZE_FACTOR = 0.5  # Reduce position size 50% for first 3 days

def select_active_strategy(
    regime:            dict,
    regime_history:    Optional[list] = None,
    days_since_change: Optional[int] = None,
) -> dict:
    """
    Select the appropriate strategy based on current market regime.

    Args:
        regime: Output from regime_detector.detect_regime()
        regime_history: Optional list of recent regime dicts for change detection
        days_since_change: If regime recently changed, apply transition discount

    Returns:
        {
            "strategy_name": str,
            "strategy":      dict,          # full strategy definition
            "size_factor":   float,         # 1.0 normally, 0.5 during transition
            "transition":    bool,
            "rationale":     str,
        }
    """
    regime_name = regime.get("regime", "SIDEWAYS")
    sub_regime  = regime.get("sub_regime", "NONE")
    confidence  = regime.get("confidence", 0.5)

    # Find matching strategy
    best_strategy = None
    for name, strategy in STRATEGIES.items():
        if regime_name in strategy["active_in_regimes"]:
            # Extra check: if LATE_BULL, use defensive
            if sub_regime == "LATE_BULL" and name != "defensive_bear":
                continue
            best_strategy = name
            break

    if best_strategy is None:
        best_strategy = DEFAULT_STRATEGY

    # Handle LATE_BULL explicitly
    if sub_regime == "LATE_BULL":
        best_strategy = "defensive_bear"

    strategy_def = STRATEGIES[best_strategy]

    # Transition discount
    is_transition = False
    size_factor = 1.0
    if days_since_change is not None and days_since_change <= TRANSITION_DAYS:
        is_transition = True
        size_factor = TRANSITION_SIZE_FACTOR

    # Reduce size if regime confidence is low
    if confidence < 0.5:
        size_factor = min(size_factor, 0.75)

    rationale = (
        f"Regime {regime_name} ({sub_regime}) → strategy '{best_strategy}': "
        f"{strategy_def['description']}. "
        + (f"Transition discount {TRANSITION_SIZE_FACTOR:.0%} applied "
           f"(day {days_since_change}/{TRANSITION_DAYS} of new regime)."
           if is_transition else "Full position sizing.")
    )

    return {
        "strategy_name": best_strategy,
        "strategy":      strategy_def,
        "size_factor":   size_factor,
        "transition":    is_transition,
        "rationale":     rationale,
    }


def check_entry_conditions(
    strategy_name: str,
    symbol:        str,
    indicators:    dict,
    mirofish_score: float,
    composite_score: float,
    sector_rank:   int = 5,
) -> dict:
    """
    Check if a specific stock meets entry conditions for the given strategy.

    Returns:
        {
            "passes": bool,
            "met_conditions": list[str],
            "failed_conditions": list[str],
            "score": float,             # 0–1, what fraction of conditions met
        }
    """
    strategy = STRATEGIES.get(strategy_name)
    if not strategy:
        return {"passes": False, "met_conditions": [], "failed_conditions": ["Unknown strategy"], "score": 0.0}

    thresholds = strategy.get("entry_score_thresholds", {})
    met = []
    failed = []

    rsi        = float(indicators.get("rsi_14") or 50)
    vol_ratio  = float(indicators.get("vol_ratio") or 1.0)
    close      = float(indicators.get("close") or 0)
    sma50      = float(indicators.get("sma_50") or 0)
    bb_lower   = float(indicators.get("bb_lower") or 0)
    week52_low = float(indicators.get("week52_low") or 0)

    def check(condition: str, passes: bool) -> None:
        (met if passes else failed).append(condition)

    if strategy_name == "momentum_bull":
        check(f"MiroFish >{thresholds.get('mirofish_min', 0.55):.2f}",
              mirofish_score > thresholds.get("mirofish_min", 0.55))
        check(f"Composite >{thresholds.get('composite_min', 0.50):.2f}",
              composite_score > thresholds.get("composite_min", 0.50))
        check(f"RSI {thresholds.get('rsi_min',50):.0f}–{thresholds.get('rsi_max',68):.0f}",
              thresholds.get("rsi_min", 50) <= rsi <= thresholds.get("rsi_max", 68))
        check(f"vol_ratio >{thresholds.get('vol_ratio_min', 1.3):.1f}",
              vol_ratio > thresholds.get("vol_ratio_min", 1.3))
        if sma50 > 0:
            check("price > SMA50", close > sma50)
        check("sector in top-2", sector_rank <= 2)

    elif strategy_name == "mean_reversion_sideways":
        check(f"RSI <{thresholds.get('rsi_max', 35):.0f}",
              rsi < thresholds.get("rsi_max", 35))
        check(f"MiroFish >{thresholds.get('mirofish_min', 0):.2f}",
              mirofish_score > thresholds.get("mirofish_min", 0))
        check(f"vol_ratio <{thresholds.get('vol_ratio_max', 0.8):.1f}",
              vol_ratio < thresholds.get("vol_ratio_max", 0.8))
        if bb_lower > 0 and close > 0:
            pct_above_bb = (close - bb_lower) / close
            check(f"within 5% of BB lower",
                  pct_above_bb < thresholds.get("price_vs_bb_lower_pct", 0.05))
        if week52_low > 0 and close > 0:
            pct_above_52low = (close - week52_low) / week52_low
            check("price >10% above 52w low",
                  pct_above_52low > thresholds.get("pct_from_52w_low", 0.10))

    elif strategy_name == "defensive_bear":
        check(f"MiroFish >{thresholds.get('mirofish_min', 0.70):.2f}",
              mirofish_score > thresholds.get("mirofish_min", 0.70))
        check(f"RSI <{thresholds.get('rsi_max', 28):.0f}",
              rsi < thresholds.get("rsi_max", 28))
        check(f"vol_ratio >{thresholds.get('vol_ratio_min', 3.0):.1f}",
              vol_ratio > thresholds.get("vol_ratio_min", 3.0))
        blue_chips = thresholds.get("blue_chip_only", [])
        check(f"blue-chip stock", symbol.upper() in blue_chips)

    total = len(met) + len(failed)
    score = len(met) / total if total > 0 else 0.0
    passes = len(failed) == 0 and len(met) > 0

    return {
        "passes":            passes,
        "met_conditions":    met,
        "failed_conditions": failed,
        "score":             round(score, 2),
    }


def check_exit_conditions(
    strategy_name:       str,
    entry_price:         float,
    current_price:       float,
    highest_price_since: float,
    days_held:           int,
    current_rsi:         float,
    mirofish_score:      float,
    current_regime:      str,
) -> dict:
    """
    Check if open position should be exited based on strategy exit conditions.

    Returns:
        {
            "should_exit": bool,
            "exit_reason": str or None,
            "urgency": "IMMEDIATE" | "NEXT_OPEN" | "WATCH",
        }
    """
    strategy = STRATEGIES.get(strategy_name)
    if not strategy:
        return {"should_exit": False, "exit_reason": None, "urgency": "WATCH"}

    thresholds = strategy.get("exit_thresholds", {})
    max_hold   = strategy.get("max_hold_days", 30)
    pct_change = (current_price - entry_price) / entry_price if entry_price > 0 else 0

    # Hard stop loss
    stop_pct = thresholds.get("stop_loss_pct", -0.08)
    if pct_change <= stop_pct:
        return {
            "should_exit": True,
            "exit_reason": f"Stop loss hit: {pct_change:.1%} (threshold {stop_pct:.1%})",
            "urgency": "IMMEDIATE",
        }

    # Trailing stop
    trailing_pct = thresholds.get("trailing_stop_pct", 0.05)
    if highest_price_since > 0 and trailing_pct:
        trailing_stop = highest_price_since * (1 - trailing_pct)
        if current_price < trailing_stop:
            return {
                "should_exit": True,
                "exit_reason": f"Trailing stop: price {current_price:.2f} < trailing stop {trailing_stop:.2f}",
                "urgency": "NEXT_OPEN",
            }

    # Profit target
    target_pct = thresholds.get("profit_target_pct", 0.18)
    if target_pct and pct_change >= target_pct:
        return {
            "should_exit": True,
            "exit_reason": f"Profit target reached: {pct_change:.1%} (target {target_pct:.1%})",
            "urgency": "NEXT_OPEN",
        }

    # RSI overbought
    rsi_high = thresholds.get("rsi_exit_high")
    if rsi_high and current_rsi > rsi_high:
        return {
            "should_exit": True,
            "exit_reason": f"RSI overbought: {current_rsi:.0f} > {rsi_high:.0f}",
            "urgency": "NEXT_OPEN",
        }

    # MiroFish reversal
    mf_exit = thresholds.get("mirofish_exit", -0.30)
    if mirofish_score < mf_exit:
        return {
            "should_exit": True,
            "exit_reason": f"MiroFish turned bearish: score {mirofish_score:.2f}",
            "urgency": "NEXT_OPEN",
        }

    # Max hold days
    if days_held >= max_hold:
        return {
            "should_exit": True,
            "exit_reason": f"Max hold period reached: {days_held} >= {max_hold} days",
            "urgency": "NEXT_OPEN",
        }

    # Regime change (from momentum_bull)
    if strategy_name == "momentum_bull" and current_regime in ("BEAR", "SIDEWAYS"):
        return {
            "should_exit": True,
            "exit_reason": f"Regime shifted to {current_regime} — momentum strategy no longer active",
            "urgency": "NEXT_OPEN",
        }

    return {"should_exit": False, "exit_reason": None, "urgency": "WATCH"}


if __name__ == "__main__":
    # Demo
    from strategy.regime_detector import detect_regime
    regime = {"regime": "BULL", "sub_regime": "EARLY_BULL", "confidence": 0.82}
    sel = select_active_strategy(regime)
    print(f"Active strategy: {sel['strategy_name']}")
    print(f"Size factor:     {sel['size_factor']:.0%}")
    print(f"Rationale:       {sel['rationale']}")

    # Check entry for NHPC
    entry = check_entry_conditions(
        strategy_name="momentum_bull",
        symbol="NHPC",
        indicators={"rsi_14": 54, "vol_ratio": 1.5, "close": 320, "sma_50": 300},
        mirofish_score=0.72,
        composite_score=0.62,
        sector_rank=1,
    )
    print(f"\nNHPC entry check: {'PASS' if entry['passes'] else 'FAIL'} ({entry['score']:.0%})")
    for c in entry["met_conditions"]:
        print(f"  ✓ {c}")
    for c in entry["failed_conditions"]:
        print(f"  ✗ {c}")
