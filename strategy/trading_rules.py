"""
strategy/trading_rules.py

NEPSE-specific trading rules engine.
Hard constraints that override signal strength — these are non-negotiable.
Also implements Kelly criterion position sizing.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ── Hard rule definitions ─────────────────────────────────────────────────────

NEPSE_HARD_RULES = {
    "circuit_breaker_buffer": {
        "description": "Never enter if stock gained > 7% today",
        "reason": "NEPSE ±10% circuit breaker caps upside — entering near ceiling is poor risk/reward",
        "severity": "HARD",
    },
    "t3_liquidity": {
        "description": "Max 20% of portfolio in one stock",
        "reason": "T+3 settlement locks capital for 3 days — NEPSE liquidity can trap oversized positions",
        "severity": "HARD",
    },
    "book_close_blackout": {
        "description": "No buy within 5 trading days before book close",
        "reason": "Price spikes pre-book-close for dividend/bonus, corrects sharply after",
        "severity": "HARD",
    },
    "minimum_liquidity": {
        "description": "Min avg daily turnover NPR 1 crore (10M) over 10 days",
        "reason": "Illiquid stocks are hard to exit — must be able to sell within 1–2 days",
        "severity": "HARD",
    },
    "ipo_listing_wait": {
        "description": "No buy on IPO listing day or day+1",
        "reason": "Listing day prices inflated by hype — wait for price discovery by day 2–3",
        "severity": "HARD",
    },
    "max_positions": {
        "description": "Hold no more than 6 open positions simultaneously",
        "reason": "NEPSE monitoring is manual-intensive — >6 positions is unmanageable",
        "severity": "HARD",
    },
}

NEPSE_SOFT_WARNINGS = {
    "approaching_circuit_up": {
        "description": "Stock up > 5% today — approaching upper circuit",
        "threshold": 0.05,
    },
    "book_close_warning": {
        "description": "Book close within 5–10 days — price spike risk",
        "threshold": 10,
    },
    "low_volume_warning": {
        "description": "Daily turnover < NPR 2 crore — low but not disqualifying",
        "threshold": 20_000_000,
    },
    "high_rsi_warning": {
        "description": "RSI > 68 — approaching overbought territory",
        "threshold": 68,
    },
    "high_pe_warning": {
        "description": "P/E ratio > 30 — premium valuation in NEPSE context",
        "threshold": 30,
    },
}


# ── Rule checkers ─────────────────────────────────────────────────────────────

def check_circuit_breaker_buffer(ltp: float, prev_close: float) -> tuple[bool, str]:
    """Returns (passes: bool, reason: str)."""
    if prev_close <= 0:
        return True, "No prior close available — skip check"
    daily_gain = (ltp - prev_close) / prev_close
    if daily_gain > 0.07:
        return False, f"Stock gained {daily_gain:.1%} today — above 7% circuit buffer threshold"
    return True, f"Daily gain {daily_gain:.1%} within acceptable range"


def check_t3_liquidity(position_pct: float) -> tuple[bool, str]:
    """Returns (passes: bool, reason: str)."""
    if position_pct > 0.20:
        return False, f"Position size {position_pct:.1%} exceeds 20% max (T+3 risk)"
    return True, f"Position size {position_pct:.1%} within T+3 limit"


def check_book_close_blackout(days_to_book_close: Optional[int]) -> tuple[bool, str]:
    """Returns (passes: bool, reason: str). days_to_book_close < 0 = already past."""
    if days_to_book_close is None:
        return True, "No book close date known — skip check"
    if 0 <= days_to_book_close <= 5:
        return False, f"Book close in {days_to_book_close} trading days — blackout period"
    return True, (f"Book close in {days_to_book_close} days — outside blackout"
                  if days_to_book_close > 5 else "Book close already passed")


def check_minimum_liquidity(avg_turnover_npr: float) -> tuple[bool, str]:
    """Returns (passes: bool, reason: str)."""
    MIN_TURNOVER = 10_000_000  # NPR 1 crore
    if avg_turnover_npr < MIN_TURNOVER:
        return False, (f"Avg turnover NPR {avg_turnover_npr:,.0f} below minimum "
                       f"NPR {MIN_TURNOVER:,.0f} (1 crore)")
    return True, f"Avg turnover NPR {avg_turnover_npr:,.0f} meets liquidity requirement"


def check_ipo_listing_wait(days_since_listing: Optional[int]) -> tuple[bool, str]:
    """Returns (passes: bool, reason: str). None = not an IPO."""
    if days_since_listing is None:
        return True, "Not an IPO stock"
    if days_since_listing <= 1:
        return False, f"IPO listing only {days_since_listing} day(s) ago — wait for price discovery"
    return True, f"IPO listed {days_since_listing} days ago — price discovery complete"


def check_max_positions(open_positions: int, max_positions: int = 6) -> tuple[bool, str]:
    """Returns (passes: bool, reason: str)."""
    if open_positions >= max_positions:
        return False, f"Already holding {open_positions} positions — max {max_positions} reached"
    return True, f"Currently {open_positions} open positions — room for more"


# ── Soft warning checkers ─────────────────────────────────────────────────────

def get_soft_warnings(
    ltp: float = 0,
    prev_close: float = 0,
    days_to_book_close: Optional[int] = None,
    avg_turnover_npr: float = 0,
    rsi: float = 50,
    pe_ratio: Optional[float] = None,
) -> list[str]:
    """Return list of soft warning strings (non-blocking but noted in output)."""
    warnings = []

    if prev_close > 0:
        daily_gain = (ltp - prev_close) / prev_close
        if daily_gain > NEPSE_SOFT_WARNINGS["approaching_circuit_up"]["threshold"]:
            warnings.append(f"Stock up {daily_gain:.1%} today — near upper circuit (use caution)")

    if days_to_book_close is not None and 5 < days_to_book_close <= 10:
        warnings.append(f"Book close in {days_to_book_close} days — monitor for pre-book spike")

    if 0 < avg_turnover_npr < NEPSE_SOFT_WARNINGS["low_volume_warning"]["threshold"]:
        warnings.append(f"Avg turnover NPR {avg_turnover_npr:,.0f} — low (exit may be slow)")

    if rsi > NEPSE_SOFT_WARNINGS["high_rsi_warning"]["threshold"]:
        warnings.append(f"RSI {rsi:.0f} approaching overbought (>68)")

    if pe_ratio and pe_ratio > NEPSE_SOFT_WARNINGS["high_pe_warning"]["threshold"]:
        warnings.append(f"P/E {pe_ratio:.1f} is elevated for NEPSE — prefer P/E < 20")

    return warnings


# ── Main validator ────────────────────────────────────────────────────────────

def validate_trade(
    symbol:            str,
    action:            str,
    position_pct:      float,
    open_positions:    int,
    ltp:               float               = 0.0,
    prev_close:        float               = 0.0,
    avg_turnover_npr:  float               = 0.0,
    days_to_book_close: Optional[int]      = None,
    days_since_listing: Optional[int]      = None,
    rsi:               float               = 50.0,
    pe_ratio:          Optional[float]     = None,
    max_open_positions: int                = 6,
) -> dict:
    """
    Run all hard rules against a proposed trade.

    Returns:
        {
            "approved": bool,
            "failed_rules": list[str],
            "passed_rules": list[str],
            "soft_warnings": list[str],
            "adjusted_position_pct": float,
            "summary": str,
        }
    """
    failed_rules = []
    passed_rules = []

    # Only run hard checks for BUY actions (SELL/HOLD don't need most of these)
    if action in ("BUY",):
        checks = [
            ("circuit_breaker_buffer", check_circuit_breaker_buffer(ltp, prev_close)),
            ("t3_liquidity",           check_t3_liquidity(position_pct)),
            ("book_close_blackout",    check_book_close_blackout(days_to_book_close)),
            ("minimum_liquidity",      check_minimum_liquidity(avg_turnover_npr)),
            ("ipo_listing_wait",       check_ipo_listing_wait(days_since_listing)),
            ("max_positions",          check_max_positions(open_positions, max_open_positions)),
        ]
        for rule_name, (passes, reason) in checks:
            if passes:
                passed_rules.append(f"✓ {rule_name}: {reason}")
            else:
                failed_rules.append(f"✗ {rule_name}: {reason}")
    else:
        passed_rules.append(f"SELL/HOLD — hard rules not applicable")

    # Soft warnings always computed
    soft_warnings = get_soft_warnings(
        ltp=ltp, prev_close=prev_close,
        days_to_book_close=days_to_book_close,
        avg_turnover_npr=avg_turnover_npr,
        rsi=rsi, pe_ratio=pe_ratio,
    )

    approved = len(failed_rules) == 0

    # Adjust position size if soft warnings
    adjusted_position_pct = position_pct
    if soft_warnings:
        adjusted_position_pct = round(position_pct * 0.75, 3)
    if not approved:
        adjusted_position_pct = 0.0

    summary = (
        f"{symbol}: {'APPROVED' if approved else 'REJECTED'} — "
        f"{len(failed_rules)} failed rule(s), {len(soft_warnings)} warning(s)"
    )

    return {
        "approved":              approved,
        "failed_rules":          failed_rules,
        "passed_rules":          passed_rules,
        "soft_warnings":         soft_warnings,
        "adjusted_position_pct": adjusted_position_pct,
        "summary":               summary,
    }


# ── Kelly position sizing ─────────────────────────────────────────────────────

def apply_position_sizing(
    signal:               dict,
    portfolio_value_npr:  float,
    validation_result:    dict,
) -> dict:
    """
    Apply Kelly criterion position sizing to a signal.

    Full Kelly   → conviction=HIGH and all hard rules pass
    Half Kelly   → conviction=MEDIUM
    Quarter Kelly → conviction=LOW or any soft warning triggered

    Returns:
        {
            "kelly_fraction": float,
            "max_position_pct": float,
            "max_position_npr": float,
            "shares_at_price": dict,  # if price provided
            "sizing_reason": str,
        }
    """
    conviction = signal.get("conviction", "LOW")
    base_pct   = signal.get("position_size_pct", 0.05)
    soft_warns = validation_result.get("soft_warnings", [])
    approved   = validation_result.get("approved", False)

    if not approved:
        return {
            "kelly_fraction":   0.0,
            "max_position_pct": 0.0,
            "max_position_npr": 0.0,
            "sizing_reason":    "Position rejected — hard rules failed",
        }

    if conviction == "HIGH" and not soft_warns:
        kelly_fraction = 1.0
        sizing_reason  = "Full Kelly — HIGH conviction, no warnings"
    elif conviction == "MEDIUM" or (conviction == "HIGH" and soft_warns):
        kelly_fraction = 0.5
        sizing_reason  = "Half Kelly — MEDIUM conviction or soft warnings present"
    else:
        kelly_fraction = 0.25
        sizing_reason  = "Quarter Kelly — LOW conviction or multiple warnings"

    adjusted_pct = min(0.20, base_pct * kelly_fraction)  # Hard cap at 20%
    adjusted_npr = portfolio_value_npr * adjusted_pct

    return {
        "kelly_fraction":   kelly_fraction,
        "max_position_pct": round(adjusted_pct, 3),
        "max_position_npr": round(adjusted_npr, 2),
        "sizing_reason":    sizing_reason,
    }


if __name__ == "__main__":
    # Demo validation
    result = validate_trade(
        symbol="NHPC",
        action="BUY",
        position_pct=0.15,
        open_positions=3,
        ltp=318.0,
        prev_close=310.0,
        avg_turnover_npr=25_000_000,
        days_to_book_close=12,
        days_since_listing=None,
        rsi=54.0,
        pe_ratio=22.0,
    )
    print(result["summary"])
    for r in result["passed_rules"]:
        print(f"  {r}")
    for r in result["failed_rules"]:
        print(f"  {r}")
    if result["soft_warnings"]:
        print("Warnings:")
        for w in result["soft_warnings"]:
            print(f"  ⚠ {w}")
