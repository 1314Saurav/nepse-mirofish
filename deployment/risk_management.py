"""
deployment/risk_management.py
Live trading risk management rules for NEPSE MiroFish.

Applies ONLY after go-live with real capital.
During paper trading, these are advisory only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Live risk rule constants
# ---------------------------------------------------------------------------

LIVE_RISK_RULES: dict[str, Any] = {
    "account": {
        "starting_capital_npr": 500_000,    # Start small, scale after 3 months
        "max_portfolio_npr": 2_000_000,     # Ceiling for 12 months
        "capital_scaling": "add 25% per quarter if Sharpe >= 1.2",
    },
    "position": {
        "max_per_stock_pct": 15,            # Max 15% in one stock
        "max_open_positions": 5,            # Max 5 simultaneous
        "min_position_size_npr": 10_000,    # Floor to avoid high brokerage %
        "max_position_size_npr": 75_000,    # 15% of 500k starting
    },
    "loss_limits": {
        "daily_loss_limit_pct": 3,          # Stop trading if down 3% on the day
        "weekly_loss_limit_pct": 7,         # Reduce positions if down 7% this week
        "monthly_loss_limit_pct": 15,       # Emergency stop if down 15% this month
        "max_consecutive_losses": 5,        # Review strategy after 5 losses in a row
    },
    "emergency": {
        "circuit_breaker_threshold_pct": 9.5,   # NEPSE circuit breaker
        "emergency_exit_on_circuit_break": True,
        "halt_on_api_error_minutes": 30,
        "halt_on_zep_error_minutes": 60,
    },
    "review_triggers": [
        "signal_accuracy_5d < 45% for 2 weeks",
        "drawdown > 12% at any point",
        "3 consecutive weeks negative return",
        "MiroFish quality flags > 25% for 1 week",
        "NEPSE circuit breaker triggered",
    ],
}

# Regime exposure multipliers — scale position size down in weaker regimes
_REGIME_MULTIPLIER: dict[str, float] = {
    "BULL": 1.0,
    "SIDEWAYS": 0.6,
    "BEAR": 0.3,
    "UNKNOWN": 0.5,
}

# ---------------------------------------------------------------------------
# Risk limit checker
# ---------------------------------------------------------------------------


def check_live_risk_limits(
    portfolio_state: dict[str, Any],
    daily_pnl_pct: float,
) -> dict[str, Any]:
    """
    Evaluate all active daily, weekly, and monthly loss limits.

    Parameters
    ----------
    portfolio_state:
        Dict containing at minimum:
          - ``"weekly_pnl_pct"``   (float) — week-to-date P&L percentage
          - ``"monthly_pnl_pct"``  (float) — month-to-date P&L percentage
          - ``"consecutive_losses"`` (int) — current loss streak count
    daily_pnl_pct:
        Today's realised + unrealised P&L as a percentage of portfolio value.

    Returns
    -------
    dict with keys:
      - ``can_trade``  (bool)  — False if any hard stop has been breached
      - ``alerts``     (list[str])  — warning messages (soft limits)
      - ``actions``    (list[str])  — mandatory actions required
    """
    limits = LIVE_RISK_RULES["loss_limits"]
    can_trade = True
    alerts: list[str] = []
    actions: list[str] = []

    weekly_pnl: float = float(portfolio_state.get("weekly_pnl_pct", 0.0))
    monthly_pnl: float = float(portfolio_state.get("monthly_pnl_pct", 0.0))
    consecutive_losses: int = int(portfolio_state.get("consecutive_losses", 0))

    # --- Daily loss limit (hard stop) ----------------------------------------
    daily_limit = limits["daily_loss_limit_pct"]
    if daily_pnl_pct <= -daily_limit:
        can_trade = False
        actions.append(
            f"HARD STOP: daily loss {daily_pnl_pct:.2f}% breached "
            f"-{daily_limit}% limit — halt all trading for today."
        )
        logger.warning(
            "Daily loss limit breached: %.2f%% (limit: -%.1f%%)",
            daily_pnl_pct,
            daily_limit,
        )

    # --- Weekly loss limit (position reduction) --------------------------------
    weekly_limit = limits["weekly_loss_limit_pct"]
    if weekly_pnl <= -weekly_limit:
        alerts.append(
            f"Weekly loss {weekly_pnl:.2f}% breached -{weekly_limit}% threshold."
        )
        actions.append(
            "Reduce all open positions by 50% and avoid opening new ones "
            "until weekly loss recovers above the threshold."
        )
        logger.warning(
            "Weekly loss limit breached: %.2f%% (limit: -%.1f%%)",
            weekly_pnl,
            weekly_limit,
        )

    # --- Monthly loss limit (emergency stop) -----------------------------------
    monthly_limit = limits["monthly_loss_limit_pct"]
    if monthly_pnl <= -monthly_limit:
        can_trade = False
        actions.append(
            f"EMERGENCY STOP: monthly loss {monthly_pnl:.2f}% breached "
            f"-{monthly_limit}% limit — close all positions and pause strategy."
        )
        logger.error(
            "Monthly loss limit breached: %.2f%% (limit: -%.1f%%)",
            monthly_pnl,
            monthly_limit,
        )

    # --- Consecutive losses (review trigger) -----------------------------------
    max_streak = limits["max_consecutive_losses"]
    if consecutive_losses >= max_streak:
        alerts.append(
            f"Consecutive loss streak: {consecutive_losses} trades. "
            f"Threshold is {max_streak}."
        )
        actions.append(
            "Pause live trading and conduct a full strategy review before "
            "resuming. Check signal quality and regime conditions."
        )
        logger.warning(
            "Consecutive loss streak of %d reached (threshold: %d)",
            consecutive_losses,
            max_streak,
        )

    # --- Soft early warning at 50% of each limit -------------------------------
    if -daily_limit / 2 >= daily_pnl_pct > -daily_limit:
        alerts.append(
            f"Daily loss at {daily_pnl_pct:.2f}% — approaching "
            f"-{daily_limit}% hard stop."
        )
    if -weekly_limit / 2 >= weekly_pnl > -weekly_limit:
        alerts.append(
            f"Weekly loss at {weekly_pnl:.2f}% — approaching "
            f"-{weekly_limit}% reduction threshold."
        )

    return {"can_trade": can_trade, "alerts": alerts, "actions": actions}


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def compute_live_position_size(
    symbol: str,
    signal_score: float,
    regime: str,
    portfolio_value: float,
    current_positions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute a Kelly-inspired position size capped at LIVE_RISK_RULES limits.

    The formula uses a fractional Kelly approach:
      base_fraction = (signal_score - 0.5) * 2  [maps 0.5-1.0 → 0-1]
      kelly_pct     = base_fraction * max_per_stock_pct * 0.5  (half-Kelly)
      adjusted_pct  = kelly_pct * regime_multiplier

    Parameters
    ----------
    symbol:
        Stock ticker (e.g. ``"NABIL"``).
    signal_score:
        Model confidence in range [0, 1].  0.5 = no edge, 1.0 = maximum.
    regime:
        Market regime string — one of ``"BULL"``, ``"SIDEWAYS"``, ``"BEAR"``,
        or ``"UNKNOWN"``.
    portfolio_value:
        Current total portfolio value in NPR.
    current_positions:
        List of open position dicts, each with at least ``"symbol"`` and
        ``"value_npr"`` keys.

    Returns
    -------
    dict with keys:
      - ``qty``              (int)   — share quantity to buy (0 if blocked)
      - ``value_npr``        (float) — position value in NPR
      - ``pct_of_portfolio`` (float) — percentage of portfolio
      - ``rationale``        (str)   — human-readable sizing explanation
    """
    pos_rules = LIVE_RISK_RULES["position"]
    max_per_stock_pct: float = float(pos_rules["max_per_stock_pct"])
    max_open: int = int(pos_rules["max_open_positions"])
    min_value: float = float(pos_rules["min_position_size_npr"])
    max_value: float = float(pos_rules["max_position_size_npr"])

    _zero = {"qty": 0, "value_npr": 0.0, "pct_of_portfolio": 0.0, "rationale": ""}

    # --- Guard: maximum simultaneous positions ---------------------------------
    open_count = len(current_positions)
    already_in_symbol = any(
        p.get("symbol") == symbol for p in current_positions
    )
    if already_in_symbol:
        return {
            **_zero,
            "rationale": f"Already holding a position in {symbol} — skipping.",
        }
    if open_count >= max_open:
        return {
            **_zero,
            "rationale": (
                f"Maximum open positions ({max_open}) already reached — "
                "cannot add {symbol}."
            ),
        }

    if portfolio_value <= 0:
        return {**_zero, "rationale": "Portfolio value is zero or negative."}

    # --- Kelly fraction --------------------------------------------------------
    # signal_score below 0.5 implies negative edge — size is zero
    edge = max(0.0, signal_score - 0.5) * 2.0          # [0, 1]
    half_kelly_pct = edge * max_per_stock_pct * 0.5     # half-Kelly cap

    # Regime multiplier
    regime_key = regime.upper() if isinstance(regime, str) else "UNKNOWN"
    multiplier = _REGIME_MULTIPLIER.get(regime_key, _REGIME_MULTIPLIER["UNKNOWN"])
    adjusted_pct = half_kelly_pct * multiplier

    # --- Value in NPR ----------------------------------------------------------
    raw_value = portfolio_value * adjusted_pct / 100.0

    # Apply hard floors and ceilings
    if raw_value < min_value:
        # Below floor — either skip or bump up to minimum if edge is positive
        if edge > 0:
            raw_value = min_value
            adjusted_pct = min_value / portfolio_value * 100.0
            note = f"bumped to floor NPR {min_value:,.0f}"
        else:
            return {
                **_zero,
                "rationale": (
                    f"signal_score {signal_score:.2f} implies no edge — "
                    "position size is zero."
                ),
            }
    else:
        note = "Kelly-sized"

    if raw_value > max_value:
        raw_value = max_value
        adjusted_pct = max_value / portfolio_value * 100.0
        note = f"capped at ceiling NPR {max_value:,.0f}"

    # Placeholder qty — caller must divide by current market price
    # We cannot know price here, so we return value and note qty=0 sentinel
    rationale = (
        f"{symbol}: signal={signal_score:.2f}, edge={edge:.2f}, "
        f"regime={regime_key} (×{multiplier}), "
        f"half-Kelly={half_kelly_pct:.1f}% → adjusted={adjusted_pct:.1f}% "
        f"= NPR {raw_value:,.0f} ({note})"
    )

    logger.debug("Position size: %s", rationale)

    return {
        "qty": 0,           # Caller divides raw_value by current price
        "value_npr": round(raw_value, 2),
        "pct_of_portfolio": round(adjusted_pct, 2),
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Risk summary formatter
# ---------------------------------------------------------------------------


def format_risk_summary(portfolio_state: dict[str, Any]) -> str:
    """
    Build a Telegram-friendly text summary of current live risk metrics.

    Parameters
    ----------
    portfolio_state:
        Dict expected to contain (all optional — missing values shown as N/A):
          ``current_value``, ``starting_capital``, ``daily_pnl_pct``,
          ``weekly_pnl_pct``, ``monthly_pnl_pct``, ``max_drawdown_pct``,
          ``consecutive_losses``, ``open_positions`` (list).

    Returns
    -------
    Formatted multi-line string suitable for a Telegram message.
    """

    def _pct(key: str, default: str = "N/A") -> str:
        val = portfolio_state.get(key)
        if val is None:
            return default
        return f"{float(val):+.2f}%"

    def _npr(key: str, default: str = "N/A") -> str:
        val = portfolio_state.get(key)
        if val is None:
            return default
        return f"NPR {float(val):,.0f}"

    current_value = portfolio_state.get("current_value")
    starting_capital = portfolio_state.get("starting_capital")
    if current_value is not None and starting_capital and starting_capital > 0:
        total_return_pct = (current_value - starting_capital) / starting_capital * 100
        total_return_str = f"{total_return_pct:+.2f}%"
    else:
        total_return_str = "N/A"

    open_positions: list[dict] = portfolio_state.get("open_positions", [])
    pos_lines = ""
    if open_positions:
        pos_lines = "\n".join(
            f"  • {p.get('symbol', '?')}  "
            f"qty={p.get('qty', '?')}  "
            f"P&L={p.get('pnl_pct', 0.0):+.1f}%"
            for p in open_positions
        )
    else:
        pos_lines = "  (no open positions)"

    limits = LIVE_RISK_RULES["loss_limits"]
    summary = (
        "📊 *NEPSE MiroFish — Live Risk Summary*\n"
        "─────────────────────────────\n"
        f"Portfolio value : {_npr('current_value')}\n"
        f"Total return    : {total_return_str}\n"
        f"Daily P&L       : {_pct('daily_pnl_pct')}"
        f"  (limit: -{limits['daily_loss_limit_pct']}%)\n"
        f"Weekly P&L      : {_pct('weekly_pnl_pct')}"
        f"  (limit: -{limits['weekly_loss_limit_pct']}%)\n"
        f"Monthly P&L     : {_pct('monthly_pnl_pct')}"
        f"  (limit: -{limits['monthly_loss_limit_pct']}%)\n"
        f"Max drawdown    : {_pct('max_drawdown_pct')}\n"
        f"Loss streak     : {portfolio_state.get('consecutive_losses', 0)} trades\n"
        "─────────────────────────────\n"
        f"Open positions  : {len(open_positions)} / "
        f"{LIVE_RISK_RULES['position']['max_open_positions']}\n"
        f"{pos_lines}\n"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="NEPSE MiroFish — Live Risk Management Dry-Run Check"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a dry-run risk check with example portfolio state and print results.",
    )
    args = parser.parse_args()

    if not args.check:
        parser.print_help()
        sys.exit(0)

    # --- Example portfolio state for dry-run -----------------------------------
    example_state: dict[str, Any] = {
        "current_value": 510_000,
        "starting_capital": 500_000,
        "daily_pnl_pct": -1.5,
        "weekly_pnl_pct": -4.0,
        "monthly_pnl_pct": 2.0,
        "max_drawdown_pct": -5.2,
        "consecutive_losses": 2,
        "open_positions": [
            {"symbol": "NABIL", "qty": 50, "pnl_pct": 3.2},
            {"symbol": "UPPER", "qty": 100, "pnl_pct": -1.1},
        ],
    }

    print("\n=== Dry-run: check_live_risk_limits ===")
    risk_result = check_live_risk_limits(
        portfolio_state=example_state,
        daily_pnl_pct=example_state["daily_pnl_pct"],
    )
    print(json.dumps(risk_result, indent=2))

    print("\n=== Dry-run: compute_live_position_size ===")
    size_result = compute_live_position_size(
        symbol="NLIC",
        signal_score=0.72,
        regime="BULL",
        portfolio_value=example_state["current_value"],
        current_positions=example_state["open_positions"],
    )
    print(json.dumps(size_result, indent=2))

    print("\n=== Dry-run: format_risk_summary (Telegram preview) ===")
    print(format_risk_summary(example_state))


if __name__ == "__main__":
    _main()
