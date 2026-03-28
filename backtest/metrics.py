"""
backtest/metrics.py
Comprehensive performance metrics for NEPSE strategy evaluation.
Includes Sharpe, Sortino, Calmar, drawdown, regime/sector breakdowns,
MiroFish accuracy attribution, and colour-coded threshold display.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Deployment thresholds (used for colour-coded display)
# ---------------------------------------------------------------------------

REQUIRED_THRESHOLDS = {
    "sharpe_ratio":           (">=", 1.0),
    "max_drawdown_pct":       ("<=", 20.0),
    "alpha_pct":              (">",  0.0),
    "total_return_pct":       (">",  0.0),
}

RECOMMENDED_THRESHOLDS = {
    "sortino_ratio":          (">=", 1.3),
    "profit_factor":          (">=", 1.4),
    "win_rate_pct":           (">=", 50.0),
    "calmar_ratio":           (">=", 0.8),
    "mirofish_overall_accuracy_pct": (">=", 55.0),
}


def _passes(value: float, op: str, threshold: float) -> bool:
    ops = {">=": value >= threshold, "<=": value <= threshold,
           ">": value > threshold, "<": value < threshold, "==": value == threshold}
    return ops.get(op, False)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def _daily_returns(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.diff(arr) / arr[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return returns


def _cagr(initial: float, final: float, n_days: int) -> float:
    if initial <= 0 or n_days <= 0:
        return 0.0
    years = n_days / 252.0   # NEPSE ~240 trading days/year; use 252 for convention
    try:
        return ((final / initial) ** (1.0 / years) - 1.0) * 100.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _max_drawdown(values: list[float]) -> tuple[float, int]:
    """Returns (max_drawdown_pct, duration_in_days)."""
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return 0.0, 0

    peak = arr[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_duration = 0
    current_dd_start = 0

    for i in range(1, len(arr)):
        if arr[i] > peak:
            peak = arr[i]
            peak_idx = i
            current_dd_start = i
        else:
            dd = (peak - arr[i]) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = i - peak_idx

    return round(max_dd, 2), max_dd_duration


def _sharpe(daily_returns: np.ndarray, risk_free_annual: float = 0.07) -> float:
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = daily_returns - rf_daily
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(252))


def _sortino(daily_returns: np.ndarray, risk_free_annual: float = 0.07) -> float:
    if len(daily_returns) < 2:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float('inf')
    downside_dev = np.std(downside, ddof=1) * math.sqrt(252)
    if downside_dev == 0:
        return 0.0
    ann_return = (np.mean(daily_returns) * 252)
    return float((ann_return - risk_free_annual) / downside_dev)


def _beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    min_len = min(len(strategy_returns), len(benchmark_returns))
    if min_len < 2:
        return 1.0
    s = strategy_returns[:min_len]
    b = benchmark_returns[:min_len]
    b_var = np.var(b, ddof=1)
    if b_var == 0:
        return 1.0
    return float(np.cov(s, b, ddof=1)[0][1] / b_var)


# ---------------------------------------------------------------------------
# Trade statistics
# ---------------------------------------------------------------------------

def _trade_stats(trades: list[dict]) -> dict:
    """Compute win/loss stats from list of trade dicts with 'pnl_pct' key."""
    closed = [t for t in trades if t.get("action") == "SELL" or "pnl_pct" in t]
    if not closed:
        return {
            "total_trades": 0, "win_rate_pct": 0.0, "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0, "profit_factor": 0.0, "avg_hold_days": 0.0,
            "largest_win_pct": 0.0, "largest_loss_pct": 0.0,
            "consecutive_losses_max": 0,
        }

    pnls = [float(t.get("pnl_pct", 0)) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    hold_days = [float(t.get("hold_days", 0)) for t in closed]

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.001  # avoid division by zero

    # Consecutive losses
    max_consec = 0
    current_consec = 0
    for p in pnls:
        if p <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    return {
        "total_trades": len(closed),
        "win_rate_pct": round(len(wins) / len(pnls) * 100, 1),
        "avg_win_pct": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 2),
        "avg_hold_days": round(sum(hold_days) / len(hold_days), 1) if hold_days else 0.0,
        "largest_win_pct": round(max(pnls), 2) if pnls else 0.0,
        "largest_loss_pct": round(min(pnls), 2) if pnls else 0.0,
        "consecutive_losses_max": max_consec,
    }


def _regime_breakdown(trades: list[dict]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        regime = t.get("regime", "UNKNOWN")
        if "pnl_pct" in t:
            groups[regime].append(float(t["pnl_pct"]))

    breakdown = {}
    for regime, pnls in groups.items():
        wins = [p for p in pnls if p > 0]
        breakdown[regime] = {
            "trades": len(pnls),
            "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
            "avg_return": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        }
    return breakdown


def _sector_breakdown(trades: list[dict]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        sector = t.get("sector", "unknown")
        if "pnl_pct" in t:
            groups[sector].append(float(t["pnl_pct"]))

    breakdown = {}
    for sector, pnls in groups.items():
        wins = [p for p in pnls if p > 0]
        breakdown[sector] = {
            "trades": len(pnls),
            "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
            "avg_return": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        }
    return breakdown


def _mirofish_accuracy(trades: list[dict]) -> dict:
    """
    Accuracy: did the MiroFish signal direction match the trade outcome?
    bullish (score > 0.3) → stock rose (pnl_pct > 0) = correct
    bearish (score < -0.3) → stock fell (pnl_pct < 0) = correct (we avoided)
    """
    bullish_signals = [t for t in trades if float(t.get("mirofish_score", 0)) > 0.3
                       and "pnl_pct" in t]
    bearish_avoided = [t for t in trades if float(t.get("mirofish_score", 0)) < -0.3]

    bull_correct = sum(1 for t in bullish_signals if float(t["pnl_pct"]) > 0)
    bull_acc = bull_correct / len(bullish_signals) * 100 if bullish_signals else 0.0

    # Bearish accuracy: signal says avoid, and indeed the stock fell
    # We can only measure this if we have "avoided trade" records — approximate with trade data
    overall_correct = sum(1 for t in bullish_signals if float(t.get("pnl_pct", 0)) > 0)
    overall_total = len(bullish_signals)
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0.0

    return {
        "bullish_signals_correct_pct": round(bull_acc, 1),
        "bearish_signals_correct_pct": 0.0,   # requires avoided-trade log
        "overall_accuracy_pct": round(overall_acc, 1),
    }


def _capital_utilisation(daily_snapshots: list[dict]) -> float:
    """Average % of capital deployed (in open positions vs total equity)."""
    if not daily_snapshots:
        return 0.0
    utils = []
    for snap in daily_snapshots:
        total = snap.get("portfolio_value", 1)
        cash = snap.get("cash", total)
        deployed = (total - cash) / total * 100 if total > 0 else 0
        utils.append(deployed)
    return round(sum(utils) / len(utils), 1)


def _trades_per_month(trades: list[dict], n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    months = n_days / 21.0  # ~21 trading days/month
    return round(len(trades) / months, 1) if months > 0 else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    daily_portfolio_values: list[float],
    trades: list[dict],
    nepse_index_values: list[float],
    daily_snapshots: Optional[list[dict]] = None,
    risk_free_rate_annual: float = 0.07,
) -> dict:
    """
    Compute the full metrics suite.

    Parameters
    ----------
    daily_portfolio_values : portfolio equity for each trading day
    trades                 : list of trade dicts with pnl_pct, regime, sector, etc.
    nepse_index_values     : NEPSE index values aligned with portfolio_values
    daily_snapshots        : optional list of {portfolio_value, cash, open_positions}
    risk_free_rate_annual  : Nepal 91-day T-bill rate (~7%)
    """
    if len(daily_portfolio_values) < 2:
        return {"error": "insufficient data"}

    initial = daily_portfolio_values[0]
    final = daily_portfolio_values[-1]
    n_days = len(daily_portfolio_values) - 1

    strat_returns = _daily_returns(daily_portfolio_values)
    bench_returns = _daily_returns(nepse_index_values) if len(nepse_index_values) >= 2 else strat_returns * 0

    total_return = (final / initial - 1.0) * 100.0
    bench_initial = nepse_index_values[0] if nepse_index_values else initial
    bench_final = nepse_index_values[-1] if nepse_index_values else initial
    bench_return = (bench_final / bench_initial - 1.0) * 100.0

    ann_return = _cagr(initial, final, n_days)
    bench_ann = _cagr(bench_initial, bench_final, n_days)

    dd, dd_duration = _max_drawdown(daily_portfolio_values)
    vol_annual = float(np.std(strat_returns, ddof=1) * math.sqrt(252) * 100)
    downside = strat_returns[strat_returns < 0]
    downside_dev = float(np.std(downside, ddof=1) * math.sqrt(252)) if len(downside) > 1 else 0.001

    sharpe = _sharpe(strat_returns, risk_free_rate_annual)
    sortino = _sortino(strat_returns, risk_free_rate_annual)
    calmar = ann_return / dd if dd > 0 else (99.0 if ann_return > 0 else 0.0)
    beta = _beta(strat_returns, bench_returns)
    alpha = ann_return - (risk_free_rate_annual * 100 + beta * (bench_ann - risk_free_rate_annual * 100))

    t_stats = _trade_stats(trades)
    closed_trades = [t for t in trades if "pnl_pct" in t]

    metrics = {
        # Returns
        "total_return_pct": round(total_return, 2),
        "annualised_return_pct": round(ann_return, 2),
        "benchmark_return_pct": round(bench_return, 2),
        "benchmark_annualised_pct": round(bench_ann, 2),
        "alpha_pct": round(alpha, 2),
        "beta": round(beta, 3),
        # Risk
        "max_drawdown_pct": round(dd, 2),
        "max_drawdown_duration_days": dd_duration,
        "volatility_annual_pct": round(vol_annual, 2),
        "downside_deviation": round(downside_dev * 100, 2),
        # Risk-adjusted
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        # Trade stats
        **t_stats,
        # Strategy-specific
        "trades_per_month_avg": _trades_per_month(closed_trades, n_days),
        "capital_utilisation_avg_pct": _capital_utilisation(daily_snapshots or []),
        "regime_breakdown": _regime_breakdown(trades),
        "sector_breakdown": _sector_breakdown(trades),
        "mirofish_accuracy": _mirofish_accuracy(trades),
    }
    return metrics


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def print_metrics_table(metrics: dict, title: str = "Backtest Performance Metrics") -> None:
    """Print colour-coded metrics table to terminal."""
    print(f"\n{_BOLD}{'='*62}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'='*62}{_RESET}")

    sections = [
        ("RETURNS", [
            ("Total Return", "total_return_pct", "%"),
            ("Annualised Return (CAGR)", "annualised_return_pct", "%"),
            ("Benchmark (NEPSE) Return", "benchmark_return_pct", "%"),
            ("Alpha vs Benchmark", "alpha_pct", "pp"),
            ("Beta", "beta", ""),
        ]),
        ("RISK", [
            ("Max Drawdown", "max_drawdown_pct", "%"),
            ("Max Drawdown Duration", "max_drawdown_duration_days", " days"),
            ("Annual Volatility", "volatility_annual_pct", "%"),
            ("Downside Deviation", "downside_deviation", "%"),
        ]),
        ("RISK-ADJUSTED", [
            ("Sharpe Ratio", "sharpe_ratio", ""),
            ("Sortino Ratio", "sortino_ratio", ""),
            ("Calmar Ratio", "calmar_ratio", ""),
        ]),
        ("TRADE STATISTICS", [
            ("Total Trades", "total_trades", ""),
            ("Win Rate", "win_rate_pct", "%"),
            ("Average Win", "avg_win_pct", "%"),
            ("Average Loss", "avg_loss_pct", "%"),
            ("Profit Factor", "profit_factor", ""),
            ("Average Hold Duration", "avg_hold_days", " days"),
            ("Largest Win", "largest_win_pct", "%"),
            ("Largest Loss", "largest_loss_pct", "%"),
            ("Max Consecutive Losses", "consecutive_losses_max", ""),
        ]),
        ("STRATEGY", [
            ("Trades per Month", "trades_per_month_avg", ""),
            ("Capital Utilisation", "capital_utilisation_avg_pct", "%"),
        ]),
    ]

    for section_name, rows in sections:
        print(f"\n  {_BOLD}{section_name}{_RESET}")
        for label, key, unit in rows:
            value = metrics.get(key)
            if value is None:
                continue
            val_str = f"{value:+.2f}{unit}" if isinstance(value, float) else f"{value}{unit}"

            # Determine colour
            colour = _RESET
            if key in REQUIRED_THRESHOLDS:
                op, thresh = REQUIRED_THRESHOLDS[key]
                colour = _GREEN if _passes(float(value), op, thresh) else _RED
            elif key in RECOMMENDED_THRESHOLDS:
                op, thresh = RECOMMENDED_THRESHOLDS[key]
                colour = _GREEN if _passes(float(value), op, thresh) else _YELLOW

            print(f"    {label:<35} {colour}{val_str:>12}{_RESET}")

    # MiroFish accuracy
    mf = metrics.get("mirofish_accuracy", {})
    if mf:
        print(f"\n  {_BOLD}MIROFISH ACCURACY{_RESET}")
        acc = mf.get("overall_accuracy_pct", 0)
        acc_colour = _GREEN if acc >= 55 else _YELLOW if acc >= 50 else _RED
        print(f"    {'Overall Accuracy':<35} {acc_colour}{acc:>11.1f}%{_RESET}")
        print(f"    {'Bullish Signal Accuracy':<35} {mf.get('bullish_signals_correct_pct', 0):>11.1f}%")

    print(f"\n{_BOLD}{'='*62}{_RESET}\n")


def metrics_to_dataframe(metrics: dict) -> "pd.DataFrame":
    """Convert flat metrics dict to a DataFrame for export."""
    rows = []
    skip_keys = {"regime_breakdown", "sector_breakdown", "mirofish_accuracy"}
    for k, v in metrics.items():
        if k in skip_keys:
            continue
        rows.append({"metric": k, "value": v})
    return pd.DataFrame(rows)
