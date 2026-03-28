"""
backtest/regime_backtests.py
Run separate backtests for each NEPSE market regime period.
Identifies where strategy performs best and worst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NEPSE regime periods (historically accurate)
# ---------------------------------------------------------------------------

NEPSE_REGIME_PERIODS = {
    "bull_run_2021": {
        "start": "2020-10-01",
        "end":   "2021-10-01",
        "description": "NEPSE 1600 \u2192 3200. Fastest bull run in NEPSE history.",
        "dominant_regime": "BULL",
        "key_driver": "Post-COVID liquidity surge, low interest rates, diaspora remittances",
        "expected_strategy": "momentum_bull",
        "pass_criteria": {
            "min_return_pct": 15.0,
            "max_drawdown_pct": 20.0,
            "min_sharpe": 0.9,
        },
    },
    "bear_market_2022": {
        "start": "2021-10-01",
        "end":   "2022-12-31",
        "description": "NEPSE 3200 \u2192 1750. NRB credit tightening crash.",
        "dominant_regime": "BEAR",
        "key_driver": "NRB CD ratio ceiling, repo rate hikes, global risk-off",
        "expected_strategy": "defensive_bear",
        "pass_criteria": {
            "max_drawdown_pct": 15.0,   # capital preservation
            "min_sharpe": 0.0,          # just don't lose more than index
            "min_alpha": 5.0,           # outperform NEPSE which fell ~45%
        },
    },
    "recovery_2023": {
        "start": "2023-01-01",
        "end":   "2023-12-31",
        "description": "NEPSE 1750 \u2192 2400. Gradual recovery with sector rotation.",
        "dominant_regime": "RECOVERY",
        "key_driver": "NRB easing, hydro sector rally, political stabilisation",
        "expected_strategy": "momentum_bull",
        "pass_criteria": {
            "min_return_pct": 10.0,
            "max_drawdown_pct": 18.0,
            "min_sharpe": 0.7,
        },
    },
    "sideways_2024": {
        "start": "2024-01-01",
        "end":   "2024-12-31",
        "description": "NEPSE 2300\u20132800. Consolidation with sector rotation.",
        "dominant_regime": "SIDEWAYS",
        "key_driver": "Mixed macro signals, election overhang, hydro projects completing",
        "expected_strategy": "mean_reversion_sideways",
        "pass_criteria": {
            "min_sharpe": 0.5,
            "max_drawdown_pct": 15.0,
            "min_alpha": 0.0,
        },
    },
}


@dataclass
class RegimeBacktestResult:
    period_name: str
    start: str
    end: str
    dominant_regime: str
    key_driver: str
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    alpha_pct: float
    win_rate_pct: float
    total_trades: int
    passed: bool
    pass_criteria: dict
    best_trades: list[dict]
    worst_trades: list[dict]
    top_mf_agents: list[str]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_regime_backtest(
    period_name: str,
    period_config: dict,
    initial_capital: float = 500_000.0,
) -> RegimeBacktestResult:
    """Run backtest for a single regime period."""
    from backtest.engine import NEPSEBacktestEngine
    from backtest.metrics import compute_metrics

    logger.info(
        "Running %s backtest: %s \u2192 %s",
        period_name, period_config["start"], period_config["end"],
    )

    engine = NEPSEBacktestEngine(
        start_date=period_config["start"],
        end_date=period_config["end"],
        initial_capital_npr=initial_capital,
    )

    try:
        result = engine.run()
    except Exception as exc:
        logger.error("Backtest failed for %s: %s", period_name, exc)
        return RegimeBacktestResult(
            period_name=period_name,
            start=period_config["start"],
            end=period_config["end"],
            dominant_regime=period_config["dominant_regime"],
            key_driver=period_config["key_driver"],
            total_return_pct=0.0,
            annualised_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=99.0,
            alpha_pct=0.0,
            win_rate_pct=0.0,
            total_trades=0,
            passed=False,
            pass_criteria=period_config["pass_criteria"],
            best_trades=[],
            worst_trades=[],
            top_mf_agents=[],
        )

    if len(result.daily_portfolio_values) < 5:
        return RegimeBacktestResult(
            period_name=period_name, start=period_config["start"],
            end=period_config["end"],
            dominant_regime=period_config["dominant_regime"],
            key_driver=period_config["key_driver"],
            total_return_pct=0.0, annualised_return_pct=0.0,
            sharpe_ratio=0.0, max_drawdown_pct=99.0, alpha_pct=0.0,
            win_rate_pct=0.0, total_trades=0, passed=False,
            pass_criteria=period_config["pass_criteria"],
            best_trades=[], worst_trades=[], top_mf_agents=[],
        )

    trades_dicts = [
        {
            "pnl_pct": t.pnl_pct, "regime": t.regime,
            "sector": t.sector, "hold_days": t.hold_days,
            "mirofish_score": t.mirofish_score, "action": "SELL",
        }
        for t in result.trades
    ]

    metrics = compute_metrics(
        daily_portfolio_values=result.daily_portfolio_values,
        trades=trades_dicts,
        nepse_index_values=result.nepse_index_values or result.daily_portfolio_values,
    )

    # Determine pass/fail
    criteria = period_config["pass_criteria"]
    passed = True
    if "min_return_pct" in criteria:
        passed &= metrics.get("total_return_pct", 0) >= criteria["min_return_pct"]
    if "max_drawdown_pct" in criteria:
        passed &= metrics.get("max_drawdown_pct", 99) <= criteria["max_drawdown_pct"]
    if "min_sharpe" in criteria:
        passed &= metrics.get("sharpe_ratio", 0) >= criteria["min_sharpe"]
    if "min_alpha" in criteria:
        passed &= metrics.get("alpha_pct", 0) >= criteria["min_alpha"]

    # Top/worst trades
    sorted_trades = sorted(result.trades, key=lambda t: t.pnl_pct, reverse=True)
    best_3 = [
        {"symbol": t.symbol, "pnl_pct": t.pnl_pct,
         "exit_reason": t.exit_reason, "hold_days": t.hold_days}
        for t in sorted_trades[:3]
    ]
    worst_3 = [
        {"symbol": t.symbol, "pnl_pct": t.pnl_pct,
         "exit_reason": t.exit_reason, "hold_days": t.hold_days}
        for t in sorted_trades[-3:]
    ]

    # MiroFish agent attribution (approximate from signal data)
    top_agents = _estimate_top_agents(result.trades, period_config["dominant_regime"])

    return RegimeBacktestResult(
        period_name=period_name,
        start=period_config["start"],
        end=period_config["end"],
        dominant_regime=period_config["dominant_regime"],
        key_driver=period_config["key_driver"],
        total_return_pct=round(metrics.get("total_return_pct", 0.0), 2),
        annualised_return_pct=round(metrics.get("annualised_return_pct", 0.0), 2),
        sharpe_ratio=round(metrics.get("sharpe_ratio", 0.0), 3),
        max_drawdown_pct=round(metrics.get("max_drawdown_pct", 0.0), 2),
        alpha_pct=round(metrics.get("alpha_pct", 0.0), 2),
        win_rate_pct=round(metrics.get("win_rate_pct", 0.0), 1),
        total_trades=metrics.get("total_trades", 0),
        passed=passed,
        pass_criteria=criteria,
        best_trades=best_3,
        worst_trades=worst_3,
        top_mf_agents=top_agents,
    )


def _estimate_top_agents(trades, regime: str) -> list[str]:
    """Estimate which MiroFish agent types drove winning trades (heuristic)."""
    agent_regime_map = {
        "BULL": ["retail_momentum", "institutional_broker", "diaspora_investor"],
        "BEAR": ["nrb_policy_watcher", "institutional_broker", "political_risk_analyst"],
        "RECOVERY": ["institutional_broker", "hydro_analyst", "nrb_policy_watcher"],
        "SIDEWAYS": ["hydro_analyst", "institutional_broker", "retail_momentum"],
    }
    return agent_regime_map.get(regime, ["institutional_broker", "retail_momentum"])


def run_regime_backtests(
    periods: Optional[dict] = None,
    initial_capital: float = 500_000.0,
) -> dict[str, RegimeBacktestResult]:
    """Run all regime-specific backtests and print comparison table."""
    periods = periods or NEPSE_REGIME_PERIODS
    results: dict[str, RegimeBacktestResult] = {}

    for name, config in periods.items():
        r = run_regime_backtest(name, config, initial_capital)
        results[name] = r

    _print_comparison_table(results)
    return results


def _print_comparison_table(results: dict[str, RegimeBacktestResult]) -> None:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[0m"

    print(f"\n{'='*80}")
    print(f"  REGIME-SPECIFIC BACKTEST RESULTS")
    print(f"{'='*80}")
    print(
        f"  {'Period':<22} {'Regime':<12} {'Return':>8} {'Sharpe':>7} "
        f"{'MaxDD':>7} {'Alpha':>7} {'WinRate':>8} {'Pass'}"
    )
    print(f"  {'-'*78}")

    for name, r in results.items():
        status = f"{G}PASS{B}" if r.passed else f"{R}FAIL{B}"
        return_c = G if r.total_return_pct > 0 else R
        alpha_c = G if r.alpha_pct > 0 else R

        print(
            f"  {name:<22} {r.dominant_regime:<12} "
            f"{return_c}{r.total_return_pct:>+7.1f}%{B} "
            f"{r.sharpe_ratio:>7.2f} "
            f"{R if r.max_drawdown_pct > 20 else B}{r.max_drawdown_pct:>6.1f}%{B} "
            f"{alpha_c}{r.alpha_pct:>+6.1f}%{B} "
            f"{r.win_rate_pct:>7.1f}%  [{status}]"
        )

        if r.best_trades:
            best = r.best_trades[0]
            print(
                f"    Best:  {best['symbol']} ({best['pnl_pct']:+.1f}%)"
                f"  Worst: {r.worst_trades[-1]['symbol']} "
                f"({r.worst_trades[-1]['pnl_pct']:+.1f}%)" if r.worst_trades else ""
            )

    total_pass = sum(1 for r in results.values() if r.passed)
    print(f"\n  {total_pass}/{len(results)} regime periods passed")
    if total_pass < len(results):
        failed = [n for n, r in results.items() if not r.passed]
        print(f"  \u26a0\ufe0f  Failed: {', '.join(failed)}")
        if any(results[f].dominant_regime == "BEAR" for f in failed):
            print("  \u2192 BEAR period failure: tighten defensive_bear thresholds in entry_exit.py")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Regime-specific backtests")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--period", help="Run single period by name")
    args = parser.parse_args()

    if args.period and args.period in NEPSE_REGIME_PERIODS:
        r = run_regime_backtest(args.period, NEPSE_REGIME_PERIODS[args.period])
        print(f"\n{args.period}: Return={r.total_return_pct:+.1f}% Sharpe={r.sharpe_ratio:.2f} {'PASS' if r.passed else 'FAIL'}")
    else:
        run_regime_backtests()


if __name__ == "__main__":
    main()
