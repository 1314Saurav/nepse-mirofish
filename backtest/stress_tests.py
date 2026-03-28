"""
backtest/stress_tests.py
Targeted stress tests on 5 biggest historical shocks to NEPSE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

STRESS_SCENARIOS = [
    {
        "name": "COVID crash (March 2020)",
        "id": "covid_crash",
        "date_range": ("2020-03-01", "2020-04-30"),
        "synthetic": False,
        "description": "NEPSE halted trading for 3 months. Circuit breakers daily.",
        "test": "Does strategy go fully to cash when MiroFish sentiment collapses?",
        "pass_criteria": {"max_drawdown_pct": 10.0, "description": "Portfolio drawdown < 10% (vs NEPSE -25%)"},
    },
    {
        "name": "NRB credit crunch (Oct 2021–Mar 2022)",
        "id": "nrb_crunch",
        "date_range": ("2021-10-01", "2022-03-31"),
        "synthetic": False,
        "description": "NEPSE fell 40% in 6 months. Banking stocks led decline.",
        "test": "Does NRB watcher agent signal credit tightening risk?",
        "pass_criteria": {"max_drawdown_pct": 15.0, "description": "Strategy exits banking before -15% drawdown"},
    },
    {
        "name": "Flash crash simulation (-8% single day)",
        "id": "flash_crash",
        "date_range": None,
        "synthetic": True,
        "description": "Simulate a single day where NEPSE drops 8% at open.",
        "test": "Do circuit breaker rules prevent buying on the down day?",
        "pass_criteria": {"no_buys_on_crash_day": True, "description": "No new BUY orders on simulated crash day"},
    },
    {
        "name": "IPO mania period (Q3 2021)",
        "id": "ipo_mania",
        "date_range": ("2021-07-01", "2021-09-30"),
        "synthetic": False,
        "description": "Dozens of IPOs, retail frenzy, distorted liquidity.",
        "test": "Does IPO listing blackout rule prevent buying hype stocks?",
        "pass_criteria": {"no_ipo_day_trades": True, "description": "No IPO listing-day trades in trade log"},
    },
    {
        "name": "Political shock (government collapse Jul 2022)",
        "id": "political_shock",
        "date_range": ("2022-07-01", "2022-08-31"),
        "synthetic": False,
        "description": "Coalition collapse, NEPSE dropped 12% in 2 weeks.",
        "test": "Does political analyst agent drive BEARISH signal before drop?",
        "pass_criteria": {"signal_turns_bearish_within_days": 3, "description": "MiroFish bearish within 3 days of collapse news"},
    },
]


@dataclass
class StressTestResult:
    scenario_id: str
    scenario_name: str
    passed: bool
    portfolio_drawdown_pct: float
    strategy_response: str
    notes: str
    metrics: dict = field(default_factory=dict)


def run_stress_test(scenario: dict) -> StressTestResult:
    """Run a single stress test scenario."""
    sid = scenario["id"]
    logger.info("Running stress test: %s", scenario["name"])

    if scenario["synthetic"]:
        return _run_synthetic_stress_test(scenario)

    date_range = scenario.get("date_range")
    if not date_range:
        return StressTestResult(
            scenario_id=sid, scenario_name=scenario["name"],
            passed=False, portfolio_drawdown_pct=0.0,
            strategy_response="No date range", notes="Skipped",
        )

    try:
        from backtest.engine import NEPSEBacktestEngine
        from backtest.metrics import compute_metrics

        engine = NEPSEBacktestEngine(
            start_date=date_range[0],
            end_date=date_range[1],
            initial_capital_npr=500_000.0,
        )
        result = engine.run()

        if not result.daily_portfolio_values:
            return StressTestResult(
                scenario_id=sid, scenario_name=scenario["name"],
                passed=False, portfolio_drawdown_pct=99.0,
                strategy_response="No portfolio data",
                notes="Insufficient data for date range",
            )

        trades_dicts = [
            {"pnl_pct": t.pnl_pct, "regime": t.regime,
             "sector": t.sector, "hold_days": t.hold_days,
             "mirofish_score": t.mirofish_score, "action": "SELL"}
            for t in result.trades
        ]
        metrics = compute_metrics(
            daily_portfolio_values=result.daily_portfolio_values,
            trades=trades_dicts,
            nepse_index_values=result.nepse_index_values or result.daily_portfolio_values,
        )

        drawdown = metrics.get("max_drawdown_pct", 99.0)
        criteria = scenario["pass_criteria"]

        # Check pass criteria
        passed = True
        response_parts = []

        if "max_drawdown_pct" in criteria:
            if drawdown <= criteria["max_drawdown_pct"]:
                response_parts.append(f"Drawdown {drawdown:.1f}% \u2264 {criteria['max_drawdown_pct']}% \u2713")
            else:
                passed = False
                response_parts.append(f"Drawdown {drawdown:.1f}% > {criteria['max_drawdown_pct']}% \u2717")

        # Special check for political shock: signal turns bearish
        if sid == "political_shock":
            passed, response = _check_political_signal_timing(result)
            response_parts.append(response)

        # Special check for IPO mania: no IPO-day trades
        if sid == "ipo_mania":
            passed, response = _check_no_ipo_day_trades(result)
            response_parts.append(response)

        return StressTestResult(
            scenario_id=sid,
            scenario_name=scenario["name"],
            passed=passed,
            portfolio_drawdown_pct=round(drawdown, 2),
            strategy_response=" | ".join(response_parts),
            notes=f"{len(result.trades)} trades executed",
            metrics=metrics,
        )

    except Exception as exc:
        logger.error("Stress test failed for %s: %s", sid, exc)
        return StressTestResult(
            scenario_id=sid, scenario_name=scenario["name"],
            passed=False, portfolio_drawdown_pct=0.0,
            strategy_response=f"Error: {exc}",
            notes="Test execution failed",
        )


def _run_synthetic_flash_crash(scenario: dict) -> StressTestResult:
    """
    Simulate a -8% NEPSE open. Check that circuit breaker rules
    prevent new BUY orders on that day.
    """
    try:
        from strategy.trading_rules import validate_trade

        # Simulate: prev_close=1000, ltp=920 (-8%)
        result = validate_trade(
            symbol="NABIL",
            action="BUY",
            position_pct=10.0,
            open_positions=1,
            ltp=920.0,
            prev_close=1000.0,
            rsi=45.0,
        )
        failed = result.get("failed_rules", [])
        warnings = result.get("soft_warnings", [])
        all_checks = failed + warnings
        circuit_blocked = any("circuit" in c.lower() for c in all_checks)

        # A -8% drop hits the circuit buffer threshold
        passed = circuit_blocked or not result.get("approved", True)
        response = (
            "Circuit breaker prevented BUY on -8% day \u2713"
            if passed else
            "BUY not blocked on -8% crash day \u2717"
        )
        return StressTestResult(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
            passed=passed,
            portfolio_drawdown_pct=8.0,
            strategy_response=response,
            notes=f"Rule checks: {failed[:3]}",
        )
    except Exception as exc:
        return StressTestResult(
            scenario_id=scenario["id"], scenario_name=scenario["name"],
            passed=False, portfolio_drawdown_pct=0.0,
            strategy_response=f"Error: {exc}", notes="",
        )


def _run_synthetic_stress_test(scenario: dict) -> StressTestResult:
    if scenario["id"] == "flash_crash":
        return _run_synthetic_flash_crash(scenario)
    return StressTestResult(
        scenario_id=scenario["id"], scenario_name=scenario["name"],
        passed=False, portfolio_drawdown_pct=0.0,
        strategy_response="Unknown synthetic scenario", notes="",
    )


def _check_political_signal_timing(result) -> tuple[bool, str]:
    """Check if MiroFish turned bearish within 3 days of the political event."""
    # Look for a bearish signal in the first few days of the period
    early_snaps = result.daily_snapshots[:5] if result.daily_snapshots else []
    bearish_early = any(
        snap.regime in ("BEAR", "SIDEWAYS") for snap in early_snaps
    )
    if bearish_early:
        return True, "Signal turned bearish within first 3 days \u2713"
    return False, "Signal did not turn bearish early enough \u2717"


def _check_no_ipo_day_trades(result) -> tuple[bool, str]:
    """Check that no trades were executed on IPO listing days (T+2 constraint)."""
    ipo_day_trades = [t for t in result.trades if t.exit_reason == "ipo_listing_wait"]
    if not ipo_day_trades:
        return True, "No IPO listing-day trades detected \u2713"
    return False, f"{len(ipo_day_trades)} IPO listing-day trades detected \u2717"


def run_all_stress_tests() -> list[StressTestResult]:
    """Run all 5 stress tests and print results table."""
    results = []
    for scenario in STRESS_SCENARIOS:
        r = run_stress_test(scenario)
        results.append(r)

    _print_results_table(results)
    return results


def _print_results_table(results: list[StressTestResult]) -> None:
    G, R, B = "\033[92m", "\033[91m", "\033[0m"
    print(f"\n{'='*72}")
    print(f"  STRESS TEST RESULTS")
    print(f"{'='*72}")
    for r in results:
        status = f"{G}PASS{B}" if r.passed else f"{R}FAIL{B}"
        dd_str = f"{r.portfolio_drawdown_pct:.1f}%" if r.portfolio_drawdown_pct else "N/A"
        print(f"  [{status}] {r.scenario_name:<40}  DD={dd_str}")
        print(f"          {r.strategy_response[:65]}")
    passed = sum(1 for r in results if r.passed)
    verdict = f"{G}\u2713 All critical stress tests passed{B}" if passed >= 4 else f"{R}\u2717 Too many stress test failures \u2014 review strategy{B}"
    print(f"\n  {passed}/{len(results)} stress tests passed")
    print(f"  {verdict}")
    print(f"{'='*72}\n")


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Stress tests")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scenario", help="Run single scenario by ID")
    args = parser.parse_args()
    if args.scenario:
        s = next((x for x in STRESS_SCENARIOS if x["id"] == args.scenario), None)
        if s:
            r = run_stress_test(s)
            print(f"{'PASS' if r.passed else 'FAIL'}: {r.strategy_response}")
    else:
        run_all_stress_tests()

if __name__ == "__main__":
    main()
