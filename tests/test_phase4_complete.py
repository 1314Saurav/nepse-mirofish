"""
tests/test_phase4_complete.py
Phase 4 sign-off integration tests.
Validates the complete backtesting layer without requiring live DB or API.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trades():
    """Realistic sample trades for testing metrics and Monte Carlo."""
    return [
        {"pnl_pct": 12.4, "action": "SELL", "regime": "BULL",     "sector": "banking",    "hold_days": 12, "mirofish_score": 0.72},
        {"pnl_pct": -3.1, "action": "SELL", "regime": "BULL",     "sector": "hydropower", "hold_days":  5, "mirofish_score": 0.35},
        {"pnl_pct":  8.2, "action": "SELL", "regime": "RECOVERY", "sector": "banking",    "hold_days": 18, "mirofish_score": 0.61},
        {"pnl_pct": -5.7, "action": "SELL", "regime": "BEAR",     "sector": "finance",    "hold_days":  7, "mirofish_score": -0.20},
        {"pnl_pct": 15.3, "action": "SELL", "regime": "BULL",     "sector": "hydropower", "hold_days": 22, "mirofish_score": 0.80},
        {"pnl_pct": -2.8, "action": "SELL", "regime": "SIDEWAYS", "sector": "banking",    "hold_days":  8, "mirofish_score": 0.15},
        {"pnl_pct":  9.8, "action": "SELL", "regime": "RECOVERY", "sector": "banking",    "hold_days": 14, "mirofish_score": 0.55},
        {"pnl_pct": -6.1, "action": "SELL", "regime": "BEAR",     "sector": "insurance",  "hold_days":  6, "mirofish_score": -0.40},
        {"pnl_pct":  5.7, "action": "SELL", "regime": "SIDEWAYS", "sector": "banking",    "hold_days": 10, "mirofish_score": 0.30},
        {"pnl_pct":  4.5, "action": "SELL", "regime": "BULL",     "sector": "hydropower", "hold_days": 16, "mirofish_score": 0.50},
    ] * 4  # 40 trades for statistical significance


@pytest.fixture
def sample_portfolio_values():
    """Realistic portfolio equity curve."""
    np.random.seed(42)
    initial = 1_000_000.0
    returns = np.random.normal(0.001, 0.01, 240)  # ~1yr of daily returns
    curve = [initial]
    for r in returns:
        curve.append(curve[-1] * (1 + r))
    return curve


@pytest.fixture
def sample_metrics(sample_portfolio_values, sample_trades):
    """Compute metrics from sample data."""
    from backtest.metrics import compute_metrics
    return compute_metrics(
        daily_portfolio_values=sample_portfolio_values,
        trades=sample_trades,
        nepse_index_values=[v * 0.95 for v in sample_portfolio_values],
    )


# ---------------------------------------------------------------------------
# Check 1: Look-ahead bias prevention
# ---------------------------------------------------------------------------

class TestLookAheadPrevention:
    """Verify look-ahead bias tests pass."""

    def test_no_future_data_in_price_cache(self):
        """PriceCache.get_history_up_to must never return future rows."""
        from backtest.engine import PriceCache
        import pandas as pd

        cache = PriceCache.__new__(PriceCache)
        cache._data = {}
        dates = pd.date_range("2024-01-07", periods=30, freq="B")
        prices = np.linspace(1000, 1100, 30)
        cache._data["TEST"] = pd.DataFrame(
            {"open": prices, "high": prices * 1.01, "low": prices * 0.99,
             "close": prices, "volume": [100_000] * 30},
            index=dates,
        )

        cutoff = dates[14].date()
        history = cache.get_history_up_to("TEST", cutoff, days=300)

        for idx in history.index:
            row_date = idx.date() if hasattr(idx, "date") else idx
            assert row_date <= cutoff


# ---------------------------------------------------------------------------
# Check 2: NEPSE calendar correctness
# ---------------------------------------------------------------------------

class TestNEPSECalendar:
    """Verify Sun-Thu trading week is enforced."""

    def test_trading_days_all_sun_to_thu(self):
        from backtest.calendar import get_trading_days
        days = get_trading_days("2024-01-01", "2024-03-31")
        assert len(days) > 0
        for d in days:
            assert d.weekday() not in {4, 5}, \
                f"Friday/Saturday in trading calendar: {d}"

    def test_approximately_240_days_per_year(self):
        from backtest.calendar import get_nepse_year_trading_days
        n = get_nepse_year_trading_days(2024)
        assert 200 <= n <= 265, f"Unexpected trading days: {n} (expected ~240)"

    def test_next_trading_day_after_thursday_is_sunday(self):
        from backtest.calendar import get_next_trading_day, is_trading_day
        # Find a Thursday
        test_date = date(2024, 1, 4)  # Thursday
        assert test_date.weekday() == 3  # Verify it's Thursday
        nxt = get_next_trading_day(test_date)
        assert nxt.weekday() not in {4, 5}, \
            f"Next trading day after Thu should not be Fri/Sat"
        assert is_trading_day(nxt), "Next trading day must be valid"


# ---------------------------------------------------------------------------
# Check 3: Performance metrics correctness
# ---------------------------------------------------------------------------

class TestMetricsComputation:
    """Core metrics must be mathematically correct."""

    def test_sharpe_positive_for_rising_portfolio(self, sample_portfolio_values):
        from backtest.metrics import compute_metrics
        rising = [1_000_000 * (1.001 ** i) for i in range(250)]
        metrics = compute_metrics(
            daily_portfolio_values=rising,
            trades=[],
            nepse_index_values=[1000 * (1.0005 ** i) for i in range(250)],
        )
        assert metrics["sharpe_ratio"] > 0, "Rising portfolio should have positive Sharpe"

    def test_max_drawdown_zero_for_monotone_rising(self):
        from backtest.metrics import _max_drawdown
        values = [100 + i for i in range(50)]  # strictly increasing
        dd, _ = _max_drawdown(values)
        assert dd == 0.0, "No drawdown in monotone rising series"

    def test_win_rate_correct(self, sample_trades):
        from backtest.metrics import _trade_stats
        stats = _trade_stats(sample_trades)
        wins = sum(1 for t in sample_trades if t.get("pnl_pct", 0) > 0)
        total = len([t for t in sample_trades if "pnl_pct" in t])
        expected_wr = wins / total * 100
        assert abs(stats["win_rate_pct"] - expected_wr) < 0.1

    def test_all_required_metric_keys_present(self, sample_metrics):
        required_keys = [
            "total_return_pct", "annualised_return_pct", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
            "win_rate_pct", "profit_factor", "total_trades",
        ]
        for key in required_keys:
            assert key in sample_metrics, f"Missing metric: {key}"


# ---------------------------------------------------------------------------
# Check 4: Deployment decision logic
# ---------------------------------------------------------------------------

class TestDeploymentDecision:
    """Decision logic must correctly classify DEPLOY/REVISE/REJECT."""

    def _good_metrics(self):
        return {
            "sharpe_ratio": 1.25,
            "max_drawdown_pct": 15.0,
            "alpha_pct": 6.5,
            "sortino_ratio": 1.5,
            "profit_factor": 1.6,
            "win_rate_pct": 58.0,
            "calmar_ratio": 1.1,
            "walk_forward": {"windows_passed": 5, "windows_total": 5},
            "stress_tests": {"passed": 5, "total": 5},
            "monte_carlo": {"p5_final_value": 1_050_000, "initial_capital": 1_000_000},
            "mirofish_accuracy": {"overall_accuracy_pct": 62.0},
        }

    def test_good_metrics_result_in_deploy(self):
        from backtest.deployment_decision import evaluate_deployment_readiness
        result = evaluate_deployment_readiness(self._good_metrics())
        assert result["decision"] == "DEPLOY", f"Expected DEPLOY, got {result['decision']}"
        assert result["required_passed"] == result["required_total"]

    def test_poor_sharpe_result_in_reject(self):
        from backtest.deployment_decision import evaluate_deployment_readiness
        metrics = self._good_metrics()
        metrics["sharpe_ratio"] = 0.5  # below threshold
        result = evaluate_deployment_readiness(metrics)
        assert result["decision"] == "REJECT"
        assert any("Sharpe" in f for f in result["blocking_failures"])

    def test_excessive_drawdown_result_in_reject(self):
        from backtest.deployment_decision import evaluate_deployment_readiness
        metrics = self._good_metrics()
        metrics["max_drawdown_pct"] = 28.0  # above threshold
        result = evaluate_deployment_readiness(metrics)
        assert result["decision"] == "REJECT"

    def test_all_required_pass_few_recommended_gives_revise(self):
        from backtest.deployment_decision import evaluate_deployment_readiness
        metrics = self._good_metrics()
        # Fail all recommended
        metrics["sortino_ratio"] = 0.5
        metrics["profit_factor"] = 1.1
        metrics["win_rate_pct"] = 45.0
        metrics["calmar_ratio"] = 0.4
        metrics["mirofish_accuracy"] = {"overall_accuracy_pct": 45.0}
        result = evaluate_deployment_readiness(metrics)
        assert result["decision"] in ("REVISE", "REJECT")


# ---------------------------------------------------------------------------
# Check 5: Monte Carlo smoke test
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    """Monte Carlo simulation produces valid output."""

    def test_monte_carlo_runs_with_sample_trades(self, sample_trades):
        from backtest.monte_carlo import run_monte_carlo
        result = run_monte_carlo(
            base_trades=sample_trades,
            n_simulations=100,    # small for test speed
            save_chart=False,
        )
        assert "median_final_value" in result
        assert "prob_of_loss" in result
        assert 0 <= result["prob_of_loss"] <= 100
        assert result["n_simulations"] == 100

    def test_prob_of_loss_zero_for_all_winning_trades(self):
        from backtest.monte_carlo import run_monte_carlo
        all_wins = [{"pnl_pct": 10.0, "action": "SELL"} for _ in range(20)]
        result = run_monte_carlo(all_wins, n_simulations=200, save_chart=False)
        assert result["prob_of_loss"] == pytest.approx(0.0, abs=1.0)

    def test_p5_less_than_median(self, sample_trades):
        from backtest.monte_carlo import run_monte_carlo
        result = run_monte_carlo(sample_trades, n_simulations=200, save_chart=False)
        assert result.get("p5_final_value", 0) <= result.get("median_final_value", 1e9)


# ---------------------------------------------------------------------------
# Check 6: Stress test framework
# ---------------------------------------------------------------------------

class TestStressTests:
    """Stress test framework runs without crashing."""

    def test_flash_crash_synthetic_runs(self):
        """The synthetic flash crash test runs purely on trading rules (no DB)."""
        from backtest.stress_tests import run_stress_test, STRESS_SCENARIOS
        flash_scenario = next(s for s in STRESS_SCENARIOS if s["id"] == "flash_crash")
        result = run_stress_test(flash_scenario)
        assert result.scenario_id == "flash_crash"
        assert isinstance(result.passed, bool)
        assert isinstance(result.portfolio_drawdown_pct, float)


# ---------------------------------------------------------------------------
# Check 7: Signal attribution structure
# ---------------------------------------------------------------------------

class TestSignalAttribution:
    """Attribution module has correct structure."""

    def test_attribution_components_defined(self):
        from backtest.signal_attribution import ATTRIBUTION_COMPONENTS
        required_components = {"mirofish_only", "technical_only", "sector_only", "full_hybrid"}
        assert required_components == set(ATTRIBUTION_COMPONENTS.keys())
        for comp_id, config in ATTRIBUTION_COMPONENTS.items():
            weights = config.get("weights", {})
            assert abs(sum(weights.values()) - 1.0) < 0.001 or all(w == 0 for w in weights.values()), \
                f"Weights for {comp_id} don't sum to 1: {weights}"


# ---------------------------------------------------------------------------
# Check 8: Agent optimiser
# ---------------------------------------------------------------------------

class TestAgentOptimiser:
    """Agent optimiser runs and returns valid result."""

    def test_default_mix_sums_to_1000(self):
        from backtest.agent_optimiser import DEFAULT_AGENT_MIX, TOTAL_AGENTS
        assert sum(DEFAULT_AGENT_MIX.values()) == TOTAL_AGENTS

    def test_accuracy_estimator_output_range(self):
        from backtest.agent_optimiser import _estimate_accuracy_for_mix, DEFAULT_AGENT_MIX
        acc = _estimate_accuracy_for_mix(DEFAULT_AGENT_MIX)
        assert 50.0 <= acc <= 90.0, f"Accuracy estimate out of range: {acc}"

    def test_optimise_returns_valid_mix(self):
        from backtest.agent_optimiser import optimise_agent_mix, TOTAL_AGENTS
        result = optimise_agent_mix(max_combinations=50)   # fast
        assert "optimal_mix" in result
        mix = result["optimal_mix"]
        total = sum(mix.values())
        assert total == TOTAL_AGENTS, f"Optimal mix total {total} != {TOTAL_AGENTS}"


# ---------------------------------------------------------------------------
# Final sign-off table
# ---------------------------------------------------------------------------

def test_phase4_signoff(sample_trades, sample_portfolio_values, sample_metrics):
    """Print Phase 4 sign-off status table."""
    from backtest.calendar import get_nepse_year_trading_days, get_trading_days
    from backtest.deployment_decision import evaluate_deployment_readiness
    from backtest.monte_carlo import run_monte_carlo
    from backtest.agent_optimiser import DEFAULT_AGENT_MIX, _estimate_accuracy_for_mix

    checks = []

    # 1. Look-ahead bias
    checks.append(("Look-ahead bias prevention", True, "Tested in TestLookAheadPrevention"))

    # 2. Calendar (count 2024 trading days)
    n_days = get_nepse_year_trading_days(2024)
    cov_pass = 200 <= n_days <= 265
    checks.append(("NEPSE calendar (Sun-Thu)", cov_pass, f"{n_days} trading days in 2024"))

    # 3. Metrics computed
    metrics_ok = "sharpe_ratio" in sample_metrics
    checks.append(("Performance metrics suite", metrics_ok,
                   f"Sharpe={sample_metrics.get('sharpe_ratio',0):.3f}"))

    # 4. Monte Carlo
    mc = run_monte_carlo(sample_trades, n_simulations=200, save_chart=False)
    mc_ok = "prob_of_loss" in mc and mc.get("n_simulations", 0) == 200
    checks.append(("Monte Carlo (200 simulations)", mc_ok,
                   f"P(loss)={mc.get('prob_of_loss',0):.1f}%"))

    # 5. Deployment decision
    good_metrics = {
        "sharpe_ratio": 1.25, "max_drawdown_pct": 15.0, "alpha_pct": 5.0,
        "sortino_ratio": 1.4, "profit_factor": 1.5, "win_rate_pct": 57.0,
        "calmar_ratio": 1.0,
        "walk_forward": {"windows_passed": 5, "windows_total": 5},
        "stress_tests": {"passed": 5, "total": 5},
        "monte_carlo": {"p5_final_value": 1_050_000, "initial_capital": 1_000_000},
        "mirofish_accuracy": {"overall_accuracy_pct": 60.0},
    }
    decision = evaluate_deployment_readiness(good_metrics)
    deploy_ok = decision["decision"] == "DEPLOY"
    checks.append(("Deployment decision logic", deploy_ok, decision["decision"]))

    # 6. Agent optimiser
    acc = _estimate_accuracy_for_mix(DEFAULT_AGENT_MIX)
    agent_ok = 50 <= acc <= 90
    checks.append(("Agent mix optimiser", agent_ok, f"Default accuracy={acc:.1f}%"))

    # 7. Signal attribution structure
    from backtest.signal_attribution import ATTRIBUTION_COMPONENTS
    attr_ok = len(ATTRIBUTION_COMPONENTS) == 4
    checks.append(("Signal attribution components", attr_ok,
                   f"{len(ATTRIBUTION_COMPONENTS)} components defined"))

    # Print sign-off table
    G, R, B = "\033[92m", "\033[91m", "\033[0m"
    print(f"\n{'='*65}")
    print(f"  Phase 4 -- Backtesting Sign-Off")
    print(f"{'='*65}")
    print(f"  {'Check':<45} {'Status'}")
    print(f"  {'-'*63}")
    all_passed = True
    for check_name, passed, detail in checks:
        status = f"{G}PASS{B}" if passed else f"{R}FAIL{B}"
        print(f"  {check_name:<45} {status}  ({detail})")
        all_passed = all_passed and passed
    print(f"{'='*65}")
    overall = f"{G}Phase 4 COMPLETE{B}" if all_passed else f"{R}Phase 4 INCOMPLETE{B}"
    print(f"  {overall}")
    print(f"{'='*65}\n")

    assert all_passed, "One or more Phase 4 checks failed"
