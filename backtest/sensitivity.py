"""
backtest/sensitivity.py
Parameter sensitivity analysis for NEPSE MiroFish strategy.
Tests how sensitive the strategy is to key parameters.
Small parameter changes should not cause large performance swings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENSITIVITY_TESTS = [
    {
        "parameter": "bull_bear_entry_threshold",
        "description": "MiroFish score threshold to trigger BUY",
        "base_value": 0.55,
        "test_range": [0.35, 0.45, 0.55, 0.65, 0.75],
        "metric": "sharpe_ratio",
        "acceptable_variance": 0.25,
        "higher_is_better": True,
    },
    {
        "parameter": "stop_loss_pct_momentum_bull",
        "description": "Stop loss % for momentum strategy",
        "base_value": 0.08,
        "test_range": [0.05, 0.06, 0.08, 0.10, 0.12],
        "metric": "max_drawdown_pct",
        "acceptable_variance": 5.0,
        "higher_is_better": False,
    },
    {
        "parameter": "mirofish_weight",
        "description": "MiroFish signal weight in composite score",
        "base_value": 0.40,
        "test_range": [0.25, 0.32, 0.40, 0.48, 0.55],
        "metric": "sharpe_ratio",
        "acceptable_variance": 0.20,
        "higher_is_better": True,
    },
    {
        "parameter": "min_daily_turnover_npr",
        "description": "Minimum liquidity threshold for entry",
        "base_value": 10_000_000,
        "test_range": [5_000_000, 7_500_000, 10_000_000, 15_000_000, 20_000_000],
        "metric": "total_trades",
        "acceptable_variance": 20.0,
        "higher_is_better": True,
        "note": "Higher threshold = fewer trades. Ensure still > 30 trades/year.",
    },
    {
        "parameter": "rsi_bull_threshold",
        "description": "RSI threshold for bull entry (prevent overbought buys)",
        "base_value": 68.0,
        "test_range": [60.0, 65.0, 68.0, 72.0, 75.0],
        "metric": "win_rate_pct",
        "acceptable_variance": 8.0,
        "higher_is_better": True,
    },
]

TEST_DATE_RANGE = ("2023-01-01", "2024-06-30")   # 18-month validation period


@dataclass
class SensitivityResult:
    parameter: str
    description: str
    base_value: float
    test_range: list
    metric: str
    metric_values: list[float] = field(default_factory=list)
    variance: float = 0.0
    acceptable_variance: float = 0.0
    passed: bool = True
    flag_overfitted: bool = False


def _run_backtest_with_param(param: str, value: float) -> dict:
    """
    Run a lightweight backtest with a single parameter overridden.
    Returns metrics dict.
    """
    try:
        from backtest.engine import NEPSEBacktestEngine
        from backtest.metrics import compute_metrics

        strategy_config = {param: value}
        engine = NEPSEBacktestEngine(
            start_date=TEST_DATE_RANGE[0],
            end_date=TEST_DATE_RANGE[1],
            initial_capital_npr=500_000.0,
            strategy_config=strategy_config,
        )
        result = engine.run()

        if len(result.daily_portfolio_values) < 5:
            return {"sharpe_ratio": 0.0, "max_drawdown_pct": 99.0,
                    "win_rate_pct": 0.0, "total_trades": 0}

        trades_dicts = [
            {"pnl_pct": t.pnl_pct, "regime": t.regime, "sector": t.sector,
             "hold_days": t.hold_days, "mirofish_score": t.mirofish_score, "action": "SELL"}
            for t in result.trades
        ]
        return compute_metrics(
            daily_portfolio_values=result.daily_portfolio_values,
            trades=trades_dicts,
            nepse_index_values=result.nepse_index_values or result.daily_portfolio_values,
        )
    except Exception as exc:
        logger.debug("Sensitivity backtest failed (%s=%s): %s", param, value, exc)
        return {"sharpe_ratio": 0.0, "max_drawdown_pct": 99.0,
                "win_rate_pct": 0.0, "total_trades": 0}


def run_sensitivity_analysis(
    tests: Optional[list[dict]] = None,
    save_chart: bool = True,
    output_dir: str = "data/processed/backtest",
) -> pd.DataFrame:
    """
    For each parameter and each test value, run a backtest and record the metric.
    Returns a DataFrame of results.
    """
    tests = tests or SENSITIVITY_TESTS
    rows = []
    results: list[SensitivityResult] = []

    print(f"\n{'='*65}")
    print(f"  Sensitivity Analysis ({TEST_DATE_RANGE[0]} \u2192 {TEST_DATE_RANGE[1]})")
    print(f"{'='*65}")

    for test in tests:
        param = test["parameter"]
        metric = test["metric"]
        acceptable = test["acceptable_variance"]
        test_values = test["test_range"]

        logger.info("Testing parameter: %s", param)
        print(f"\n  Parameter: {param} ({test['description']})")
        print(f"  Metric: {metric} | Acceptable variance: \u00b1{acceptable}")

        metric_values = []
        for val in test_values:
            m = _run_backtest_with_param(param, val)
            metric_val = float(m.get(metric, 0.0))
            metric_values.append(metric_val)

            base_marker = " \u2190 BASE" if abs(val - test["base_value"]) < 1e-9 else ""
            print(f"    {param}={val:<15}  {metric}={metric_val:>8.3f}{base_marker}")

            rows.append({
                "parameter": param,
                "value": val,
                "metric": metric,
                "metric_value": metric_val,
                "is_base": abs(val - test["base_value"]) < 1e-9,
            })

        # Compute variance
        if metric_values:
            variance = float(np.max(metric_values) - np.min(metric_values))
            passed = variance <= acceptable
            flag = not passed

            colour = "\033[92m" if passed else "\033[91m"
            reset = "\033[0m"
            print(
                f"  \u2192 Variance: {colour}{variance:.3f}{reset}"
                f"  (acceptable: \u2264{acceptable})  "
                f"{'\u2713 OK' if passed else '\u2717 OVERFITTED \u2014 use conservative default'}"
            )

            results.append(SensitivityResult(
                parameter=param,
                description=test["description"],
                base_value=test["base_value"],
                test_range=test_values,
                metric=metric,
                metric_values=metric_values,
                variance=variance,
                acceptable_variance=acceptable,
                passed=passed,
                flag_overfitted=flag,
            ))

    df = pd.DataFrame(rows)

    if save_chart:
        _save_sensitivity_charts(results, output_dir)

    _print_summary(results)
    return df


def _save_sensitivity_charts(results: list[SensitivityResult], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        n = len(results)
        cols = min(2, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
        fig.patch.set_facecolor("#0d1117")

        if n == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]

        flat_axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

        for i, r in enumerate(results):
            ax = flat_axes[i] if i < len(flat_axes) else None
            if ax is None:
                continue
            ax.set_facecolor("#161b22")
            colour = "#3fb950" if r.passed else "#f85149"
            ax.plot(r.test_range, r.metric_values, "o-", color=colour, linewidth=2)
            ax.axvline(x=r.base_value, color="#d29922", linestyle="--",
                       alpha=0.7, label="Base value")
            ax.set_title(f"{r.parameter}\n(\u0394={r.variance:.3f}, limit={r.acceptable_variance})",
                         color="#c9d1d9", fontsize=9)
            ax.set_xlabel(r.parameter, color="#8b949e", fontsize=8)
            ax.set_ylabel(r.metric, color="#8b949e", fontsize=8)
            ax.tick_params(colors="#8b949e", labelsize=7)
            ax.spines[:].set_color("#30363d")
            ax.legend(facecolor="#161b22", labelcolor="#c9d1d9", fontsize=7)

        for i in range(n, len(flat_axes)):
            flat_axes[i].set_visible(False)

        plt.suptitle("Parameter Sensitivity Analysis \u2014 NEPSE MiroFish",
                     color="#c9d1d9", fontsize=12)
        plt.tight_layout()
        path = Path(output_dir) / "sensitivity_charts.png"
        plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()
        logger.info("Sensitivity charts saved \u2192 %s", path)
        print(f"\n  Charts saved \u2192 {path}")
    except Exception as exc:
        logger.warning("Sensitivity chart failed: %s", exc)


def _print_summary(results: list[SensitivityResult]) -> None:
    overfitted = [r for r in results if r.flag_overfitted]
    print(f"\n{'='*65}")
    print(f"  Sensitivity Summary")
    print(f"{'='*65}")
    passed = len(results) - len(overfitted)
    print(f"  {passed}/{len(results)} parameters within acceptable variance bounds")
    if overfitted:
        print(f"\n  \u26a0\ufe0f  Potentially overfitted parameters:")
        for r in overfitted:
            print(f"     - {r.parameter}: variance={r.variance:.3f} > {r.acceptable_variance}")
            print(f"       \u2192 Use a more conservative default value")
    else:
        print(f"\n  \u2713 All parameters show robust behaviour")
    print(f"{'='*65}\n")


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Sensitivity analysis")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="data/processed/backtest")
    args = parser.parse_args()
    df = run_sensitivity_analysis(output_dir=args.output)
    path = Path(args.output) / "sensitivity_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Results saved \u2192 {path}")

if __name__ == "__main__":
    main()
