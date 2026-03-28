"""
backtest/monte_carlo.py
Bootstrap Monte Carlo simulation to understand range of possible outcomes.
5000 simulations by resampling actual historical trade returns.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _compute_equity_curve(returns: list[float], initial: float = 1.0) -> list[float]:
    """Compute equity curve from list of percentage returns."""
    curve = [initial]
    for r in returns:
        curve.append(curve[-1] * (1 + r / 100.0))
    return curve


def _max_drawdown_fast(curve: list[float]) -> float:
    """Fast max drawdown computation for MC loop."""
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe_fast(returns: list[float], rf_annual: float = 0.07) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns) / 100.0
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = arr - rf_daily
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def run_monte_carlo(
    base_trades: list[dict],
    n_simulations: int = 5000,
    initial_capital: float = 1_000_000.0,
    confidence_intervals: Optional[list[float]] = None,
    save_chart: bool = True,
    output_dir: str = "data/processed/backtest",
) -> dict:
    """
    Bootstrap actual trade returns to generate n_simulations equity curves.
    Tests whether strategy performance is skill (consistent) or luck (random).

    Parameters
    ----------
    base_trades     : actual historical trades from BacktestResult
    n_simulations   : number of Monte Carlo paths (default 5000)
    initial_capital : starting portfolio value in NPR
    confidence_intervals : percentiles to report (default: [0.05,0.25,0.50,0.75,0.95])
    """
    confidence_intervals = confidence_intervals or [0.05, 0.25, 0.50, 0.75, 0.95]

    # Extract trade returns
    trade_returns = [float(t.get("pnl_pct", 0)) for t in base_trades if "pnl_pct" in t]

    if len(trade_returns) < 5:
        logger.warning("Insufficient trades for Monte Carlo (%d trades)", len(trade_returns))
        return {
            "error": "insufficient_trades",
            "n_trades": len(trade_returns),
            "median_final_value": initial_capital,
            "prob_of_loss": 0.5,
        }

    n_trades = len(trade_returns)
    logger.info(
        "Running Monte Carlo: %d simulations \u00d7 %d trades (bootstrap)",
        n_simulations, n_trades,
    )

    # Run simulations
    final_values: list[float] = []
    max_drawdowns: list[float] = []
    sharpe_ratios: list[float] = []
    equity_curves: list[list[float]] = []

    # Store fewer curves for plotting (memory efficiency)
    store_curves_n = min(500, n_simulations)

    for i in range(n_simulations):
        # Resample with replacement
        sampled = random.choices(trade_returns, k=n_trades)
        curve = _compute_equity_curve(sampled, initial=initial_capital)

        final_values.append(curve[-1])
        max_drawdowns.append(_max_drawdown_fast(curve))
        sharpe_ratios.append(_sharpe_fast(sampled))

        if i < store_curves_n:
            equity_curves.append(curve)

    # Compute statistics
    final_arr = np.array(final_values)
    dd_arr = np.array(max_drawdowns)
    sharpe_arr = np.array(sharpe_ratios)

    # Percentiles
    pctiles = {f"p{int(ci*100)}_final_value": float(np.percentile(final_arr, ci * 100))
               for ci in confidence_intervals}
    pctiles_dd = {f"p{int(ci*100)}_max_drawdown": float(np.percentile(dd_arr, ci * 100))
                  for ci in confidence_intervals}

    result = {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "initial_capital": initial_capital,
        # Final value stats
        "median_final_value": float(np.median(final_arr)),
        "mean_final_value": float(np.mean(final_arr)),
        "std_final_value": float(np.std(final_arr)),
        # Risk metrics
        "prob_of_loss": float(np.mean(final_arr < initial_capital) * 100),
        "prob_sharpe_gt_1": float(np.mean(sharpe_arr > 1.0) * 100),
        "median_max_drawdown": float(np.median(dd_arr)),
        "p95_max_drawdown": float(np.percentile(dd_arr, 95)),
        # CI bands
        **pctiles,
        **pctiles_dd,
        # For plotting
        "equity_curve_bands": _compute_curve_bands(equity_curves, confidence_intervals),
    }

    _print_mc_summary(result, initial_capital)

    if save_chart:
        _save_chart(result, equity_curves, initial_capital, output_dir)

    return result


def _compute_curve_bands(
    curves: list[list[float]],
    confidence_intervals: list[float],
) -> dict:
    """Compute percentile bands across all equity curves for plotting."""
    if not curves:
        return {}

    max_len = max(len(c) for c in curves)
    # Pad shorter curves with final value
    padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
    matrix = np.array(padded)

    bands = {}
    for ci in confidence_intervals:
        bands[f"p{int(ci*100)}"] = list(np.percentile(matrix, ci * 100, axis=0))

    return bands


def _print_mc_summary(result: dict, initial: float) -> None:
    G, R, Y, B = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
    print(f"\n{'='*60}")
    print(f"  Monte Carlo Simulation Results ({result['n_simulations']:,} runs)")
    print(f"{'='*60}")
    print(f"  Initial Capital:      NPR {initial:>12,.0f}")
    print(f"  Median Final Value:   NPR {result['median_final_value']:>12,.0f}  "
          f"({(result['median_final_value']/initial-1)*100:+.1f}%)")
    print(f"  P5  Final Value:      NPR {result.get('p5_final_value',0):>12,.0f}  (worst 5%)")
    print(f"  P95 Final Value:      NPR {result.get('p95_final_value',0):>12,.0f}  (best 5%)")

    prob_loss = result["prob_of_loss"]
    prob_c = R if prob_loss > 20 else (Y if prob_loss > 10 else G)
    print(f"\n  Probability of Loss:  {prob_c}{prob_loss:.1f}%{B}  (target: < 20%)")

    ps = result["prob_sharpe_gt_1"]
    ps_c = G if ps > 50 else (Y if ps > 30 else R)
    print(f"  P(Sharpe > 1.0):      {ps_c}{ps:.1f}%{B}  (target: > 50%)")

    dd95 = result["p95_max_drawdown"]
    dd_c = G if dd95 < 25 else R
    print(f"  Median Max Drawdown:  {result['median_max_drawdown']:.1f}%")
    print(f"  P95 Max Drawdown:     {dd_c}{dd95:.1f}%{B}  (target: < 25%)")

    # Robust checks
    p5_positive = result.get("p5_final_value", 0) > initial
    print(f"\n  P5 final > initial:   {'\u2713 PASS' if p5_positive else '\u2717 FAIL'}")
    print(f"{'='*60}\n")


def _save_chart(
    result: dict,
    equity_curves: list[list[float]],
    initial: float,
    output_dir: str,
) -> None:
    """Save Monte Carlo fan chart as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        bands = result.get("equity_curve_bands", {})
        if not bands:
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        n_points = len(list(bands.values())[0])
        x = list(range(n_points))

        p50 = bands.get("p50", [])
        p25 = bands.get("p25", [])
        p75 = bands.get("p75", [])
        p5 = bands.get("p5", [])
        p95 = bands.get("p95", [])

        if p5 and p95:
            ax.fill_between(x, p5, p95, alpha=0.2, color="#58a6ff",
                            label="5th\u201395th percentile")
        if p25 and p75:
            ax.fill_between(x, p25, p75, alpha=0.4, color="#58a6ff",
                            label="25th\u201375th percentile")
        if p50:
            ax.plot(x, p50, color="#3fb950", linewidth=2, label="Median")

        # Initial capital line
        ax.axhline(y=initial, color="#f85149", linestyle="--",
                   linewidth=1, label="Initial capital", alpha=0.7)

        ax.set_xlabel("Trade Number", color="#8b949e")
        ax.set_ylabel("Portfolio Value (NPR)", color="#8b949e")
        ax.set_title("Monte Carlo Simulation \u2014 NEPSE MiroFish Strategy",
                     color="#c9d1d9", fontsize=13)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        ax.legend(facecolor="#161b22", labelcolor="#c9d1d9")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"NPR {x/1e6:.1f}M")
        )

        path = Path(output_dir) / "monte_carlo_chart.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        logger.info("Monte Carlo chart saved \u2192 %s", path)
        print(f"  Chart saved \u2192 {path}")
    except Exception as exc:
        logger.warning("Chart save failed: %s", exc)


def main() -> None:
    import argparse, json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Monte Carlo simulation")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--trades-file", default="data/processed/backtest/backtest_data.json")
    parser.add_argument("--output", default="data/processed/backtest")
    args = parser.parse_args()

    try:
        with open(args.trades_file) as fh:
            data = json.load(fh)
        trades = data.get("trades", [])
    except FileNotFoundError:
        logger.warning("No backtest data found at %s \u2014 using demo data", args.trades_file)
        trades = [{"pnl_pct": p, "action": "SELL"} for p in
                  [8.2, -3.1, 12.4, 5.7, -2.8, 15.3, -1.2, 9.8, 4.5, -6.1] * 8]

    run_monte_carlo(trades, n_simulations=args.n, output_dir=args.output)

if __name__ == "__main__":
    main()
