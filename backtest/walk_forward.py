"""
backtest/walk_forward.py
Walk-forward validation for NEPSE MiroFish strategy.
Tests that strategy generalises to out-of-sample data and isn't curve-fitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Walk-forward windows for NEPSE 2021–2024
# ---------------------------------------------------------------------------

WALK_FORWARD_WINDOWS = [
    {
        "window": 1,
        "train_start": "2021-01-01", "train_end": "2022-06-30",
        "test_start":  "2022-07-01", "test_end":  "2022-12-31",
        "description": "Train: Jan 2021–Jun 2022 | Test: Jul–Dec 2022",
    },
    {
        "window": 2,
        "train_start": "2021-07-01", "train_end": "2022-12-31",
        "test_start":  "2023-01-01", "test_end":  "2023-06-30",
        "description": "Train: Jul 2021–Dec 2022 | Test: Jan–Jun 2023",
    },
    {
        "window": 3,
        "train_start": "2022-01-01", "train_end": "2023-06-30",
        "test_start":  "2023-07-01", "test_end":  "2023-12-31",
        "description": "Train: Jan 2022–Jun 2023 | Test: Jul–Dec 2023",
    },
    {
        "window": 4,
        "train_start": "2022-07-01", "train_end": "2023-12-31",
        "test_start":  "2024-01-01", "test_end":  "2024-06-30",
        "description": "Train: Jul 2022–Dec 2023 | Test: Jan–Jun 2024",
    },
    {
        "window": 5,
        "train_start": "2023-01-01", "train_end": "2024-06-30",
        "test_start":  "2024-07-01", "test_end":  "2024-12-31",
        "description": "Train: Jan 2023–Jun 2024 | Test: Jul–Dec 2024",
    },
]

# Signal weight grid for optimisation
WEIGHT_GRID_STEP = 0.05  # 5% increments


@dataclass
class WindowResult:
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    optimal_weights: dict
    train_sharpe: float
    test_sharpe: float
    test_alpha: float
    test_max_drawdown: float
    test_total_return: float
    passed: bool
    notes: str = ""


@dataclass
class WalkForwardResult:
    windows: list[WindowResult] = field(default_factory=list)
    windows_passed: int = 0
    windows_total: int = 0
    avg_optimal_weights: dict = field(default_factory=dict)
    overall_passed: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------
# Weight grid generation
# ---------------------------------------------------------------------------

def _generate_weight_grid(step: float = 0.05) -> list[dict]:
    """Generate all valid weight combinations that sum to 1.0."""
    weights = []
    vals = list(np.arange(0.20, 0.65, step))  # MF: 20-60%
    for mf in vals:
        for tech in vals:
            sector = round(1.0 - mf - tech, 4)
            if 0.10 <= sector <= 0.35:
                weights.append({
                    "mf": round(float(mf), 4),
                    "tech": round(float(tech), 4),
                    "sector": round(float(sector), 4),
                })
    return weights


# ---------------------------------------------------------------------------
# Mini backtest runner for weight optimisation
# ---------------------------------------------------------------------------

def _run_mini_backtest(
    start: str, end: str,
    weights: dict,
    capital: float = 500_000.0,
) -> dict:
    """
    Run a lightweight backtest with given signal weights.
    Returns metrics dict with sharpe_ratio, alpha_pct, max_drawdown_pct.
    """
    try:
        from backtest.engine import NEPSEBacktestEngine
        from backtest.metrics import compute_metrics

        engine = NEPSEBacktestEngine(
            start_date=start,
            end_date=end,
            initial_capital_npr=capital,
            strategy_config={"signal_weights": weights},
        )
        result = engine.run()

        if len(result.daily_portfolio_values) < 10:
            return {"sharpe_ratio": 0.0, "alpha_pct": 0.0, "max_drawdown_pct": 99.0}

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
        return metrics
    except Exception as exc:
        logger.debug("Mini backtest failed (%s–%s, w=%s): %s", start, end, weights, exc)
        return {"sharpe_ratio": 0.0, "alpha_pct": 0.0, "max_drawdown_pct": 99.0}


# ---------------------------------------------------------------------------
# Weight optimiser
# ---------------------------------------------------------------------------

def find_optimal_weights(
    train_start: str,
    train_end: str,
    grid_step: float = 0.05,
    objective: str = "sharpe_ratio",
) -> dict:
    """
    Grid search over signal weights to maximise Sharpe on training data.
    WARNING: ONLY use these weights on OUT-OF-SAMPLE test data.

    Returns: {"mf": 0.40, "tech": 0.35, "sector": 0.25, "train_sharpe": 1.23}
    """
    grid = _generate_weight_grid(grid_step)
    if not grid:
        return {"mf": 0.40, "tech": 0.35, "sector": 0.25, "train_sharpe": 0.0}

    best_score = -999.0
    best_weights = {"mf": 0.40, "tech": 0.35, "sector": 0.25}
    best_metrics = {}

    logger.info(
        "Optimising weights: %d combinations for %s–%s",
        len(grid), train_start, train_end,
    )

    for i, w in enumerate(grid):
        if i % 20 == 0:
            logger.debug("Weight grid progress: %d/%d (best %s=%.3f)",
                          i, len(grid), objective, best_score)

        metrics = _run_mini_backtest(train_start, train_end, w)
        score = metrics.get(objective, 0.0)

        # Penalise excessive drawdown
        if metrics.get("max_drawdown_pct", 100) > 30:
            score *= 0.5

        if score > best_score:
            best_score = score
            best_weights = w
            best_metrics = metrics

    result = {**best_weights, "train_sharpe": round(best_score, 3)}
    logger.info(
        "Optimal weights found: MF=%.0f%% Tech=%.0f%% Sector=%.0f%% (Sharpe=%.3f)",
        best_weights["mf"] * 100, best_weights["tech"] * 100,
        best_weights["sector"] * 100, best_score,
    )
    return result


# ---------------------------------------------------------------------------
# Walk-forward validation runner
# ---------------------------------------------------------------------------

def run_walk_forward_validation(
    windows: Optional[list[dict]] = None,
    train_months: int = 18,
    test_months: int = 6,
    optimise_weights: bool = True,
) -> WalkForwardResult:
    """
    Run 5-window walk-forward validation.
    For each window: optimise weights on train, evaluate on test.
    """
    windows = windows or WALK_FORWARD_WINDOWS
    wf_result = WalkForwardResult(windows_total=len(windows))
    all_optimal_weights: list[dict] = []

    print(f"\n{'='*65}")
    print(f"  Walk-Forward Validation ({len(windows)} windows)")
    print(f"{'='*65}")

    for w in windows:
        wnum = w["window"]
        logger.info("Processing window %d: %s", wnum, w["description"])

        # ── Step 1: Find optimal weights on training data ────────────────────
        if optimise_weights:
            optimal = find_optimal_weights(w["train_start"], w["train_end"])
        else:
            optimal = {"mf": 0.40, "tech": 0.35, "sector": 0.25, "train_sharpe": 0.0}

        train_sharpe = optimal.get("train_sharpe", 0.0)
        weights = {"mf": optimal["mf"], "tech": optimal["tech"], "sector": optimal["sector"]}
        all_optimal_weights.append(weights)

        # ── Step 2: Evaluate on test data (NO re-optimisation) ───────────────
        test_metrics = _run_mini_backtest(w["test_start"], w["test_end"], weights)

        test_sharpe = test_metrics.get("sharpe_ratio", 0.0)
        test_alpha = test_metrics.get("alpha_pct", 0.0)
        test_dd = test_metrics.get("max_drawdown_pct", 99.0)
        test_return = test_metrics.get("total_return_pct", 0.0)

        # Pass criteria
        passed = (
            test_sharpe >= 0.7
            and test_alpha > 0
            and test_dd <= 30.0
        )

        notes = []
        if test_sharpe < 0.7:
            notes.append(f"Sharpe {test_sharpe:.2f} < 0.7")
        if test_alpha <= 0:
            notes.append(f"Alpha {test_alpha:.1f}% \u2264 0")
        if test_dd > 30:
            notes.append(f"Drawdown {test_dd:.1f}% > 30%")

        win_result = WindowResult(
            window=wnum,
            train_start=w["train_start"],
            train_end=w["train_end"],
            test_start=w["test_start"],
            test_end=w["test_end"],
            optimal_weights=weights,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            test_alpha=test_alpha,
            test_max_drawdown=test_dd,
            test_total_return=test_return,
            passed=passed,
            notes="; ".join(notes),
        )
        wf_result.windows.append(win_result)

        if passed:
            wf_result.windows_passed += 1

        # Print row
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(
            f"  Win {wnum}: {w['description'][:40]:<40}"
            f"  Sharpe={test_sharpe:+.2f}  \u03b1={test_alpha:+.1f}%"
            f"  DD=-{test_dd:.1f}%  [{status}]"
        )
        if notes:
            print(f"         \u21b3 {'; '.join(notes)}")

    # Compute average optimal weights
    if all_optimal_weights:
        wf_result.avg_optimal_weights = {
            "mf": round(sum(w["mf"] for w in all_optimal_weights) / len(all_optimal_weights), 3),
            "tech": round(sum(w["tech"] for w in all_optimal_weights) / len(all_optimal_weights), 3),
            "sector": round(sum(w["sector"] for w in all_optimal_weights) / len(all_optimal_weights), 3),
        }

    wf_result.overall_passed = wf_result.windows_passed >= 4

    _print_summary(wf_result)
    return wf_result


def _print_summary(wf: WalkForwardResult) -> None:
    print(f"\n{'='*65}")
    avg = wf.avg_optimal_weights
    pass_str = f"{wf.windows_passed}/{wf.windows_total} windows"
    if wf.overall_passed:
        verdict = "\033[92mSTRATEGY VALIDATED \u2014 proceed to stress testing.\033[0m"
    else:
        verdict = "\033[91mSTRATEGY NEEDS REVISION \u2014 investigate failing windows.\033[0m"

    print(f"  Pass: {pass_str} Sharpe > 0.7  |  {pass_str} Alpha > 0")
    if avg:
        print(
            f"  Optimal weights (avg): "
            f"MiroFish {avg.get('mf',0)*100:.0f}%  "
            f"Technical {avg.get('tech',0)*100:.0f}%  "
            f"Sector {avg.get('sector',0)*100:.0f}%"
        )
    print(f"\n  {verdict}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--no-optimise", action="store_true")
    args = parser.parse_args()

    result = run_walk_forward_validation(
        windows=WALK_FORWARD_WINDOWS[:args.windows],
        optimise_weights=not args.no_optimise,
    )
    import sys
    sys.exit(0 if result.overall_passed else 1)


if __name__ == "__main__":
    main()
