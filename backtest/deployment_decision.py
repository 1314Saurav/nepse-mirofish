"""
backtest/deployment_decision.py
Final go/no-go deployment decision checker for NEPSE MiroFish strategy.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deployment criteria
# ---------------------------------------------------------------------------

DEPLOYMENT_CRITERIA = {
    "required": [
        {"metric": "sharpe_ratio",          "op": ">=", "threshold": 1.0,    "weight": "REQUIRED",    "label": "Sharpe Ratio >= 1.0"},
        {"metric": "max_drawdown_pct",       "op": "<=", "threshold": 20.0,   "weight": "REQUIRED",    "label": "Max Drawdown <= 20%"},
        {"metric": "walk_forward_pass_rate", "op": ">=", "threshold": 0.8,    "weight": "REQUIRED",    "label": "Walk-Forward >= 4/5 windows"},
        {"metric": "stress_test_pass_rate",  "op": ">=", "threshold": 0.8,    "weight": "REQUIRED",    "label": "Stress Tests >= 4/5"},
        {"metric": "monte_carlo_p5_positive","op": "==", "threshold": True,   "weight": "REQUIRED",    "label": "Monte Carlo P5 > initial capital"},
        {"metric": "alpha_pct",              "op": ">",  "threshold": 0.0,    "weight": "REQUIRED",    "label": "Alpha vs NEPSE > 0"},
    ],
    "recommended": [
        {"metric": "sortino_ratio",          "op": ">=", "threshold": 1.3,    "weight": "RECOMMENDED", "label": "Sortino Ratio >= 1.3"},
        {"metric": "profit_factor",          "op": ">=", "threshold": 1.4,    "weight": "RECOMMENDED", "label": "Profit Factor >= 1.4"},
        {"metric": "mirofish_accuracy_pct",  "op": ">=", "threshold": 55.0,   "weight": "RECOMMENDED", "label": "MiroFish Accuracy >= 55%"},
        {"metric": "win_rate_pct",           "op": ">=", "threshold": 50.0,   "weight": "RECOMMENDED", "label": "Win Rate >= 50%"},
        {"metric": "calmar_ratio",           "op": ">=", "threshold": 0.8,    "weight": "RECOMMENDED", "label": "Calmar Ratio >= 0.8"},
    ],
}


def _check(value, op: str, threshold) -> bool:
    """Evaluate a single criterion."""
    if op == ">=":
        return float(value) >= float(threshold)
    elif op == "<=":
        return float(value) <= float(threshold)
    elif op == ">":
        return float(value) > float(threshold)
    elif op == "<":
        return float(value) < float(threshold)
    elif op == "==":
        if isinstance(threshold, bool):
            return bool(value) == threshold
        return value == threshold
    return False


def _get_value(metrics: dict, metric_key: str):
    """Extract metric value from flat or nested metrics dict."""
    # Direct lookup
    if metric_key in metrics:
        return metrics[metric_key]

    # Walk-forward pass rate
    if metric_key == "walk_forward_pass_rate":
        wf = metrics.get("walk_forward", {})
        if isinstance(wf, dict):
            passed = wf.get("windows_passed", 0)
            total = wf.get("windows_total", 5)
            return passed / total if total > 0 else 0.0

    # Stress test pass rate
    if metric_key == "stress_test_pass_rate":
        st = metrics.get("stress_tests", {})
        if isinstance(st, dict):
            passed = st.get("passed", 0)
            total = st.get("total", 5)
            return passed / total if total > 0 else 0.0

    # Monte Carlo P5 positive
    if metric_key == "monte_carlo_p5_positive":
        mc = metrics.get("monte_carlo", {})
        if isinstance(mc, dict):
            p5 = mc.get("p5_final_value", 0)
            initial = mc.get("initial_capital", 1_000_000)
            return p5 > initial

    # MiroFish accuracy
    if metric_key == "mirofish_accuracy_pct":
        mf_acc = metrics.get("mirofish_accuracy", {})
        if isinstance(mf_acc, dict):
            return mf_acc.get("overall_accuracy_pct", 0.0)
        attr = metrics.get("attribution", {})
        if isinstance(attr, dict):
            return attr.get("mf_accuracy", 0.0)

    return None


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_deployment_readiness(metrics: dict) -> dict:
    """
    Check all deployment criteria and return a structured decision.

    Parameters
    ----------
    metrics : combined metrics dict (from compute_metrics + walk_forward + stress + mc results)

    Returns
    -------
    Decision dict with keys: decision, required_passed, required_total,
    recommended_passed, recommended_total, blocking_failures, warnings,
    recommendation, next_steps
    """
    blocking_failures = []
    required_passed = 0
    required_total = len(DEPLOYMENT_CRITERIA["required"])

    for criterion in DEPLOYMENT_CRITERIA["required"]:
        value = _get_value(metrics, criterion["metric"])
        if value is None:
            blocking_failures.append(
                f"{criterion['label']} -- NO DATA (metric '{criterion['metric']}' missing)"
            )
            continue

        if _check(value, criterion["op"], criterion["threshold"]):
            required_passed += 1
        else:
            blocking_failures.append(
                f"{criterion['label']} -- FAILED (got {value}, need {criterion['op']} {criterion['threshold']})"
            )

    warnings = []
    recommended_passed = 0
    recommended_total = len(DEPLOYMENT_CRITERIA["recommended"])

    for criterion in DEPLOYMENT_CRITERIA["recommended"]:
        value = _get_value(metrics, criterion["metric"])
        if value is None:
            continue

        if _check(value, criterion["op"], criterion["threshold"]):
            recommended_passed += 1
        else:
            warnings.append(
                f"{criterion['label']} -- below threshold (got {_fmt(value)}, target {criterion['op']} {criterion['threshold']})"
            )

    # Decision logic
    all_required_pass = len(blocking_failures) == 0
    if all_required_pass and recommended_passed >= 3:
        decision = "DEPLOY"
    elif all_required_pass:
        decision = "REVISE"
    else:
        decision = "REJECT"

    recommendation = _build_recommendation(decision, blocking_failures, warnings, metrics)
    next_steps = _build_next_steps(decision, blocking_failures, warnings)

    return {
        "decision": decision,
        "required_passed": required_passed,
        "required_total": required_total,
        "recommended_passed": recommended_passed,
        "recommended_total": recommended_total,
        "blocking_failures": blocking_failures,
        "warnings": warnings,
        "recommendation": recommendation,
        "next_steps": next_steps,
    }


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _build_recommendation(
    decision: str,
    blocking_failures: list[str],
    warnings: list[str],
    metrics: dict,
) -> str:
    sharpe = metrics.get("sharpe_ratio", 0)
    dd = metrics.get("max_drawdown_pct", 100)

    if decision == "DEPLOY":
        return (
            f"Strategy meets all required deployment criteria. "
            f"Sharpe={sharpe:.2f}, MaxDD={dd:.1f}%. "
            "Proceed to Phase 5 paper trading with initial capital of NPR 500,000. "
            "Monitor closely for first 30 trading days before scaling."
        )
    elif decision == "REVISE":
        warn_str = "; ".join(warnings[:2]) if warnings else "See warnings above"
        return (
            f"All required criteria pass but {len(warnings)} recommended criteria fail. "
            f"Issues: {warn_str}. "
            "Proceed to paper trading at 50% normal position sizes while investigating warnings."
        )
    else:
        fail_str = "; ".join(blocking_failures[:2]) if blocking_failures else "See failures above"
        return (
            f"STRATEGY NOT READY FOR DEPLOYMENT. {len(blocking_failures)} required criteria fail. "
            f"Critical: {fail_str}. "
            "Do not proceed to paper trading until all required criteria are met."
        )


def _build_next_steps(decision: str, failures: list[str], warnings: list[str]) -> list[str]:
    steps = []
    if decision == "DEPLOY":
        steps = [
            "Start Phase 5: Paper trading with NPR 500,000 initial capital",
            "Set position sizing to 50% of backtest levels for first month",
            "Monitor daily: regime detection, MiroFish signal, exit conditions",
            "Schedule weekly report every Friday 18:00 NST",
            "Re-run backtest quarterly to detect strategy decay",
        ]
    elif decision == "REVISE":
        steps = ["Start paper trading at reduced position sizes (50%)"]
        for w in warnings[:3]:
            if "Sortino" in w:
                steps.append("Investigate: tighten stop losses to improve Sortino ratio")
            elif "Profit Factor" in w:
                steps.append("Investigate: improve exit conditions to increase profit factor")
            elif "MiroFish" in w:
                steps.append("Improve: enrich agent personas with more NEPSE-specific prompts")
            elif "Win Rate" in w:
                steps.append("Improve: raise entry thresholds to improve win rate")
        steps.append("Re-evaluate in 3 months of paper trading")
    else:
        for f in failures[:4]:
            if "Sharpe" in f:
                steps.append("Fix: Review signal weights -- Sharpe too low")
            elif "Drawdown" in f:
                steps.append("Fix: Tighten stop losses -- drawdown exceeds 20%")
            elif "Walk-Forward" in f:
                steps.append("Fix: Strategy overfitting -- re-tune entry parameters")
            elif "Stress" in f:
                steps.append("Fix: Review trading rules for stress scenarios")
            elif "Monte Carlo" in f:
                steps.append("Fix: Strategy loses money in bad-luck scenarios -- reduce risk")
            elif "Alpha" in f:
                steps.append("Fix: Strategy underperforms NEPSE index -- fundamentally broken")
        steps.append("Re-run full Phase 4 backtest after fixes")
    return steps


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_deployment_decision(result: dict) -> None:
    G, R, Y, B, BOLD = "\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m"

    decision = result["decision"]
    decision_colour = {
        "DEPLOY": G,
        "REVISE": Y,
        "REJECT": R,
    }.get(decision, B)

    print(f"\n{'='*62}")
    print(f"  {BOLD}DEPLOYMENT DECISION{B}")
    print(f"{'='*62}")
    print(f"\n  Required criteria:    "
          f"{result['required_passed']}/{result['required_total']} passed")
    print(f"  Recommended criteria: "
          f"{result['recommended_passed']}/{result['recommended_total']} passed")

    if result["blocking_failures"]:
        print(f"\n  {R}BLOCKING FAILURES:{B}")
        for f in result["blocking_failures"]:
            print(f"    {R}x{B} {f}")

    if result["warnings"]:
        print(f"\n  {Y}WARNINGS:{B}")
        for w in result["warnings"]:
            print(f"    {Y}!{B} {w}")

    print(f"\n  {'='*58}")
    print(f"  {BOLD}DECISION: {decision_colour}{decision}{B}")
    print(f"  {'='*58}")
    print(f"\n  {result['recommendation']}")

    if result["next_steps"]:
        print(f"\n  Next Steps:")
        for i, step in enumerate(result["next_steps"], 1):
            print(f"    {i}. {step}")

    print(f"\n{'='*62}\n")


def save_decision(result: dict, output_dir: str = "data/processed/backtest") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "deployment_decision.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Deployment decision saved -> %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Deployment readiness check")
    parser.add_argument("--data", default="data/processed/backtest/backtest_data.json")
    parser.add_argument("--output", default="data/processed/backtest")
    args = parser.parse_args()

    try:
        with open(args.data) as fh:
            data = json.load(fh)
        metrics = data.get("metrics", data)
    except FileNotFoundError:
        print(f"No data at {args.data}. Run backtest engine first.")
        return

    result = evaluate_deployment_readiness(metrics)
    print_deployment_decision(result)
    save_decision(result, args.output)
    import sys
    sys.exit(0 if result["decision"] == "DEPLOY" else 1)

if __name__ == "__main__":
    main()
