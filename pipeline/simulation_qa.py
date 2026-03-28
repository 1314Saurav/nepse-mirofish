"""
pipeline/simulation_qa.py

Quality checker that validates each MiroFish simulation output before
signal extraction. Flags issues and saves a QA report.

Usage:
    from pipeline.simulation_qa import run_qa_checks
    flagged, report = run_qa_checks(sim_output, signal)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

_ROOT   = Path(__file__).resolve().parent.parent
QA_DIR  = _ROOT / "data" / "processed" / "qa"

# ── Quality check definitions ──────────────────────────────────────────────────

# Each check is: (check_name, predicate, warning_message)
# Predicate receives the sim_output dict and signal dict
QualityCheck = tuple[str, Callable, str]

QUALITY_CHECKS: list[QualityCheck] = [
    (
        "min_active_agents",
        lambda s, sig: len(s.get("actions", [])) >= 100,
        "Fewer than 100 agent actions — simulation may have very few participants"
    ),
    (
        "min_rounds_completed",
        lambda s, sig: len(s.get("timeline", [])) >= 3 or len(s.get("actions", [])) > 50,
        "Fewer than 3 rounds completed — consider increasing max_rounds or simulation timeout"
    ),
    (
        "no_sentiment_lock",
        lambda s, sig: abs(sig.get("bull_bear_score", 0)) < 0.95,
        "Sentiment score is suspiciously extreme (>0.95) — possible LLM groupthink, consider rerun"
    ),
    (
        "platform_not_diverged_wildly",
        lambda s, sig: abs(
            sig.get("twitter_score", 0) - sig.get("reddit_score", 0)
        ) < 0.6,
        "Twitter and Reddit platforms disagree by >0.6 — check seed quality and agent diversity"
    ),
    (
        "sector_coverage",
        lambda s, sig: len([v for v in sig.get("sector_signals", {}).values() if v != 0.0]) >= 2,
        "Fewer than 2 sectors with non-zero signal — sector-specific agents may not have activated"
    ),
    (
        "has_key_themes",
        lambda s, sig: len(sig.get("key_themes", [])) >= 1,
        "No key themes extracted — content may be too sparse or off-topic for NEPSE"
    ),
    (
        "has_top_drivers",
        lambda s, sig: len(sig.get("top_driver_agent_types", [])) >= 1,
        "No dominant agent types identified — agent type classification may need tuning"
    ),
    (
        "round_scores_not_flat",
        lambda s, sig: len(set(
            round(r, 1) for r in sig.get("raw_round_scores", [0.0, 0.0])
        )) > 1 or len(sig.get("raw_round_scores", [])) < 2,
        "Round-by-round scores are completely flat — sentiment may not be evolving across rounds"
    ),
    (
        "confidence_in_range",
        lambda s, sig: 10.0 <= sig.get("confidence_pct", 50) <= 99.0,
        "Confidence percentage is outside 10-99% — likely a scoring edge case, verify manually"
    ),
    (
        "actions_have_content",
        lambda s, sig: sum(
            1 for a in s.get("actions", [])
            if (a.get("content") or a.get("text") or "").strip()
        ) >= max(1, len(s.get("actions", [])) * 0.5),
        "More than 50% of actions have empty content — simulation may have generated hollow posts"
    ),
]


def run_qa_checks(sim_output: dict, signal: dict) -> tuple[list[str], dict]:
    """
    Run all quality checks against simulation output and extracted signal.

    Returns
    -------
    (quality_flags, qa_report)
        quality_flags: list of warning strings (empty if all pass)
        qa_report:     full structured QA report dict
    """
    date_str  = sim_output.get("date",
                               datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    results   = []
    flags     = []

    for check_name, predicate, warning in QUALITY_CHECKS:
        try:
            passed = predicate(sim_output, signal)
        except Exception as exc:
            passed  = False
            warning = f"{warning} (check error: {exc})"

        status = "PASS" if passed else "WARN"
        if not passed:
            flags.append(f"{check_name}: {warning}")

        results.append({
            "check":   check_name,
            "status":  status,
            "message": "" if passed else warning,
        })

    qa_report = {
        "date":              date_str,
        "simulation_id":     sim_output.get("simulation_id", ""),
        "checked_at":        datetime.now(timezone.utc).isoformat(),
        "total_checks":      len(QUALITY_CHECKS),
        "passed":            sum(1 for r in results if r["status"] == "PASS"),
        "warnings":          len(flags),
        "quality_flags":     flags,
        "check_results":     results,
        "signal_summary": {
            "direction":        signal.get("direction"),
            "bull_bear_score":  signal.get("bull_bear_score"),
            "confidence_pct":   signal.get("confidence_pct"),
            "total_actions":    signal.get("total_actions"),
            "signal_strength":  signal.get("signal_strength"),
        },
    }

    # Save QA report
    QA_DIR.mkdir(parents=True, exist_ok=True)
    qa_path = QA_DIR / f"qa_{date_str}.json"
    qa_path.write_text(
        json.dumps(qa_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return flags, qa_report


def print_qa_report(qa_report: dict) -> None:
    """Print a formatted QA report to terminal."""
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    RESET  = "\033[0m"
    CYAN   = "\033[96m"

    total   = qa_report["total_checks"]
    passed  = qa_report["passed"]
    warned  = qa_report["warnings"]
    sim_id  = qa_report.get("simulation_id", "N/A")

    print(f"\n{BOLD}Simulation QA Report{RESET}")
    print("=" * 56)
    print(f"  Simulation ID : {sim_id[:16]}..." if len(sim_id) > 16 else f"  Simulation ID : {sim_id}")
    print(f"  Checks        : {passed}/{total} passed  ({warned} warnings)")
    print()

    for r in qa_report.get("check_results", []):
        icon   = f"{GREEN}PASS{RESET}" if r["status"] == "PASS" else f"{YELLOW}WARN{RESET}"
        name   = r["check"].replace("_", " ").ljust(30)
        detail = f"  {r['message'][:60]}..." if len(r.get("message","")) > 60 else f"  {r.get('message','')}"
        print(f"  [{icon}] {name}" + (f"\n        {YELLOW}{detail}{RESET}" if r["message"] else ""))

    print()
    sig = qa_report.get("signal_summary", {})
    print(f"  {CYAN}Signal summary:{RESET}")
    print(f"    Direction    : {sig.get('direction', 'N/A')}")
    print(f"    Bull/Bear    : {sig.get('bull_bear_score', 0):+.3f}")
    print(f"    Confidence   : {sig.get('confidence_pct', 0):.1f}%")
    print(f"    Actions      : {sig.get('total_actions', 0)}")

    if warned == 0:
        print(f"\n  {GREEN}All {total} quality checks passed.{RESET}")
    else:
        print(f"\n  {YELLOW}{warned} warning(s) flagged — review before using signal.{RESET}")
        print(f"  QA report saved to data/processed/qa/qa_{qa_report['date']}.json")
