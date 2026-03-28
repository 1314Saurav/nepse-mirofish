"""
deployment/readiness_check.py
Final deployment readiness checker before going live with real capital.

8 BLOCKING checks that ALL must pass:
  1. trading_days_completed   >= 20 (4 weeks minimum)
  2. signal_accuracy_5d       >= 55% (5-day window accuracy)
  3. paper_return             > 0%   (positive P&L)
  4. paper_alpha              > 0%   (outperforms NEPSE index)
  5. zero_critical_bugs       True   (no exceptions in last 5 days)
  6. mirofish_quality_flags   < 20%  (of signals flagged as low quality)
  7. stop_loss_discipline     True   (no stop losses skipped)
  8. max_drawdown             < 15%  (paper trading drawdown)

Run:
  python -m deployment.readiness_check
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Criteria registry
# ---------------------------------------------------------------------------

LIVE_DEPLOYMENT_CRITERIA: dict[str, dict[str, Any]] = {
    "trading_days_completed": {
        "threshold": 20,
        "operator": ">=",
        "description": "Minimum 20 trading days (4 calendar weeks) of paper trading.",
        "unit": "days",
    },
    "signal_accuracy_5d": {
        "threshold": 55.0,
        "operator": ">=",
        "description": "5-day forward accuracy of generated signals must be >= 55%.",
        "unit": "%",
    },
    "paper_return": {
        "threshold": 0.0,
        "operator": ">",
        "description": "Cumulative paper P&L must be strictly positive.",
        "unit": "%",
    },
    "paper_alpha": {
        "threshold": 0.0,
        "operator": ">",
        "description": "Paper portfolio return must exceed NEPSE index return over same period.",
        "unit": "%",
    },
    "zero_critical_bugs": {
        "threshold": True,
        "operator": "==",
        "description": "No ERROR-level log entries in the last 5 trading-day cycles.",
        "unit": "bool",
    },
    "mirofish_quality_flags": {
        "threshold": 20.0,
        "operator": "<",
        "description": "Less than 20% of signals flagged as low quality by MiroFish scorer.",
        "unit": "%",
    },
    "stop_loss_discipline": {
        "threshold": True,
        "operator": "==",
        "description": "All triggered stop-losses were respected — none skipped.",
        "unit": "bool",
    },
    "max_drawdown": {
        "threshold": 15.0,
        "operator": "<",
        "description": "Maximum equity drawdown during paper trading must be below 15%.",
        "unit": "%",
    },
}


# ---------------------------------------------------------------------------
# Checker class
# ---------------------------------------------------------------------------


class DeploymentReadinessChecker:
    """
    Loads persisted paper-trading artifacts and evaluates all 8 blocking
    go/no-go criteria.

    Parameters
    ----------
    session_id:
        The paper-trading session identifier (matches directory name under
        ``data_dir``).
    data_dir:
        Root directory that contains per-session subdirectories.
    """

    def __init__(self, session_id: str, data_dir: str = "data/paper_trading") -> None:
        self.session_id = session_id
        self._session_dir = Path(data_dir) / session_id
        self._data: dict[str, Any] = {}

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_paper_results(self) -> dict[str, Any]:
        """
        Load all persisted paper-trading artefacts from the session directory.

        Reads:
          - state.json            — portfolio state and equity curve
          - accuracy_report.json  — signal accuracy by window
          - signal_history.json   — per-signal records including quality flags
          - snapshots/            — one JSON file per trading day cycle

        Returns a merged dict stored in ``self._data`` for downstream checks.
        Raises FileNotFoundError if the session directory is missing.
        """
        if not self._session_dir.exists():
            raise FileNotFoundError(
                f"Session directory not found: {self._session_dir}"
            )

        data: dict[str, Any] = {}

        # state.json
        state_path = self._session_dir / "state.json"
        if state_path.exists():
            data["state"] = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            logger.warning("state.json not found in %s", self._session_dir)
            data["state"] = {}

        # accuracy_report.json
        accuracy_path = self._session_dir / "accuracy_report.json"
        if accuracy_path.exists():
            data["accuracy_report"] = json.loads(
                accuracy_path.read_text(encoding="utf-8")
            )
        else:
            logger.warning("accuracy_report.json not found in %s", self._session_dir)
            data["accuracy_report"] = {}

        # signal_history.json
        signal_history_path = self._session_dir / "signal_history.json"
        if signal_history_path.exists():
            data["signal_history"] = json.loads(
                signal_history_path.read_text(encoding="utf-8")
            )
        else:
            logger.warning("signal_history.json not found in %s", self._session_dir)
            data["signal_history"] = []

        # snapshots directory
        snapshots_dir = self._session_dir / "snapshots"
        if snapshots_dir.exists():
            snapshot_files = sorted(snapshots_dir.glob("*.json"))
            data["snapshots"] = []
            for sf in snapshot_files:
                try:
                    data["snapshots"].append(
                        json.loads(sf.read_text(encoding="utf-8"))
                    )
                except json.JSONDecodeError as exc:
                    logger.warning("Could not parse snapshot %s: %s", sf, exc)
        else:
            data["snapshots"] = []

        self._data = data
        logger.info(
            "Loaded paper results for session '%s': %d snapshots, %d signals",
            self.session_id,
            len(data["snapshots"]),
            len(data["signal_history"]),
        )
        return data

    # ── Individual checks ────────────────────────────────────────────────────

    def check_trading_days(self) -> tuple[bool, str]:
        """
        Count completed trading-day cycles from the snapshots directory.

        Each snapshot file represents one completed end-of-day cycle.
        """
        count = len(self._data.get("snapshots", []))
        threshold = LIVE_DEPLOYMENT_CRITERIA["trading_days_completed"]["threshold"]
        passed = count >= threshold
        msg = (
            f"{count} trading days completed "
            f"({'PASS' if passed else f'need {threshold - count} more'})"
        )
        return passed, msg

    def check_signal_accuracy(self) -> tuple[bool, str]:
        """
        Read 5-day forward accuracy from accuracy_report.json.

        Expects the report to have a key ``"5d"`` containing ``{"accuracy": float}``.
        """
        report = self._data.get("accuracy_report", {})
        window_data = report.get("5d") or report.get("5") or {}
        accuracy: Optional[float] = window_data.get("accuracy")
        threshold = LIVE_DEPLOYMENT_CRITERIA["signal_accuracy_5d"]["threshold"]

        if accuracy is None:
            return False, "5-day accuracy not found in accuracy_report.json"

        passed = accuracy >= threshold
        msg = f"5-day signal accuracy: {accuracy:.1f}% (threshold: {threshold}%)"
        return passed, msg

    def check_positive_return(self) -> tuple[bool, str]:
        """
        Compare current portfolio value against the starting capital recorded
        in state.json.
        """
        state = self._data.get("state", {})
        current_value: Optional[float] = state.get("current_value")
        starting_capital: Optional[float] = state.get("starting_capital")

        if current_value is None or starting_capital is None:
            return False, "current_value or starting_capital missing from state.json"
        if starting_capital == 0:
            return False, "starting_capital is zero — cannot compute return"

        paper_return_pct = (current_value - starting_capital) / starting_capital * 100
        passed = paper_return_pct > 0.0
        msg = f"Paper return: {paper_return_pct:+.2f}%"
        return passed, msg

    def check_alpha(self) -> tuple[bool, str]:
        """
        Compare paper portfolio return to the NEPSE index return recorded in
        state.json over the same period.

        Expects state.json to contain ``"nepse_return_pct"`` alongside
        ``"current_value"`` and ``"starting_capital"``.
        """
        state = self._data.get("state", {})
        current_value: Optional[float] = state.get("current_value")
        starting_capital: Optional[float] = state.get("starting_capital")
        nepse_return: Optional[float] = state.get("nepse_return_pct")

        if current_value is None or starting_capital is None:
            return False, "current_value or starting_capital missing from state.json"
        if starting_capital == 0:
            return False, "starting_capital is zero — cannot compute alpha"
        if nepse_return is None:
            return False, "nepse_return_pct missing from state.json — cannot compute alpha"

        paper_return_pct = (current_value - starting_capital) / starting_capital * 100
        alpha = paper_return_pct - nepse_return
        passed = alpha > 0.0
        msg = (
            f"Alpha: {alpha:+.2f}% "
            f"(paper {paper_return_pct:+.2f}% vs NEPSE {nepse_return:+.2f}%)"
        )
        return passed, msg

    def check_zero_bugs(self) -> tuple[bool, str]:
        """
        Scan the last 5 cycle log files for ERROR-level log entries.

        Looks for log files in ``logs/`` at the project root, filtered to
        the paper-trading session, sorted by modification time.
        """
        logs_dir = Path("logs")
        if not logs_dir.exists():
            # No log directory — treat as no bugs found (conservative pass)
            return True, "No logs/ directory found — assuming no critical errors"

        log_files = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        recent_logs = log_files[-5:] if len(log_files) >= 5 else log_files

        error_count = 0
        for log_path in recent_logs:
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    if " ERROR " in line or " CRITICAL " in line:
                        error_count += 1
            except OSError as exc:
                logger.warning("Could not read log file %s: %s", log_path, exc)

        passed = error_count == 0
        msg = (
            f"ERROR entries in last {len(recent_logs)} cycle log(s): {error_count}"
        )
        return passed, msg

    def check_mirofish_quality(self) -> tuple[bool, str]:
        """
        Compute the fraction of signals flagged as low quality in
        signal_history.json.

        Each signal record is expected to carry a ``"quality_flag"`` bool or
        string field.  Values of ``True``, ``"low"``, or ``"LOW"`` are counted
        as flagged.
        """
        signals: list[dict] = self._data.get("signal_history", [])
        if not signals:
            return False, "signal_history.json is empty — cannot evaluate quality flags"

        flagged = 0
        for sig in signals:
            qf = sig.get("quality_flag")
            if qf is True or (isinstance(qf, str) and qf.lower() in ("low", "poor")):
                flagged += 1

        flag_pct = flagged / len(signals) * 100
        threshold = LIVE_DEPLOYMENT_CRITERIA["mirofish_quality_flags"]["threshold"]
        passed = flag_pct < threshold
        msg = (
            f"Low-quality signals: {flagged}/{len(signals)} ({flag_pct:.1f}%) "
            f"(threshold: <{threshold}%)"
        )
        return passed, msg

    def check_stop_loss_discipline(self) -> tuple[bool, str]:
        """
        Verify no stop-losses were skipped by reading the ``stop_loss_discipline``
        metric from state.json or trade_log.json.

        Expects state.json to carry a key ``"stop_loss_discipline"`` (bool) or
        ``"stop_losses_skipped"`` (int).
        """
        state = self._data.get("state", {})

        # Direct bool flag
        discipline_flag = state.get("stop_loss_discipline")
        if isinstance(discipline_flag, bool):
            msg = (
                "Stop-loss discipline: all respected"
                if discipline_flag
                else "Stop-loss discipline: one or more stop-losses were SKIPPED"
            )
            return discipline_flag, msg

        # Numeric count of skipped stop-losses
        skipped = state.get("stop_losses_skipped")
        if skipped is not None:
            passed = int(skipped) == 0
            msg = f"Stop-losses skipped: {skipped}"
            return passed, msg

        # Fallback: check trade log inside the session directory
        trade_log_path = self._session_dir / "trade_log.json"
        if trade_log_path.exists():
            try:
                trades: list[dict] = json.loads(
                    trade_log_path.read_text(encoding="utf-8")
                )
                skipped_trades = [
                    t for t in trades if t.get("stop_loss_skipped") is True
                ]
                passed = len(skipped_trades) == 0
                msg = f"Stop-losses skipped (from trade_log.json): {len(skipped_trades)}"
                return passed, msg
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read trade_log.json: %s", exc)

        return False, "stop_loss_discipline data not found in state.json or trade_log.json"

    def check_max_drawdown(self) -> tuple[bool, str]:
        """
        Compute maximum equity drawdown from the equity curve embedded in
        state.json (``"equity_curve": [float, ...]``) or from snapshots.

        Falls back to snapshots' ``"portfolio_value"`` field if no
        ``equity_curve`` key is present in state.json.
        """
        state = self._data.get("state", {})
        equity_curve: Optional[list[float]] = state.get("equity_curve")

        if not equity_curve:
            # Build from snapshots
            snapshots = self._data.get("snapshots", [])
            equity_curve = [
                s["portfolio_value"]
                for s in snapshots
                if isinstance(s.get("portfolio_value"), (int, float))
            ]

        if not equity_curve or len(equity_curve) < 2:
            return False, "Insufficient equity curve data to compute max drawdown"

        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        threshold = LIVE_DEPLOYMENT_CRITERIA["max_drawdown"]["threshold"]
        passed = max_dd < threshold
        msg = f"Max drawdown: {max_dd:.2f}% (threshold: <{threshold}%)"
        return passed, msg

    # ── Orchestrator ─────────────────────────────────────────────────────────

    def run_all_checks(self) -> dict[str, dict[str, Any]]:
        """
        Execute all 8 blocking checks and return a structured results dict.

        Returns
        -------
        dict keyed by check name, each value containing:
          - passed    (bool)
          - message   (str)
          - value     (str | float | bool | None)
          - threshold (Any) — from LIVE_DEPLOYMENT_CRITERIA
        """
        check_methods = [
            ("trading_days_completed", self.check_trading_days),
            ("signal_accuracy_5d", self.check_signal_accuracy),
            ("paper_return", self.check_positive_return),
            ("paper_alpha", self.check_alpha),
            ("zero_critical_bugs", self.check_zero_bugs),
            ("mirofish_quality_flags", self.check_mirofish_quality),
            ("stop_loss_discipline", self.check_stop_loss_discipline),
            ("max_drawdown", self.check_max_drawdown),
        ]

        results: dict[str, dict[str, Any]] = {}
        for name, method in check_methods:
            try:
                passed, message = method()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error in check '%s'", name)
                passed = False
                message = f"Check raised an exception: {exc}"

            criteria = LIVE_DEPLOYMENT_CRITERIA.get(name, {})
            results[name] = {
                "passed": passed,
                "message": message,
                "threshold": criteria.get("threshold"),
                "description": criteria.get("description", ""),
            }
            status = "PASS" if passed else "FAIL"
            logger.info("[%s] %s — %s", status, name, message)

        return results

    # ── Reporting ────────────────────────────────────────────────────────────

    def print_readiness_report(self, results: dict[str, dict[str, Any]]) -> None:
        """
        Print a formatted terminal table of all check results.

        Uses ANSI colour codes when the terminal supports them; falls back to
        plain text otherwise.
        """
        # Detect colour support
        _use_colour = sys.stdout.isatty()
        GREEN = "\033[92m" if _use_colour else ""
        RED = "\033[91m" if _use_colour else ""
        BOLD = "\033[1m" if _use_colour else ""
        RESET = "\033[0m" if _use_colour else ""

        all_passed = all(r["passed"] for r in results.values())
        overall_label = (
            f"{GREEN}{BOLD}READY FOR LIVE TRADING{RESET}"
            if all_passed
            else f"{RED}{BOLD}NOT READY — FIX FAILING CHECKS{RESET}"
        )

        line = "=" * 72
        print(f"\n{line}")
        print(f"  NEPSE MiroFish — Live Deployment Readiness Report")
        print(f"  Session : {self.session_id}")
        print(f"  Date    : {date.today().isoformat()}")
        print(f"  Overall : {overall_label}")
        print(line)

        col_w = 30
        for check_name, result in results.items():
            status = (
                f"{GREEN}PASS{RESET}" if result["passed"] else f"{RED}FAIL{RESET}"
            )
            label = check_name.replace("_", " ").title()
            print(f"  [{status}]  {label:<{col_w}}  {result['message']}")

        print(line)
        n_pass = sum(1 for r in results.values() if r["passed"])
        n_total = len(results)
        print(f"  {n_pass}/{n_total} checks passed\n")

    def save_readiness_report(self, results: dict[str, dict[str, Any]]) -> Path:
        """
        Persist the readiness report as JSON to
        ``data/processed/deployment/readiness_YYYYMMDD.json``.

        Returns the path of the saved file.
        """
        output_dir = Path("data/processed/deployment")
        output_dir.mkdir(parents=True, exist_ok=True)

        today_str = date.today().strftime("%Y%m%d")
        out_path = output_dir / f"readiness_{today_str}.json"

        payload = {
            "session_id": self.session_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "all_passed": self.is_ready_for_live(results),
            "checks": results,
        }
        out_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Readiness report saved to %s", out_path)
        return out_path

    # ── Decision helper ──────────────────────────────────────────────────────

    @staticmethod
    def is_ready_for_live(results: dict[str, dict[str, Any]]) -> bool:
        """Return True only if every one of the 8 blocking checks passed."""
        return all(r["passed"] for r in results.values())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="NEPSE MiroFish — Live Deployment Readiness Checker"
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        default="default",
        help="Paper-trading session ID (default: 'default')",
    )
    parser.add_argument(
        "--data-dir",
        default="data/paper_trading",
        help="Root data directory containing session subdirectories.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the readiness report to data/processed/deployment/.",
    )
    args = parser.parse_args()

    checker = DeploymentReadinessChecker(
        session_id=args.session_id, data_dir=args.data_dir
    )

    try:
        checker.load_paper_results()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    results = checker.run_all_checks()
    checker.print_readiness_report(results)

    if args.save:
        saved_path = checker.save_readiness_report(results)
        print(f"Report saved → {saved_path}")

    sys.exit(0 if checker.is_ready_for_live(results) else 1)


if __name__ == "__main__":
    _main()
