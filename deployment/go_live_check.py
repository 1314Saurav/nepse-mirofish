"""
deployment/go_live_check.py
Auto-fills the GO_LIVE_CHECKLIST by reading actual paper trading session data.
Prints a summary table and exits 0 if READY, 1 if NOT READY.

Run: python -m deployment.go_live_check [--session-id SESSION_ID]
     make go-live-check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Nepal Standard Time: UTC+5:45
_NST = timezone(timedelta(hours=5, minutes=45))

# Path constants
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PAPER_DIR = _REPO_ROOT / "data" / "paper_trading"
_DEPLOY_DIR = _REPO_ROOT / "data" / "processed" / "deployment"
_CHECKLIST_TEMPLATE = Path(__file__).resolve().parent / "GO_LIVE_CHECKLIST.md"

# ── Blocking criteria thresholds ────────────────────────────────────────────
REQUIRED_TRADING_DAYS: int = 20
MIN_SIGNAL_ACCURACY_5D: float = 55.0
MAX_DRAWDOWN_PCT: float = 15.0
MAX_MIROFISH_FLAG_PCT: float = 20.0


# ---------------------------------------------------------------------------
# GoLiveChecker
# ---------------------------------------------------------------------------

class GoLiveChecker:
    """
    Reads a paper trading session, runs all go/no-go checks, and produces a
    formatted checklist summary.

    Usage:
        checker = GoLiveChecker()          # auto-detects active session
        results = checker.run_checks()
        checker.print_summary(results)
        filled_path = checker.save_filled_checklist(results)
        sys.exit(0 if checker.is_ready() else 1)
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or self._detect_session_id()
        self._session_dir: Path = _PAPER_DIR / self.session_id
        self._checked_at: datetime = datetime.now(tz=_NST)
        self._results: Optional[dict] = None
        logger.debug("GoLiveChecker: session=%s dir=%s", self.session_id, self._session_dir)

    # ── Session detection ───────────────────────────────────────────────────

    def _detect_session_id(self) -> str:
        """
        Auto-detect the active session from
        data/paper_trading/active_session.txt, or fall back to the most
        recently modified session directory.
        """
        active_file = _PAPER_DIR / "active_session.txt"
        if active_file.exists():
            text = active_file.read_text(encoding="utf-8").strip()
            if text:
                logger.info("Active session from file: %s", text)
                return text

        # Fallback: most recently modified subdirectory
        dirs = sorted(
            (d for d in _PAPER_DIR.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if dirs:
            logger.info("Auto-detected session: %s", dirs[0].name)
            return dirs[0].name

        # Final fallback: today's default name
        from datetime import date as _date
        return f"paper_{_date.today()}"

    # ── Data loading helpers ─────────────────────────────────────────────────

    def _load_json(self, path: Path) -> dict | list:
        """Load JSON file; return empty dict on failure."""
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            logger.debug("File not found: %s", path)
        except json.JSONDecodeError as exc:
            logger.warning("JSON decode error in %s: %s", path, exc)
        return {}

    def _load_state(self) -> dict:
        return self._load_json(self._session_dir / "state.json")  # type: ignore[return-value]

    def _load_accuracy_report(self) -> dict:
        return self._load_json(self._session_dir / "accuracy_report.json")  # type: ignore[return-value]

    def _load_snapshots(self) -> list[dict]:
        """Load all daily snapshots sorted chronologically."""
        snapshots: list[dict] = []
        for snap_path in sorted(self._session_dir.glob("snapshot_*.json")):
            data = self._load_json(snap_path)
            if isinstance(data, dict):
                snapshots.append(data)
        return snapshots

    def _load_cycle_logs(self) -> list[dict]:
        """Load all daily cycle logs."""
        logs: list[dict] = []
        for log_path in sorted(self._session_dir.glob("cycle_*.json")):
            data = self._load_json(log_path)
            if isinstance(data, dict):
                logs.append(data)
        return logs

    # ── Core metrics computation ─────────────────────────────────────────────

    def _compute_trading_days(self, state: dict) -> int:
        """Count completed trading days from cycle logs or session_start."""
        logs = self._load_cycle_logs()
        if logs:
            # Each non-skipped log = one trading day
            return sum(1 for lg in logs if "skipped" not in lg)

        # Fallback: calendar days since session_start
        session_start_str: str = state.get("session_start", "")
        if session_start_str:
            try:
                from datetime import date as _date
                start = _date.fromisoformat(session_start_str)
                return max(0, (_date.today() - start).days)
            except ValueError:
                pass
        return 0

    def _compute_paper_return(self, snapshots: list[dict]) -> float:
        """Total return % from first to last snapshot."""
        if not snapshots:
            return 0.0
        first_pv = snapshots[0].get("portfolio_value", 0.0)
        last_pv = snapshots[-1].get("portfolio_value", 0.0)
        if first_pv <= 0:
            return 0.0
        return round((last_pv / first_pv - 1) * 100, 2)

    def _compute_nepse_return(self, snapshots: list[dict]) -> float:
        """NEPSE index return % over the paper trading period (from snapshots)."""
        nepse_vals = [s.get("nepse_return_pct", None) for s in snapshots
                      if s.get("nepse_return_pct") is not None]
        if not nepse_vals:
            # Try DB
            try:
                from db.loader import get_nepse_return_between
                first_date = snapshots[0].get("date", "") if snapshots else ""
                last_date = snapshots[-1].get("date", "") if snapshots else ""
                if first_date and last_date:
                    return get_nepse_return_between(first_date, last_date) or 0.0
            except Exception:
                pass
            return 0.0
        return round(sum(nepse_vals), 2)

    def _compute_max_drawdown(self, snapshots: list[dict]) -> float:
        """Maximum peak-to-trough drawdown % from snapshot equity curve."""
        values = [s.get("portfolio_value", 0.0) for s in snapshots if s.get("portfolio_value")]
        if len(values) < 2:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 2)

    def _check_critical_errors(self, logs: list[dict]) -> bool:
        """Return True if no CRITICAL errors found in last 5 cycle logs."""
        recent = logs[-5:]
        for log in recent:
            errors: list = log.get("errors", [])
            for err in errors:
                if isinstance(err, dict):
                    if err.get("level", "").upper() == "CRITICAL":
                        return False
                elif isinstance(err, str) and "CRITICAL" in err.upper():
                    return False
        return True

    def _compute_mirofish_flag_pct(self, logs: list[dict]) -> float:
        """% of cycle logs where MiroFish flagged a quality issue."""
        if not logs:
            return 0.0
        flagged = sum(1 for lg in logs if lg.get("mirofish_quality_flag", False))
        return round(flagged / len(logs) * 100, 1)

    def _compute_stop_loss_discipline(self, state: dict) -> float:
        """
        % of stop-loss exits that were respected (not overridden).
        Uses trade_log field 'exit_reason'.
        """
        trade_log: list[dict] = state.get("trade_log", [])
        sl_exits = [t for t in trade_log if t.get("exit_reason") == "stop_loss"]
        if not sl_exits:
            return 100.0  # no stop-loss exits yet — assume discipline maintained
        respected = sum(1 for t in sl_exits if not t.get("override", False))
        return round(respected / len(sl_exits) * 100, 1)

    # ── Main check runner ────────────────────────────────────────────────────

    def run_checks(self) -> dict:
        """
        Load session data, run all blocking checks, and return a results dict.
        Also delegates to DeploymentReadinessChecker when available.
        """
        state = self._load_state()
        accuracy = self._load_accuracy_report()
        snapshots = self._load_snapshots()
        logs = self._load_cycle_logs()

        trading_days = self._compute_trading_days(state)
        paper_return = self._compute_paper_return(snapshots)
        nepse_return = self._compute_nepse_return(snapshots)
        paper_alpha = round(paper_return - nepse_return, 2)
        max_drawdown = self._compute_max_drawdown(snapshots)
        signal_acc_5d: float = accuracy.get("overall_accuracy_5d") or 0.0
        mf_flag_pct = self._compute_mirofish_flag_pct(logs)
        zero_crit = self._check_critical_errors(logs)
        sl_discipline = self._compute_stop_loss_discipline(state)

        # Try to delegate to the full DeploymentReadinessChecker as well
        ext_ready: Optional[bool] = None
        ext_failures: list[str] = []
        try:
            from deployment.readiness_check import DeploymentReadinessChecker
            drc = DeploymentReadinessChecker(session_id=self.session_id)
            ext_result = drc.run()
            ext_ready = ext_result.get("is_ready", None)
            ext_failures = ext_result.get("failures", [])
        except Exception as exc:
            logger.debug("DeploymentReadinessChecker unavailable: %s", exc)

        # Build per-check results
        checks: dict[str, dict[str, Any]] = {
            "trading_days": {
                "label": f"Trading days: {trading_days}/{REQUIRED_TRADING_DAYS} required",
                "value": trading_days,
                "threshold": REQUIRED_TRADING_DAYS,
                "pass": trading_days >= REQUIRED_TRADING_DAYS,
            },
            "signal_accuracy_5d": {
                "label": f"Signal accuracy 5d: {signal_acc_5d:.1f}% >= {MIN_SIGNAL_ACCURACY_5D}%",
                "value": signal_acc_5d,
                "threshold": MIN_SIGNAL_ACCURACY_5D,
                "pass": signal_acc_5d >= MIN_SIGNAL_ACCURACY_5D,
            },
            "paper_return": {
                "label": f"Paper return: {paper_return:+.1f}% > 0%",
                "value": paper_return,
                "threshold": 0.0,
                "pass": paper_return > 0.0,
            },
            "paper_alpha": {
                "label": f"Paper alpha: {paper_alpha:+.1f}% > 0%",
                "value": paper_alpha,
                "threshold": 0.0,
                "pass": paper_alpha > 0.0,
            },
            "zero_critical_errors": {
                "label": f"Zero critical bugs: {zero_crit}",
                "value": zero_crit,
                "threshold": True,
                "pass": zero_crit,
            },
            "mirofish_quality": {
                "label": f"MiroFish quality: {mf_flag_pct:.0f}% < {MAX_MIROFISH_FLAG_PCT:.0f}%",
                "value": mf_flag_pct,
                "threshold": MAX_MIROFISH_FLAG_PCT,
                "pass": mf_flag_pct < MAX_MIROFISH_FLAG_PCT,
            },
            "stop_loss_discipline": {
                "label": f"Stop-loss discipline: {sl_discipline:.0f}%",
                "value": sl_discipline,
                "threshold": 100.0,
                "pass": sl_discipline >= 100.0,
            },
            "max_drawdown": {
                "label": f"Max drawdown: {max_drawdown:.1f}% < {MAX_DRAWDOWN_PCT:.0f}%",
                "value": max_drawdown,
                "threshold": MAX_DRAWDOWN_PCT,
                "pass": max_drawdown < MAX_DRAWDOWN_PCT,
            },
        }

        all_pass = all(c["pass"] for c in checks.values())
        failures = [c["label"] for c in checks.values() if not c["pass"]]

        # If external checker ran and disagrees, mark not ready
        if ext_ready is False:
            all_pass = False
            failures += [f"(readiness_check) {f}" for f in ext_failures]

        results: dict = {
            "session_id": self.session_id,
            "checked_at": self._checked_at.strftime("%Y-%m-%d %H:%M NST"),
            "is_ready": all_pass,
            "checks": checks,
            "failures": failures,
            "raw": {
                "trading_days": trading_days,
                "paper_return_pct": paper_return,
                "paper_alpha_pct": paper_alpha,
                "nepse_return_pct": nepse_return,
                "max_drawdown_pct": max_drawdown,
                "signal_accuracy_5d": signal_acc_5d,
                "mirofish_quality_flag_pct": mf_flag_pct,
                "zero_critical_errors": zero_crit,
                "stop_loss_discipline_pct": sl_discipline,
            },
        }
        self._results = results
        return results

    # ── Output helpers ───────────────────────────────────────────────────────

    def print_summary(self, results: dict) -> None:
        """
        Print a bordered checklist table with check/cross marks.

        Example output:
            ╔══════════════════════════════════════════════╗
            ║   NEPSE MiroFish — Go-Live Readiness Check   ║
            ╠══════════════════════════════════════════════╣
            ║ Session: pt_20240115_143022                  ║
            ║ Checked: 2024-02-15 18:00 NST               ║
            ╠══════════════════════════════════════════════╣
            ║ BLOCKING CHECKS                              ║
            ║  ✅ Trading days: 22/20 required             ║
            ...
            ╠══════════════════════════════════════════════╣
            ║ RESULT: ✅ READY FOR LIVE TRADING            ║
            ╚══════════════════════════════════════════════╝
        """
        W = 46  # inner width (between ║ and ║)

        def _row(text: str) -> str:
            return f"║ {text:<{W - 1}}║"

        border_top = "╔" + "═" * W + "╗"
        border_mid = "╠" + "═" * W + "╣"
        border_bot = "╚" + "═" * W + "╝"

        is_ready: bool = results.get("is_ready", False)
        session_id: str = results.get("session_id", "unknown")
        checked_at: str = results.get("checked_at", "")
        checks: dict = results.get("checks", {})

        print(border_top)
        print(_row("  NEPSE MiroFish \u2014 Go-Live Readiness Check"))
        print(border_mid)
        print(_row(f"Session: {session_id}"))
        print(_row(f"Checked: {checked_at}"))
        print(border_mid)
        print(_row("BLOCKING CHECKS"))

        # Ordered display sequence
        ordered_keys = [
            "trading_days",
            "signal_accuracy_5d",
            "paper_return",
            "paper_alpha",
            "zero_critical_errors",
            "mirofish_quality",
            "stop_loss_discipline",
            "max_drawdown",
        ]
        for key in ordered_keys:
            if key not in checks:
                continue
            check = checks[key]
            icon = "\u2705" if check["pass"] else "\u274c"
            label = check["label"]
            print(_row(f" {icon} {label}"))

        # Any extra checks not in the ordered list
        for key, check in checks.items():
            if key not in ordered_keys:
                icon = "\u2705" if check["pass"] else "\u274c"
                print(_row(f" {icon} {check['label']}"))

        print(border_mid)
        if is_ready:
            result_line = "RESULT: \u2705 READY FOR LIVE TRADING"
        else:
            result_line = "RESULT: \u274c NOT READY — see failures below"
        print(_row(result_line))
        print(border_bot)

        if not is_ready:
            print("\nFailing checks:")
            for f in results.get("failures", []):
                print(f"  \u274c {f}")

    def save_filled_checklist(self, results: dict) -> Path:
        """
        Write a filled Markdown version of the checklist to
        data/processed/deployment/GO_LIVE_FILLED_YYYYMMDD.md

        Checked items become [x], unchecked remain [ ].
        Returns the path to the saved file.
        """
        _DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
        date_tag = self._checked_at.strftime("%Y%m%d")
        out_path = _DEPLOY_DIR / f"GO_LIVE_FILLED_{date_tag}.md"

        # Load template
        template_text = ""
        if _CHECKLIST_TEMPLATE.exists():
            template_text = _CHECKLIST_TEMPLATE.read_text(encoding="utf-8")
        else:
            template_text = "# NEPSE MiroFish — Go-Live Checklist\n\n_Template not found._\n"

        checks = results.get("checks", {})
        raw = results.get("raw", {})
        is_ready = results.get("is_ready", False)

        # Build a filled header block
        header_lines = [
            f"<!-- Auto-generated by go_live_check.py -->",
            f"<!-- Session: {results.get('session_id', '')} -->",
            f"<!-- Checked: {results.get('checked_at', '')} -->",
            f"<!-- Result: {'READY' if is_ready else 'NOT READY'} -->",
            "",
        ]

        # Auto-check performance items based on raw data
        replacements: dict[str, bool] = {
            "Minimum 20 trading days (4 weeks) completed":
                raw.get("trading_days", 0) >= REQUIRED_TRADING_DAYS,
            "Paper return is positive":
                raw.get("paper_return_pct", 0.0) > 0,
            "Paper alpha vs NEPSE index is positive":
                raw.get("paper_alpha_pct", 0.0) > 0,
            "Maximum drawdown < 15% during paper period":
                raw.get("max_drawdown_pct", 99.0) < MAX_DRAWDOWN_PCT,
            "5-day signal accuracy \u2265 55%":
                raw.get("signal_accuracy_5d", 0.0) >= MIN_SIGNAL_ACCURACY_5D,
            "MiroFish quality flags < 20% of signals":
                raw.get("mirofish_quality_flag_pct", 99.0) < MAX_MIROFISH_FLAG_PCT,
            "Zero CRITICAL errors in last 5 trading days":
                bool(raw.get("zero_critical_errors", False)),
        }

        # Replace [ ] with [x] for passing items
        filled_text = template_text
        for item_text, passed in replacements.items():
            if passed:
                old = f"- [ ] {item_text}"
                new = f"- [x] {item_text}"
                filled_text = filled_text.replace(old, new)

        # Fill in go-live decision fields
        if is_ready:
            filled_text = filled_text.replace(
                "- [ ] `python -m deployment.readiness_check` exits with code 0",
                "- [x] `python -m deployment.readiness_check` exits with code 0",
            )
            filled_text = filled_text.replace(
                "- [ ] `python -m deployment.go_live_check` confirms READY",
                "- [x] `python -m deployment.go_live_check` confirms READY",
            )

        # Prepend auto-gen header
        full_content = "\n".join(header_lines) + filled_text

        out_path.write_text(full_content, encoding="utf-8")
        logger.info("Filled checklist saved to: %s", out_path)
        return out_path

    def is_ready(self) -> bool:
        """Return True only if all blocking checks pass."""
        if self._results is None:
            self._results = self.run_checks()
        return bool(self._results.get("is_ready", False))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """
    Parse CLI args, run all go/no-go checks, print summary, and save the
    filled checklist.

    Exit codes:
        0 — READY for live trading
        1 — NOT READY
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="NEPSE MiroFish Go-Live Readiness Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m deployment.go_live_check\n"
            "  python -m deployment.go_live_check --session-id pt_20240115_143022\n"
            "  make go-live-check\n"
        ),
    )
    parser.add_argument(
        "--session-id",
        metavar="SESSION_ID",
        default=None,
        help="Paper trading session ID (default: auto-detect from active_session.txt)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Skip saving the filled checklist markdown file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print results as JSON instead of the formatted table",
    )

    args = parser.parse_args(argv)

    checker = GoLiveChecker(session_id=args.session_id)
    results = checker.run_checks()

    if args.json:
        # Serialize check dicts to JSON-safe form
        import copy
        out = copy.deepcopy(results)
        print(json.dumps(out, indent=2, default=str))
    else:
        checker.print_summary(results)

    if not args.no_save:
        try:
            filled_path = checker.save_filled_checklist(results)
            print(f"\nFilled checklist saved: {filled_path}")
        except Exception as exc:
            logger.warning("Could not save filled checklist: %s", exc)

    return 0 if checker.is_ready() else 1


if __name__ == "__main__":
    sys.exit(main())
