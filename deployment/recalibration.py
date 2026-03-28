"""
deployment/recalibration.py
Strategy recalibration system for NEPSE MiroFish.

Defines WHEN and HOW to recalibrate:
  - Automatic triggers (checked daily)
  - Quarterly scheduled review
  - Manual override via Telegram /recalibrate command

Recalibration levels:
  MINOR   — adjust signal weights, no retraining
  MEDIUM  — re-optimise agent composition
  MAJOR   — full backtest re-run + deployment decision
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nepal Standard Time offset (UTC+5:45)
# ---------------------------------------------------------------------------
NST_OFFSET = timedelta(hours=5, minutes=45)


def _nst_now() -> datetime:
    """Return current time as NST (UTC+5:45)."""
    nst = timezone(NST_OFFSET)
    return datetime.now(tz=nst)


def _today_str() -> str:
    """Return today's date as an ISO string in NST."""
    return _nst_now().date().isoformat()


# ---------------------------------------------------------------------------
# Recalibration trigger definitions
# ---------------------------------------------------------------------------

RECALIBRATION_TRIGGERS: list[dict[str, Any]] = [
    {
        "name": "signal_accuracy_below_50",
        "condition": (
            "5-day rolling signal accuracy < 50% for 10 consecutive signals"
        ),
        "level": "MEDIUM",
        "action": "Re-optimise agent composition — run `make optimise-agents`",
        "auto_execute": True,
        "threshold": {
            "metric": "accuracy_5d",
            "value": 0.50,
            "consecutive_signals": 10,
        },
    },
    {
        "name": "sideways_regime_extended",
        "condition": (
            "Market regime detected as SIDEWAYS for > 20 consecutive trading days"
        ),
        "level": "MINOR",
        "action": "Adjust SCORE_WEIGHTS in strategy/watchlist.py for sideways conditions",
        "auto_execute": True,
        "threshold": {
            "metric": "consecutive_sideways_days",
            "value": 20,
        },
    },
    {
        "name": "mirofish_quality_flags",
        "condition": (
            "Quality flags present on > 30% of simulations over 7 consecutive days"
        ),
        "level": "MEDIUM",
        "action": "Review agent prompts and re-optimise — run `make optimise-agents`",
        "auto_execute": True,
        "threshold": {
            "metric": "quality_flag_rate",
            "value": 0.30,
            "consecutive_days": 7,
        },
    },
    {
        "name": "drawdown_threshold",
        "condition": "Portfolio drawdown exceeds 12% from peak",
        "level": "MAJOR",
        "action": (
            "Halt trading, run full review — `make backtest-full` + `make deploy-check`"
        ),
        "auto_execute": False,  # Requires human confirmation before halting
        "threshold": {
            "metric": "max_drawdown_pct",
            "value": 12.0,
        },
    },
    {
        "name": "nepse_index_divergence",
        "condition": (
            "Strategy cumulative return diverges > 20% from NEPSE index return"
        ),
        "level": "MAJOR",
        "action": (
            "Full backtest re-run — `make backtest-full` + `make deploy-check`"
        ),
        "auto_execute": False,  # Divergence can be positive; requires review
        "threshold": {
            "metric": "strategy_vs_index_divergence_pct",
            "value": 20.0,
        },
    },
    {
        "name": "quarterly_scheduled",
        "condition": "Every 3 months (90 days) since last major recalibration",
        "level": "MAJOR",
        "action": (
            "Scheduled comprehensive review — `make backtest-full` + update parameters"
        ),
        "auto_execute": True,
        "threshold": {
            "metric": "days_since_last_major",
            "value": 90,
        },
    },
]


# ---------------------------------------------------------------------------
# RecalibrationManager
# ---------------------------------------------------------------------------

class RecalibrationManager:
    """
    Evaluates recalibration triggers and executes calibration workflows.

    Recalibration levels:
      MINOR  — adjust SCORE_WEIGHTS in strategy/watchlist.py, no retraining.
      MEDIUM — re-optimise agent composition via `make optimise-agents`.
      MAJOR  — full backtest re-run + deployment decision check, trading halted.

    Parameters
    ----------
    session_id : str
        Current paper / live trading session identifier.
    data_dir : str
        Root path for paper trading data (default: "data/paper_trading").
    """

    def __init__(
        self,
        session_id: str,
        data_dir: str = "data/paper_trading",
    ) -> None:
        self.session_id = session_id
        self.data_dir = Path(data_dir)
        self.session_dir = self.data_dir / session_id
        self._root = Path(__file__).resolve().parent.parent
        self._log_path = self._root / "data" / "processed" / "recalibration_log.json"
        logger.info(
            "RecalibrationManager initialised — session_id=%s", session_id
        )

    # ── Trigger evaluation ──────────────────────────────────────────────────

    def check_all_triggers(self) -> list[dict]:
        """
        Evaluate every automatic trigger against current trading data.

        Returns
        -------
        list[dict]
            List of fired triggers, each with keys: name, level, action,
            auto_execute, current_value, threshold_value, fired.
        """
        fired: list[dict] = []
        metrics = self._load_current_metrics()

        for trigger in RECALIBRATION_TRIGGERS:
            if not trigger.get("auto_execute", False):
                continue
            result = self._evaluate_trigger(trigger, metrics)
            if result["fired"]:
                fired.append(result)
                logger.warning(
                    "TRIGGER FIRED: %s (level=%s) — %s",
                    trigger["name"], trigger["level"], trigger["condition"],
                )
            else:
                logger.debug(
                    "Trigger OK: %s (value=%.3f, threshold=%.3f)",
                    trigger["name"],
                    result.get("current_value", 0),
                    result.get("threshold_value", 0),
                )

        if not fired:
            logger.info("All recalibration triggers nominal.")
        return fired

    def _evaluate_trigger(self, trigger: dict, metrics: dict) -> dict:
        """Evaluate a single trigger against current metrics."""
        threshold = trigger.get("threshold", {})
        metric_name = threshold.get("metric", "")
        threshold_value = threshold.get("value", 0)
        current_value = metrics.get(metric_name, 0.0)

        fired = False

        if metric_name == "accuracy_5d":
            consecutive = metrics.get("consecutive_low_accuracy_signals", 0)
            required = threshold.get("consecutive_signals", 10)
            fired = (current_value < threshold_value) and (consecutive >= required)
        elif metric_name == "consecutive_sideways_days":
            fired = current_value >= threshold_value
        elif metric_name == "quality_flag_rate":
            consecutive_days = metrics.get("consecutive_flag_days", 0)
            required = threshold.get("consecutive_days", 7)
            fired = (current_value > threshold_value) and (consecutive_days >= required)
        elif metric_name == "max_drawdown_pct":
            fired = current_value >= threshold_value
        elif metric_name == "strategy_vs_index_divergence_pct":
            fired = abs(current_value) >= threshold_value
        elif metric_name == "days_since_last_major":
            fired = current_value >= threshold_value

        return {
            "name": trigger["name"],
            "level": trigger["level"],
            "action": trigger["action"],
            "auto_execute": trigger["auto_execute"],
            "condition": trigger["condition"],
            "metric": metric_name,
            "current_value": current_value,
            "threshold_value": threshold_value,
            "fired": fired,
        }

    def _load_current_metrics(self) -> dict:
        """
        Load current strategy metrics from snapshot and accuracy tracker data.
        Returns a dict keyed by metric name.
        """
        metrics: dict[str, float] = {}
        today = _today_str()

        # Accuracy metrics
        try:
            import glob
            acc_files = sorted(
                glob.glob(str(self.data_dir / "*" / "accuracy_report_*.json")),
                reverse=True,
            )
            if acc_files:
                with open(acc_files[0], encoding="utf-8") as fh:
                    acc = json.load(fh)
                metrics["accuracy_5d"] = acc.get("accuracy_last_5", 0.5)
                metrics["consecutive_low_accuracy_signals"] = acc.get(
                    "consecutive_below_50", 0
                )
        except Exception as exc:
            logger.debug("Could not load accuracy metrics: %s", exc)

        # Drawdown from latest snapshot
        try:
            snap_files = sorted(
                (self.data_dir).glob(f"*/snapshot_{today}.json"), reverse=True
            )
            if snap_files:
                with open(snap_files[0], encoding="utf-8") as fh:
                    snap = json.load(fh)
                # Return pct as reported; drawdown is negative return
                ret = snap.get("return_pct", 0.0)
                metrics["max_drawdown_pct"] = abs(min(ret, 0.0))
        except Exception as exc:
            logger.debug("Could not load snapshot metrics: %s", exc)

        # Regime — count consecutive sideways days
        try:
            import glob
            cycle_files = sorted(
                glob.glob(str(self.data_dir / "*" / "cycle_*.json")), reverse=True
            )
            sideways_streak = 0
            for cf in cycle_files[:30]:
                with open(cf, encoding="utf-8") as fh:
                    cycle = json.load(fh)
                regime_step = next(
                    (s for s in cycle.get("steps", []) if s.get("name") == "detect_regime"),
                    None,
                )
                if regime_step and regime_step.get("regime") == "SIDEWAYS":
                    sideways_streak += 1
                else:
                    break
            metrics["consecutive_sideways_days"] = float(sideways_streak)
        except Exception as exc:
            logger.debug("Could not load regime metrics: %s", exc)

        # Quality flag rate
        try:
            from db.models import get_engine
            from sqlalchemy import text
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT AVG(CASE WHEN quality_flags != '[]' THEN 1.0 ELSE 0.0 END) "
                        "FROM mirofish_signals WHERE date >= CURRENT_DATE - INTERVAL '7 days'"
                    )
                )
                flag_rate = result.scalar() or 0.0
            metrics["quality_flag_rate"] = float(flag_rate)
        except Exception:
            metrics["quality_flag_rate"] = 0.0

        # Days since last MAJOR recalibration
        metrics["days_since_last_major"] = float(self._days_since_last_major())

        # NEPSE index divergence (placeholder — computed in backtest module if available)
        try:
            from backtest.performance import get_strategy_vs_index_divergence
            metrics["strategy_vs_index_divergence_pct"] = get_strategy_vs_index_divergence(
                self.session_id
            )
        except Exception:
            metrics["strategy_vs_index_divergence_pct"] = 0.0

        return metrics

    def _days_since_last_major(self) -> int:
        """Return calendar days elapsed since the last MAJOR recalibration."""
        log = self._load_recalibration_log()
        major_entries = [
            e for e in log if e.get("level") == "MAJOR"
        ]
        if not major_entries:
            return 999  # Treat as long overdue on first run
        last_date_str = major_entries[-1].get("timestamp", "")[:10]
        try:
            last_date = date.fromisoformat(last_date_str)
            return (date.today() - last_date).days
        except Exception:
            return 999

    # ── Recalibration executors ─────────────────────────────────────────────

    def run_minor_recalibration(self) -> dict:
        """
        MINOR recalibration: adjust SCORE_WEIGHTS in strategy/watchlist.py.

        Reads the current weights, applies conservative dampening toward
        equal weighting, and writes them back. Does not retrain any model.

        Returns
        -------
        dict
            Recalibration result with old and new weights.
        """
        logger.info("Running MINOR recalibration — adjusting SCORE_WEIGHTS.")
        result: dict[str, Any] = {
            "level": "MINOR",
            "trigger": "weight_adjustment",
            "timestamp": _nst_now().isoformat(),
            "status": "STARTED",
        }

        watchlist_path = self._root / "strategy" / "watchlist.py"
        if not watchlist_path.exists():
            result["status"] = "SKIPPED"
            result["reason"] = f"watchlist.py not found at {watchlist_path}"
            logger.warning("MINOR recalibration skipped: %s", result["reason"])
            return result

        try:
            import ast
            import re

            src = watchlist_path.read_text(encoding="utf-8")
            # Extract SCORE_WEIGHTS dict via regex (handles simple inline dicts)
            pattern = r"SCORE_WEIGHTS\s*=\s*\{([^}]+)\}"
            match = re.search(pattern, src)
            if not match:
                result["status"] = "SKIPPED"
                result["reason"] = "SCORE_WEIGHTS dict not found in watchlist.py"
                return result

            weights_str = "{" + match.group(1) + "}"
            old_weights: dict = ast.literal_eval(weights_str)

            # Dampen toward equal weight (move 10% toward 1/N)
            n = len(old_weights)
            equal_weight = 1.0 / n if n > 0 else 1.0
            new_weights = {
                k: round(v * 0.9 + equal_weight * 0.1, 4)
                for k, v in old_weights.items()
            }

            # Normalise so weights sum to ~1.0
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

            # Reconstruct the SCORE_WEIGHTS block
            new_block = "SCORE_WEIGHTS = {\n"
            for k, v in new_weights.items():
                new_block += f'    "{k}": {v},\n'
            new_block += "}"

            new_src = re.sub(pattern, new_block, src)
            watchlist_path.write_text(new_src, encoding="utf-8")

            result["status"] = "COMPLETED"
            result["old_weights"] = old_weights
            result["new_weights"] = new_weights
            logger.info("MINOR recalibration complete — weights adjusted in %s", watchlist_path.name)

        except Exception as exc:
            result["status"] = "FAILED"
            result["error"] = str(exc)
            logger.error("MINOR recalibration failed: %s", exc, exc_info=True)

        self.log_recalibration("MINOR", "weight_adjustment", result)
        return result

    def run_medium_recalibration(self) -> dict:
        """
        MEDIUM recalibration: trigger `make optimise-agents` and update agent mix.

        Returns
        -------
        dict
            Recalibration result including subprocess output.
        """
        logger.info("Running MEDIUM recalibration — re-optimising agents.")
        result: dict[str, Any] = {
            "level": "MEDIUM",
            "trigger": "agent_optimisation",
            "timestamp": _nst_now().isoformat(),
            "status": "STARTED",
        }

        try:
            proc = subprocess.run(
                ["make", "optimise-agents"],
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=600,  # 10-minute timeout
            )
            result["returncode"] = proc.returncode
            result["stdout_tail"] = proc.stdout[-1000:] if proc.stdout else ""
            result["stderr_tail"] = proc.stderr[-500:] if proc.stderr else ""

            if proc.returncode == 0:
                result["status"] = "COMPLETED"
                logger.info("MEDIUM recalibration complete — agents re-optimised.")
            else:
                result["status"] = "FAILED"
                logger.error(
                    "make optimise-agents failed (rc=%d): %s",
                    proc.returncode, proc.stderr[:300],
                )
        except subprocess.TimeoutExpired:
            result["status"] = "TIMEOUT"
            result["error"] = "make optimise-agents exceeded 600s timeout"
            logger.error("MEDIUM recalibration timed out.")
        except Exception as exc:
            result["status"] = "FAILED"
            result["error"] = str(exc)
            logger.error("MEDIUM recalibration failed: %s", exc, exc_info=True)

        self.log_recalibration("MEDIUM", "agent_optimisation", result)
        return result

    def run_major_recalibration(self) -> dict:
        """
        MAJOR recalibration: run full backtest + deployment check, halt trading.

        Executes `make backtest-full` followed by `make deploy-check`.
        Sets a HALT flag in the session directory to pause live order routing.

        Returns
        -------
        dict
            Recalibration result with backtest and deploy-check outputs.
        """
        logger.warning("Running MAJOR recalibration — halting trading + full backtest.")
        result: dict[str, Any] = {
            "level": "MAJOR",
            "trigger": "full_review",
            "timestamp": _nst_now().isoformat(),
            "status": "STARTED",
            "trading_halted": False,
        }

        # Write halt flag
        halt_file = self.session_dir / "TRADING_HALTED.flag"
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            halt_file.write_text(
                json.dumps({
                    "halted_at": _nst_now().isoformat(),
                    "reason": "MAJOR recalibration triggered",
                    "session_id": self.session_id,
                }),
                encoding="utf-8",
            )
            result["trading_halted"] = True
            logger.warning("Trading HALTED — flag written to %s", halt_file)
        except Exception as exc:
            logger.error("Failed to write halt flag: %s", exc)

        # Run backtest
        backtest_result = self._run_make_target("backtest-full", timeout=1800)
        result["backtest"] = backtest_result

        # Run deploy-check
        deploy_result = self._run_make_target("deploy-check", timeout=300)
        result["deploy_check"] = deploy_result

        both_ok = (
            backtest_result.get("returncode", 1) == 0
            and deploy_result.get("returncode", 1) == 0
        )
        result["status"] = "COMPLETED" if both_ok else "COMPLETED_WITH_ERRORS"

        logger.warning(
            "MAJOR recalibration %s. Trading remains HALTED pending manual review.",
            result["status"],
        )

        self.log_recalibration("MAJOR", "full_review", result)
        return result

    def _run_make_target(self, target: str, timeout: int = 600) -> dict:
        """Run a Makefile target and return a result dict."""
        logger.info("Running: make %s", target)
        try:
            proc = subprocess.run(
                ["make", target],
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "target": target,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
                "ok": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            logger.error("make %s timed out after %ds.", target, timeout)
            return {"target": target, "returncode": -1, "ok": False, "error": "timeout"}
        except Exception as exc:
            logger.error("make %s failed: %s", target, exc)
            return {"target": target, "returncode": -1, "ok": False, "error": str(exc)}

    # ── Dispatcher ──────────────────────────────────────────────────────────

    def run_recalibration(self, level: str) -> dict:
        """
        Dispatch to the appropriate recalibration method by level.

        Parameters
        ----------
        level : str
            One of "MINOR", "MEDIUM", or "MAJOR".

        Returns
        -------
        dict
            Result dict from the chosen recalibration method.
        """
        level = level.upper()
        if level == "MINOR":
            return self.run_minor_recalibration()
        elif level == "MEDIUM":
            return self.run_medium_recalibration()
        elif level == "MAJOR":
            return self.run_major_recalibration()
        else:
            raise ValueError(
                f"Unknown recalibration level '{level}'. "
                "Must be one of: MINOR, MEDIUM, MAJOR"
            )

    # ── Logging ─────────────────────────────────────────────────────────────

    def log_recalibration(
        self,
        level: str,
        trigger: str,
        result: dict,
    ) -> None:
        """
        Append a recalibration event to data/processed/recalibration_log.json.

        Parameters
        ----------
        level : str
            MINOR / MEDIUM / MAJOR.
        trigger : str
            Name of the trigger or manual override label.
        result : dict
            Result dict from the recalibration run.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        log: list[dict] = self._load_recalibration_log()
        entry = {
            "timestamp": _nst_now().isoformat(),
            "session_id": self.session_id,
            "level": level,
            "trigger": trigger,
            "status": result.get("status", "UNKNOWN"),
            "summary": {k: v for k, v in result.items() if k not in ("stdout_tail", "stderr_tail")},
        }
        log.append(entry)

        with open(self._log_path, "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2, default=str)

        logger.info(
            "Recalibration logged: level=%s  trigger=%s  status=%s",
            level, trigger, entry["status"],
        )

    def _load_recalibration_log(self) -> list[dict]:
        """Load the existing recalibration log from disk."""
        if not self._log_path.exists():
            return []
        try:
            with open(self._log_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Could not load recalibration log: %s", exc)
            return []

    # ── Quarterly scheduler ─────────────────────────────────────────────────

    def schedule_quarterly_review(self, session_id: str) -> None:
        """
        Start an APScheduler job that triggers a MAJOR recalibration every 90 days.

        The job runs at 08:30 NST (02:45 UTC) on Sunday — the first NEPSE trading
        day of the week — whenever 90+ days have elapsed since the last MAJOR review.
        """
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("apscheduler not installed. Run: pip install apscheduler")
            return

        def _quarterly_job() -> None:
            days = self._days_since_last_major()
            if days >= 90:
                logger.info(
                    "Quarterly review due (%d days since last MAJOR) — triggering.", days
                )
                self.run_major_recalibration()
            else:
                logger.info(
                    "Quarterly check: %d days since last MAJOR review (next in %d days).",
                    days, 90 - days,
                )

        scheduler = BlockingScheduler(timezone="UTC")
        # Run every Sunday at 02:45 UTC (08:30 NST) — checks 90-day threshold
        scheduler.add_job(
            _quarterly_job,
            CronTrigger(hour=2, minute=45, day_of_week="sun"),
            id="quarterly_review",
            name="Quarterly strategy review (checks 90-day threshold)",
            misfire_grace_time=3600,
        )

        logger.info(
            "Quarterly review scheduler started (checks every Sunday 08:30 NST)."
        )
        print("\n  Quarterly recalibration scheduler running.")
        print("  Every Sunday 08:30 NST — checks 90-day threshold.")
        print("  Press Ctrl+C to stop.\n")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Quarterly review scheduler stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="NEPSE MiroFish strategy recalibration manager"
    )
    parser.add_argument(
        "--session-id", default="default",
        help="Paper/live trading session ID (default: 'default')",
    )
    parser.add_argument(
        "--check-triggers", action="store_true",
        help="Evaluate all automatic triggers against current data and exit",
    )
    parser.add_argument(
        "--run-minor", action="store_true",
        help="Run MINOR recalibration immediately (adjust signal weights)",
    )
    parser.add_argument(
        "--run-medium", action="store_true",
        help="Run MEDIUM recalibration immediately (re-optimise agents)",
    )
    parser.add_argument(
        "--run-major", action="store_true",
        help="Run MAJOR recalibration immediately (full backtest + halt trading)",
    )
    parser.add_argument(
        "--schedule-quarterly", action="store_true",
        help="Start the quarterly review scheduler (blocking)",
    )
    args = parser.parse_args()

    # Bootstrap .env if present
    _root = Path(__file__).resolve().parent.parent
    env_file = _root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    manager = RecalibrationManager(session_id=args.session_id)

    if args.check_triggers:
        fired = manager.check_all_triggers()
        if fired:
            print(f"\n{len(fired)} trigger(s) fired:")
            for t in fired:
                print(f"  [{t['level']}] {t['name']}: {t['action']}")
        else:
            print("All triggers nominal — no recalibration required.")

    elif args.run_minor:
        result = manager.run_recalibration("MINOR")
        print(json.dumps(result, indent=2, default=str))

    elif args.run_medium:
        result = manager.run_recalibration("MEDIUM")
        print(json.dumps(result, indent=2, default=str))

    elif args.run_major:
        print("WARNING: MAJOR recalibration will HALT trading. Proceeding...")
        result = manager.run_recalibration("MAJOR")
        print(json.dumps(result, indent=2, default=str))

    elif args.schedule_quarterly:
        manager.schedule_quarterly_review(session_id=args.session_id)

    else:
        # Default: check triggers and print summary
        print(f"RecalibrationManager — session: {args.session_id}")
        print(f"Log path: {manager._log_path}")
        print(f"Days since last MAJOR: {manager._days_since_last_major()}")
        print(f"\n{len(RECALIBRATION_TRIGGERS)} triggers defined:")
        for t in RECALIBRATION_TRIGGERS:
            auto = "auto" if t["auto_execute"] else "manual"
            print(f"  [{t['level']:6}|{auto:6}] {t['name']}")
        print("\nUse --check-triggers to evaluate current state.")


if __name__ == "__main__":
    main()
