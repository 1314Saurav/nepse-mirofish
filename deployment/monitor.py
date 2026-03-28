"""
deployment/monitor.py
Production monitoring for NEPSE MiroFish live trading.

Scheduled checks:
  08:00 NST — morning health check (before market open)
  16:30 NST — post-market summary
  19:00 NST — evening system check

Converts to UTC: 08:00 NST = 02:15 UTC, 16:30 NST = 10:45 UTC, 19:00 NST = 13:15 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

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
# Monitoring check registry
# ---------------------------------------------------------------------------

MONITORING_CHECKS: dict[str, str] = {
    "database_connection": "PostgreSQL connectivity",
    "nepse_data_freshness": "Stock prices updated today",
    "mirofish_simulation": "Simulation ran successfully today",
    "signal_generated": "Signal extracted and saved",
    "portfolio_snapshot": "Daily snapshot saved",
    "telegram_bot": "Bot is responsive",
    "disk_space": "Data directory < 80% full",
    "api_keys_valid": "ANTHROPIC + GROQ keys respond",
    "log_errors": "No CRITICAL errors in today's log",
    "zep_memory": "Zep API accessible",
}


# ---------------------------------------------------------------------------
# ProductionMonitor
# ---------------------------------------------------------------------------

class ProductionMonitor:
    """
    Runs health checks and sends scheduled monitoring summaries via Telegram.

    Scheduled windows (all NEPSE trading days — Sun to Thu):
      08:00 NST (02:15 UTC) — morning health check before market open
      16:30 NST (10:45 UTC) — post-market P&L and signal summary
      19:00 NST (13:15 UTC) — light evening system check

    Parameters
    ----------
    session_id : str
        Paper / live trading session identifier.
    data_dir : str
        Root directory for paper trading data (default: "data/paper_trading").
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
        logger.info(
            "ProductionMonitor initialised — session_id=%s  data_dir=%s",
            session_id, self.data_dir,
        )

    # ── Individual checks ───────────────────────────────────────────────────

    def check_database_connection(self) -> tuple[bool, str]:
        """Verify PostgreSQL connectivity via SQLAlchemy."""
        try:
            from db.models import get_engine
            from sqlalchemy import text
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "PostgreSQL connection OK"
        except Exception as exc:
            return False, f"DB connection FAILED: {exc}"

    def check_data_freshness(self) -> tuple[bool, str]:
        """
        Confirm today's stock prices exist in the stock_prices table.
        Returns True if at least one row has today's date.
        """
        today = _today_str()
        try:
            from db.models import get_engine
            from sqlalchemy import text
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM stock_prices WHERE date = :d"),
                    {"d": today},
                )
                count = result.scalar() or 0
            if count > 0:
                return True, f"Data fresh — {count} rows for {today}"
            return False, f"No stock_prices rows for {today}"
        except Exception as exc:
            return False, f"Data freshness check failed: {exc}"

    def check_simulation_ran(self) -> tuple[bool, str]:
        """
        Check whether a daily cycle log was written today.
        Looks in <session_dir>/cycle_<today>.json.
        """
        today = _today_str()
        cycle_file = self.session_dir / f"cycle_{today}.json"
        if cycle_file.exists():
            return True, f"Simulation cycle log found: {cycle_file.name}"

        # Fallback: look for cycle files in any session sub-dir
        candidates = sorted(self.data_dir.glob(f"*/cycle_{today}.json"), reverse=True)
        if candidates:
            return True, f"Simulation cycle log found: {candidates[0]}"
        return False, f"No cycle log for {today} in {self.data_dir}"

    def check_signal_generated(self) -> tuple[bool, str]:
        """
        Verify today's signal exists in the mirofish_signals DB table
        and/or in data/processed/signals/.
        """
        today = _today_str()
        # Check DB first
        try:
            from db.models import get_engine
            from sqlalchemy import text
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM mirofish_signals WHERE date = :d"),
                    {"d": today},
                )
                count = result.scalar() or 0
            if count > 0:
                return True, f"Signal in DB for {today} ({count} row(s))"
        except Exception:
            pass

        # Fallback: file system check
        signals_dir = self._root / "data" / "processed" / "signals"
        candidates = sorted(signals_dir.glob(f"*{today}*.json")) if signals_dir.exists() else []
        if candidates:
            return True, f"Signal file found: {candidates[-1].name}"
        return False, f"No signal found for {today}"

    def check_portfolio_snapshot(self) -> tuple[bool, str]:
        """
        Confirm today's portfolio snapshot JSON exists.
        Looks in <session_dir>/snapshot_<today>.json.
        """
        today = _today_str()
        snap_file = self.session_dir / f"snapshot_{today}.json"
        if snap_file.exists():
            return True, f"Snapshot found: {snap_file.name}"

        candidates = sorted(self.data_dir.glob(f"*/snapshot_{today}.json"), reverse=True)
        if candidates:
            return True, f"Snapshot found: {candidates[0]}"
        return False, f"No snapshot for {today}"

    def check_disk_space(self) -> tuple[bool, str]:
        """
        Check that the data directory uses less than 80% of its partition.
        """
        check_path = self.data_dir if self.data_dir.exists() else self._root
        try:
            usage = shutil.disk_usage(check_path)
            used_pct = usage.used / usage.total * 100
            free_gb = usage.free / 1024 ** 3
            if used_pct < 80:
                return True, f"Disk OK — {used_pct:.1f}% used, {free_gb:.1f} GB free"
            return False, f"Disk WARNING — {used_pct:.1f}% used (threshold: 80%)"
        except Exception as exc:
            return False, f"Disk check failed: {exc}"

    def check_log_errors(self) -> tuple[bool, str]:
        """
        Scan today's pipeline log file for CRITICAL-level entries.
        """
        today = _today_str()
        log_dir = self._root / "logs"
        log_file = log_dir / f"pipeline_{today}.log"
        if not log_file.exists():
            return True, f"No log file for {today} (pipeline may not have run yet)"
        try:
            critical_lines: list[str] = []
            with open(log_file, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "CRITICAL" in line:
                        critical_lines.append(line.strip()[:120])
            if not critical_lines:
                return True, f"No CRITICAL errors in {log_file.name}"
            sample = critical_lines[0]
            return False, f"{len(critical_lines)} CRITICAL error(s) — first: {sample}"
        except Exception as exc:
            return False, f"Log scan failed: {exc}"

    def check_telegram_bot(self) -> tuple[bool, str]:
        """
        Verify the Telegram Bot API endpoint is reachable and token is valid.
        """
        import urllib.request
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not bot_token or "your_" in bot_token:
            return False, "TELEGRAM_BOT_TOKEN not configured"
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read())
            if data.get("ok"):
                username = data.get("result", {}).get("username", "unknown")
                return True, f"Telegram bot responsive: @{username}"
            return False, "Telegram getMe returned ok=false"
        except Exception as exc:
            return False, f"Telegram bot unreachable: {exc}"

    def check_api_keys_valid(self) -> tuple[bool, str]:
        """
        Verify ANTHROPIC and GROQ API keys are present and non-placeholder.
        A lightweight syntax check only — does not make a billable API call.
        """
        issues: list[str] = []
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        groq_key = os.environ.get("GROQ_API_KEY", "")

        if not anthropic_key or "your_" in anthropic_key or not anthropic_key.startswith("sk-"):
            issues.append("ANTHROPIC_API_KEY missing or invalid")
        if not groq_key or "your_" in groq_key or not groq_key.startswith("gsk_"):
            issues.append("GROQ_API_KEY missing or invalid")

        if not issues:
            return True, "ANTHROPIC and GROQ API keys present and well-formed"
        return False, "; ".join(issues)

    def check_zep_memory(self) -> tuple[bool, str]:
        """
        Confirm the Zep memory API is accessible.
        """
        import urllib.request
        zep_url = os.environ.get("ZEP_API_URL", "http://localhost:8000")
        zep_key = os.environ.get("ZEP_API_KEY", "")
        health_url = zep_url.rstrip("/") + "/healthz"
        try:
            req = urllib.request.Request(
                health_url,
                headers={"Authorization": f"Api-Key {zep_key}"} if zep_key else {},
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status < 400:
                    return True, f"Zep API accessible at {zep_url}"
            return False, f"Zep API returned status {resp.status}"
        except Exception as exc:
            return False, f"Zep API unreachable ({zep_url}): {exc}"

    # ── Aggregate check runner ──────────────────────────────────────────────

    def run_monitoring_checks(self) -> dict:
        """
        Execute all MONITORING_CHECKS and return a consolidated results dict.

        Returns
        -------
        dict
            Keys: timestamp, session_id, all_ok, checks (list of check results).
        """
        check_map = {
            "database_connection": self.check_database_connection,
            "nepse_data_freshness": self.check_data_freshness,
            "mirofish_simulation": self.check_simulation_ran,
            "signal_generated": self.check_signal_generated,
            "portfolio_snapshot": self.check_portfolio_snapshot,
            "telegram_bot": self.check_telegram_bot,
            "disk_space": self.check_disk_space,
            "api_keys_valid": self.check_api_keys_valid,
            "log_errors": self.check_log_errors,
            "zep_memory": self.check_zep_memory,
        }

        results: list[dict] = []
        all_ok = True
        for check_name, description in MONITORING_CHECKS.items():
            fn = check_map.get(check_name)
            if fn is None:
                continue
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, f"Unexpected error: {exc}"
            if not ok:
                all_ok = False
            results.append({
                "check": check_name,
                "description": description,
                "ok": ok,
                "detail": detail,
            })
            status_label = "OK  " if ok else "FAIL"
            logger.info("  [%s] %s — %s", status_label, check_name, detail)

        return {
            "timestamp": _nst_now().isoformat(),
            "session_id": self.session_id,
            "all_ok": all_ok,
            "pass_count": sum(1 for r in results if r["ok"]),
            "fail_count": sum(1 for r in results if not r["ok"]),
            "checks": results,
        }

    # ── Scheduled summary senders ───────────────────────────────────────────

    def send_morning_health_check(self) -> None:
        """
        08:00 NST — run all health checks and send a pre-market Telegram summary.
        """
        logger.info("=== Morning Health Check (08:00 NST) ===")
        report = self.run_monitoring_checks()
        today = _today_str()

        ok_count = report["pass_count"]
        fail_count = report["fail_count"]
        status_icon = "✅" if report["all_ok"] else "⚠️"

        lines = [
            f"{status_icon} *MiroFish Morning Check — {today}*",
            f"Session: `{self.session_id}`",
            f"Checks: {ok_count} OK / {fail_count} FAIL",
            "",
        ]
        for r in report["checks"]:
            icon = "✅" if r["ok"] else "❌"
            lines.append(f"{icon} {r['description']}")
            if not r["ok"]:
                lines.append(f"   ↳ {r['detail']}")

        lines += [
            "",
            "Market opens at 11:00 NST.",
            f"_Generated {_nst_now().strftime('%H:%M NST')}_",
        ]
        self._telegram_send("\n".join(lines))

    def send_post_market_summary(self) -> None:
        """
        16:30 NST — send post-market P&L summary, open positions, and today's signal.
        """
        logger.info("=== Post-Market Summary (16:30 NST) ===")
        today = _today_str()

        # Load snapshot
        snapshot = self._load_snapshot(today)
        pv = snapshot.get("portfolio_value", 0)
        ret = snapshot.get("return_pct", 0.0)
        regime = snapshot.get("regime", "UNKNOWN")
        score = snapshot.get("mirofish_score", 0.0)
        cash = snapshot.get("cash", 0)
        positions: dict = snapshot.get("positions", {})

        ret_icon = "📈" if ret >= 0 else "📉"

        lines = [
            f"📊 *MiroFish Post-Market — {today}*",
            f"Session: `{self.session_id}`",
            "",
            f"{ret_icon} Portfolio: NPR {pv:,.0f} ({ret:+.2f}%)",
            f"💵 Cash: NPR {cash:,.0f}",
            f"🏷 Regime: `{regime}` | MiroFish score: `{score:+.2f}`",
            "",
        ]
        if positions:
            lines.append(f"📋 *Open Positions ({len(positions)}):*")
            for sym, pos in list(positions.items())[:5]:
                entry = pos.get("entry_price", 0)
                qty = pos.get("qty", 0)
                lines.append(f"  • {sym}: {qty} shares @ NPR {entry:,.2f}")
        else:
            lines.append("📋 No open positions.")

        lines += [
            "",
            f"_Market closed. Next session: Sunday–Thursday 11:00 NST._",
            f"_Generated {_nst_now().strftime('%H:%M NST')}_",
        ]
        self._telegram_send("\n".join(lines))

    def send_evening_system_check(self) -> None:
        """
        19:00 NST — light system check (disk, logs, API keys).
        """
        logger.info("=== Evening System Check (19:00 NST) ===")
        today = _today_str()

        # Run only lightweight checks
        checks_to_run = [
            ("disk_space", self.check_disk_space),
            ("log_errors", self.check_log_errors),
            ("api_keys_valid", self.check_api_keys_valid),
            ("zep_memory", self.check_zep_memory),
        ]
        results: list[dict] = []
        all_ok = True
        for name, fn in checks_to_run:
            try:
                ok, detail = fn()
            except Exception as exc:
                ok, detail = False, str(exc)
            if not ok:
                all_ok = False
            results.append({"check": name, "ok": ok, "detail": detail})

        status_icon = "🌙" if all_ok else "⚠️"
        lines = [
            f"{status_icon} *MiroFish Evening Check — {today}*",
            "",
        ]
        for r in results:
            icon = "✅" if r["ok"] else "❌"
            lines.append(f"{icon} {r['check']}: {r['detail']}")

        lines += [
            "",
            f"_Generated {_nst_now().strftime('%H:%M NST')}_",
        ]
        self._telegram_send("\n".join(lines))

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _load_snapshot(self, today: str) -> dict:
        """Load today's portfolio snapshot from disk."""
        snap_file = self.session_dir / f"snapshot_{today}.json"
        if snap_file.exists():
            try:
                with open(snap_file, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        # Fallback: any session
        candidates = sorted(
            (self.data_dir).glob(f"*/snapshot_{today}.json"), reverse=True
        )
        if candidates:
            try:
                with open(candidates[0], encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _telegram_send(self, message: str) -> bool:
        """Send a Telegram message."""
        import urllib.request
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not bot_token or not chat_id or "your_" in bot_token:
            logger.debug("Telegram not configured — message logged only.")
            logger.debug("MSG: %s", message[:200])
            return False
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": message[:4096],
            "parse_mode": "Markdown",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# APScheduler integration
# ---------------------------------------------------------------------------

def schedule_monitoring(session_id: str) -> None:
    """
    Start APScheduler with three monitoring cron triggers.

    Schedule (UTC, Sun–Thu):
      02:15 UTC = 08:00 NST — morning health check
      10:45 UTC = 16:30 NST — post-market summary
      13:15 UTC = 19:00 NST — evening system check
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("apscheduler not installed. Run: pip install apscheduler")
        return

    monitor = ProductionMonitor(session_id=session_id)

    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        monitor.send_morning_health_check,
        CronTrigger(hour=2, minute=15, day_of_week="sun,mon,tue,wed,thu"),
        id="morning_health_check",
        name="Morning health check (08:00 NST)",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        monitor.send_post_market_summary,
        CronTrigger(hour=10, minute=45, day_of_week="sun,mon,tue,wed,thu"),
        id="post_market_summary",
        name="Post-market summary (16:30 NST)",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        monitor.send_evening_system_check,
        CronTrigger(hour=13, minute=15, day_of_week="sun,mon,tue,wed,thu"),
        id="evening_system_check",
        name="Evening system check (19:00 NST)",
        misfire_grace_time=300,
    )

    logger.info(
        "Monitoring scheduler started. Jobs: %s",
        [j.name for j in scheduler.get_jobs()],
    )
    print("\n  MiroFish production monitor running.")
    print("  08:00 NST (02:15 UTC) — morning health check")
    print("  16:30 NST (10:45 UTC) — post-market summary")
    print("  19:00 NST (13:15 UTC) — evening system check")
    print("  Press Ctrl+C to stop.\n")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Monitoring scheduler stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="NEPSE MiroFish production monitor")
    parser.add_argument("--session-id", default="default",
                        help="Paper/live trading session ID (default: 'default')")
    parser.add_argument("--run-now", action="store_true",
                        help="Run morning health check immediately and exit")
    parser.add_argument(
        "--check",
        choices=["morning", "post-market", "evening"],
        help="Run a specific monitoring check immediately and exit",
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

    monitor = ProductionMonitor(session_id=args.session_id)

    if args.run_now or args.check == "morning":
        monitor.send_morning_health_check()
    elif args.check == "post-market":
        monitor.send_post_market_summary()
    elif args.check == "evening":
        monitor.send_evening_system_check()
    else:
        schedule_monitoring(session_id=args.session_id)


if __name__ == "__main__":
    main()
