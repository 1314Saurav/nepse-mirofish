"""
scheduler/daily_pipeline.py

APScheduler daily pipeline that runs at 15:30 Nepal Standard Time (UTC+5:45)
on weekdays (Mon-Fri).

Steps
-----
  1. scrape_market_snapshot()        → DB
  2. scrape_all_stocks(TOP_50)       → DB
  3. collect_news(last_24h)          → DB
  4. scrape_nrb_policy()             → DB
  5. scrape_ipo_calendar()           → DB
  6. build_daily_seed()              → DB + data/seed/
  7. run MiroFish simulation         → data/processed/simulations/
  8. extract_trading_signal()        → DB + data/processed/signals/
  9. Write enriched notification     (signal + market brief → data/notifications/alerts.json)

Usage
-----
  python scheduler/daily_pipeline.py            # starts scheduler
  python scheduler/daily_pipeline.py --run-now  # run once immediately
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = _ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _get_logger() -> logging.Logger:
    today  = datetime.now().strftime("%Y-%m-%d")
    logger = logging.getLogger("nepse_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # File handler (daily log)
    fh = logging.FileHandler(LOG_DIR / f"pipeline_{today}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ── Stock symbol list ──────────────────────────────────────────────────────────

TOP_50_SYMBOLS = [
    # Commercial Banks
    "NABIL", "NBBL", "NIC", "EBL", "SBI", "PCBL", "KBL", "NIMB",
    "MEGA", "GBIME", "ADBL", "CZBIL", "HBL", "MBL", "PRVU", "SANIMA",
    # Development Banks
    "MNBBL", "SHINE", "MLBBL",
    # Finance
    "GFCL", "SIFC", "CFCL",
    # Hydropower
    "NHPC", "UPPER", "BPCL", "CHCL", "API", "BARUN", "GHL", "KKHC",
    "RURU", "USHEC",
    # Insurance
    "NLIC", "NLICL", "LICN", "PRIDE", "SGIC",
    # Manufacturing
    "NIFRA", "SONA", "UNL",
    # Microfinance
    "SWBBL", "NESDO",
    # Mutual Fund
    "NIBLPF", "NMBSF1",
]


# ── Notification writer ────────────────────────────────────────────────────────

def _write_notification(msg: str) -> None:
    """Write alert to notifications file for dashboard to read."""
    import json, datetime
    path = _ROOT / "data" / "notifications" / "alerts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.datetime.now().isoformat(), "msg": msg}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def _step(name: str, fn, logger: logging.Logger):
    """Run a pipeline step; log errors but don't abort the pipeline."""
    logger.info(f"--- Step: {name} ---")
    try:
        result = fn()
        logger.info(f"    {name}: OK")
        return result
    except Exception as exc:
        logger.error(f"    {name}: FAILED — {exc}", exc_info=True)
        return None


def run_daily_pipeline() -> None:
    logger = _get_logger()
    today  = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"========== NEPSE Daily Pipeline: {today} ==========")

    # 1. Market snapshot
    from scrapers.nepse_market import scrape_market_snapshot
    snapshot = _step("market_snapshot", scrape_market_snapshot, logger)

    # 2. Top stocks
    from scrapers.nepse_stocks import scrape_all_stocks
    stocks = _step(
        "top_stocks",
        lambda: scrape_all_stocks(TOP_50_SYMBOLS),
        logger,
    )

    # 3. News
    from scrapers.news_collector import collect_news
    news_result = _step("collect_news", lambda: collect_news(hours=24), logger)

    # 4. NRB policy
    from scrapers.nrb_policy import scrape_nrb_policy
    nrb_data = _step("nrb_policy", scrape_nrb_policy, logger)

    # 5. IPO calendar
    from scrapers.nepse_ipo import scrape_ipo_calendar
    ipo_result = _step("ipo_calendar", scrape_ipo_calendar, logger)

    # 6. Build seed
    from pipeline.seed_builder import build_daily_seed
    seed_result = _step("build_seed", lambda: build_daily_seed(today), logger)

    # DB saves (only if DB is configured)
    db_url = os.getenv("NEPSE_DB_URL", "")
    if db_url and not db_url.startswith("postgresql://user:password"):
        logger.info("--- Saving to database ---")
        from db.loader import (
            upsert_market_snapshot, bulk_insert_stocks,
            insert_news_batch, upsert_nrb_rates, save_seed,
        )
        if snapshot:
            _step("db_market", lambda: upsert_market_snapshot(snapshot), logger)
        if stocks:
            _step("db_stocks", lambda: bulk_insert_stocks(stocks), logger)
        if news_result:
            news_dict, _ = news_result
            articles = news_dict.get("articles", [])
            _step("db_news", lambda: insert_news_batch(articles), logger)
        if nrb_data:
            _step("db_nrb", lambda: upsert_nrb_rates(nrb_data), logger)
        if seed_result:
            seed, _ = seed_result
            _step("db_seed", lambda: save_seed(seed), logger)
    else:
        logger.warning("NEPSE_DB_URL not configured — skipping DB writes.")

    # 7. MiroFish simulation (requires MiroFish backend running at localhost:5001)
    sim_result = None
    seed_path  = None
    if seed_result:
        from pathlib import Path as _Path
        _seed, _ = seed_result
        seed_path = _ROOT / "data" / "seed" / f"seed_{today}.json"

    if seed_path and seed_path.exists():
        try:
            from pipeline.mirofish_bridge import MiroFishBridge, MIROFISH_BASE_URL
            bridge = MiroFishBridge(MIROFISH_BASE_URL)
            if bridge.is_alive():
                logger.info("MiroFish backend detected — running simulation.")
                from pipeline.run_simulation import run_simulation
                sim_result = _step(
                    "mirofish_simulation",
                    lambda: run_simulation(seed_path, max_rounds=12, stream=False),
                    logger,
                )
            else:
                logger.warning("MiroFish backend not running at localhost:5001 — skipping simulation.")
        except ImportError as exc:
            logger.warning(f"MiroFish bridge import error: {exc} — skipping simulation.")
    else:
        logger.warning("No seed file found for today — skipping simulation.")

    # 8. Extract trading signal
    signal = None
    if sim_result:
        from pipeline.signal_extractor import extract_trading_signal, save_signal
        from pipeline.simulation_qa import run_qa_checks, print_qa_report

        signal = _step(
            "signal_extraction",
            lambda: extract_trading_signal(sim_result),
            logger,
        )
        if signal:
            # QA checks
            flags, qa_report = run_qa_checks(sim_result, signal)
            signal["quality_flags"] = flags
            if flags:
                for flag in flags:
                    logger.warning(f"    QA flag: {flag}")

            # Save signal JSON
            sig_path = _step("save_signal", lambda: save_signal(signal), logger)
            if sig_path:
                logger.info(f"    Signal saved: {sig_path}")

            # Save signal to DB
            db_url = os.getenv("NEPSE_DB_URL", "")
            if db_url and not db_url.startswith("postgresql://user:password"):
                from db.loader import save_mirofish_signal
                _step("db_signal", lambda: save_mirofish_signal(signal), logger)

    # 9. Enriched Telegram alert (signal + market brief)
    if seed_result:
        seed, brief = seed_result
        ms    = seed.get("market_summary", {})
        idx   = ms.get("nepse_index", "N/A")
        chg   = ms.get("nepse_pct_change", 0) or 0
        arrow = "+" if chg >= 0 else ""

        if signal:
            # Enriched alert with simulation signal
            direction = signal.get("direction", "N/A")
            bb_score  = signal.get("bull_bear_score", 0)
            conf      = signal.get("confidence_pct", 0)
            drivers   = ", ".join(signal.get("top_driver_agent_types", [])[:2])
            themes    = "; ".join(signal.get("key_themes", [])[:2])

            # Top 2 bullish sectors
            sec_sigs = signal.get("sector_signals", {})
            top_secs = sorted(sec_sigs.items(), key=lambda x: x[1], reverse=True)[:2]
            sec_str  = "  ".join(f"{s[0]} ({s[1]:+.2f})" for s in top_secs)

            sign_char = "+" if bb_score >= 0 else ""
            msg = (
                f"*NEPSE Daily Signal — {today}*\n"
                f"Index: *{idx}* ({arrow}{chg:.2f}%)\n"
                f"Signal: *{direction}* ({sign_char}{bb_score:.2f}, {conf:.0f}% confidence)\n"
                f"Top sectors: {sec_str}\n"
                f"Key themes: {themes}\n"
                f"Top drivers: {drivers}\n\n"
                f"_{brief[:300]}..._\n\n"
                f"_Informational only — not financial advice._"
            )
        else:
            # Simple alert without simulation
            msg = (
                f"*NEPSE Daily Brief — {today}*\n"
                f"Index: *{idx}* ({arrow}{chg:.2f}%)\n\n"
                f"{brief}"
            )

        _step("write_notification", lambda: _write_notification(msg), logger)

    logger.info("========== Pipeline complete ==========\n")


# ── Scheduler ──────────────────────────────────────────────────────────────────

def start_scheduler() -> None:
    """Start APScheduler; runs run_daily_pipeline at 15:30 NST (Mon-Fri)."""
    logger = _get_logger()
    scheduler = BlockingScheduler(timezone="Asia/Kathmandu")
    # 15:30 NST = 09:45 UTC (NST is UTC+5:45)
    # APScheduler CronTrigger with timezone handles DST-safe scheduling
    scheduler.add_job(
        run_daily_pipeline,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=15,
            minute=30,
            timezone="Asia/Kathmandu",
        ),
        id="daily_nepse_pipeline",
        name="NEPSE Daily Pipeline",
        misfire_grace_time=300,
    )
    logger.info("Scheduler started. Daily pipeline runs at 15:30 NST (Mon-Fri).")
    logger.info("Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    parser = argparse.ArgumentParser(description="NEPSE daily pipeline")
    parser.add_argument(
        "--run-now", action="store_true",
        help="Run the pipeline immediately instead of waiting for schedule",
    )
    args = parser.parse_args()

    if args.run_now:
        run_daily_pipeline()
    else:
        start_scheduler()
