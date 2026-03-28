"""
tests/test_pipeline_e2e.py

Phase 1 pipeline end-to-end health check.

Runs the full daily pipeline once (equivalent to --run-now), then:
  1. Verifies DB was populated (row counts in key tables)
  2. Verifies data/seed/ has today's seed file
  3. Validates seed JSON schema
  4. Prints a summary health-check table

Usage
-----
  pytest tests/test_pipeline_e2e.py -v
  python tests/test_pipeline_e2e.py      # standalone (no pytest needed)
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# ── Bootstrap ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

TODAY = datetime.now().strftime("%Y-%m-%d")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _db_available() -> bool:
    db_url = os.getenv("NEPSE_DB_URL", "")
    if not db_url or db_url.startswith("postgresql://user:password"):
        return False
    try:
        from db.models import get_engine
        engine = get_engine(db_url)
        with engine.connect():
            pass
        return True
    except Exception:
        return False


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pipeline_result():
    """Run the pipeline once per test session and cache result."""
    from scheduler.daily_pipeline import run_daily_pipeline
    run_daily_pipeline()
    return True


# ── Individual tests ───────────────────────────────────────────────────────────

class TestSeedFile:
    def test_seed_file_exists(self, pipeline_result):
        seed_path = _ROOT / "data" / "seed" / f"seed_{TODAY}.json"
        assert seed_path.exists(), f"Seed file not found: {seed_path}"

    def test_seed_required_keys(self, pipeline_result):
        seed_path = _ROOT / "data" / "seed" / f"seed_{TODAY}.json"
        if not seed_path.exists():
            pytest.skip("Seed file missing")
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        required = [
            "date", "market_summary", "macro_context",
            "news_articles", "ipo_events", "simulation_question",
        ]
        for key in required:
            assert key in seed, f"Missing key in seed: {key}"

    def test_seed_date_matches_today(self, pipeline_result):
        seed_path = _ROOT / "data" / "seed" / f"seed_{TODAY}.json"
        if not seed_path.exists():
            pytest.skip("Seed file missing")
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        assert seed["date"] == TODAY

    def test_market_summary_not_empty(self, pipeline_result):
        seed_path = _ROOT / "data" / "seed" / f"seed_{TODAY}.json"
        if not seed_path.exists():
            pytest.skip("Seed file missing")
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        ms = seed.get("market_summary", {})
        assert ms, "market_summary is empty"

    def test_news_articles_list(self, pipeline_result):
        seed_path = _ROOT / "data" / "seed" / f"seed_{TODAY}.json"
        if not seed_path.exists():
            pytest.skip("Seed file missing")
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        articles = seed.get("news_articles", [])
        assert isinstance(articles, list)


class TestDataFiles:
    def test_market_raw_exists(self, pipeline_result):
        raw_dir = _ROOT / "data" / "raw"
        market_files = list(raw_dir.glob("*market*"))
        assert market_files, "No market raw file found"

    def test_stocks_processed_exists(self, pipeline_result):
        candidates = sorted((_ROOT / "data" / "processed").glob("stocks_*.json"), reverse=True)
        assert candidates, "No stocks processed file found"
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        assert isinstance(data, list), "stocks JSON should be a list"
        assert len(data) > 0, "stocks list is empty"

    def test_news_processed_exists(self, pipeline_result):
        candidates = sorted((_ROOT / "data" / "processed").glob("news_*.json"), reverse=True)
        assert candidates, "No news processed file found"

    def test_ipo_calendar_exists(self, pipeline_result):
        ipo_path = _ROOT / "data" / "processed" / "ipo_calendar.json"
        assert ipo_path.exists(), "ipo_calendar.json not found"


class TestDatabasePopulation:
    @pytest.mark.skipif(not _db_available(), reason="DB not configured")
    def test_market_snapshots_populated(self, pipeline_result):
        from db.models import get_engine, MarketSnapshot
        from sqlalchemy.orm import Session
        with Session(get_engine()) as sess:
            count = sess.query(MarketSnapshot).count()
        assert count > 0, "market_snapshots table is empty"

    @pytest.mark.skipif(not _db_available(), reason="DB not configured")
    def test_stock_prices_populated(self, pipeline_result):
        from db.models import get_engine, StockPrice
        from sqlalchemy.orm import Session
        with Session(get_engine()) as sess:
            count = sess.query(StockPrice).count()
        assert count > 0, "stock_prices table is empty"

    @pytest.mark.skipif(not _db_available(), reason="DB not configured")
    def test_news_articles_populated(self, pipeline_result):
        from db.models import get_engine, NewsArticle
        from sqlalchemy.orm import Session
        with Session(get_engine()) as sess:
            count = sess.query(NewsArticle).count()
        assert count > 0, "news_articles table is empty"

    @pytest.mark.skipif(not _db_available(), reason="DB not configured")
    def test_simulation_seeds_populated(self, pipeline_result):
        from db.models import get_engine, SimulationSeed
        from sqlalchemy.orm import Session
        with Session(get_engine()) as sess:
            count = sess.query(SimulationSeed).count()
        assert count > 0, "simulation_seeds table is empty"


# ── Standalone health-check runner (no pytest) ────────────────────────────────

CHECKS = [
    # (label, fn) — fn returns (passed: bool, detail: str)
]


def _check(label: str):
    """Decorator to register a health check."""
    def decorator(fn):
        CHECKS.append((label, fn))
        return fn
    return decorator


@_check("Pipeline ran without exception")
def _chk_pipeline_ran():
    try:
        from scheduler.daily_pipeline import run_daily_pipeline
        run_daily_pipeline()
        return True, "OK"
    except Exception as exc:
        return False, str(exc)


@_check("Seed file created")
def _chk_seed_file():
    p = _ROOT / "data" / "seed" / f"seed_{TODAY}.json"
    return p.exists(), str(p)


@_check("Seed has required keys")
def _chk_seed_keys():
    p = _ROOT / "data" / "seed" / f"seed_{TODAY}.json"
    if not p.exists():
        return False, "Seed file missing"
    seed = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in ["date","market_summary","macro_context","news_articles"] if k not in seed]
    return len(missing) == 0, f"Missing: {missing}" if missing else "OK"


@_check("Stocks data file present")
def _chk_stocks():
    candidates = sorted((_ROOT / "data" / "processed").glob("stocks_*.json"), reverse=True)
    if not candidates:
        return False, "No stocks file"
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    count = len(data) if isinstance(data, list) else 0
    return count > 0, f"{count} stocks"


@_check("News data file present")
def _chk_news():
    candidates = sorted((_ROOT / "data" / "processed").glob("news_*.json"), reverse=True)
    if not candidates:
        return False, "No news file"
    return True, candidates[0].name


@_check("NRB policy file present")
def _chk_nrb():
    candidates = sorted((_ROOT / "data" / "processed").glob("nrb_policy_*.json"), reverse=True)
    if not candidates:
        return False, "No NRB file"
    return True, candidates[0].name


@_check("IPO calendar present")
def _chk_ipo():
    p = _ROOT / "data" / "processed" / "ipo_calendar.json"
    return p.exists(), "OK" if p.exists() else "Missing"


@_check("Database reachable")
def _chk_db():
    return _db_available(), "Connected" if _db_available() else "Not configured / unreachable"


def run_health_check() -> bool:
    """Run all checks and print a summary table. Returns True if all passed."""
    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"

    print(f"\n{BOLD}Phase 1 Pipeline Health Check — {TODAY}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"  {'Check':<40} {'Status':>8}  Detail")
    print(f"  {'-'*40} {'-'*8}  {'-'*20}")

    all_pass = True
    for label, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as exc:
            passed, detail = False, str(exc)

        status_txt = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {label:<40} {status_txt}  {detail}")
        if not passed:
            all_pass = False

    print(f"{CYAN}{'='*60}{RESET}")
    overall = f"{GREEN}ALL PASS{RESET}" if all_pass else f"{RED}SOME FAILURES{RESET}"
    print(f"  Overall: {overall}\n")
    return all_pass


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ok = run_health_check()
    sys.exit(0 if ok else 1)
