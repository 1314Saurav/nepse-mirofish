"""
db/loader.py

Database loader functions for nepse-mirofish.

All functions accept Python dicts / dataclasses (serialized via asdict)
and upsert / bulk-insert into PostgreSQL via SQLAlchemy.

Usage
-----
    from db.loader import upsert_market_snapshot, bulk_insert_stocks, ...
    engine = get_engine()   # reads NEPSE_DB_URL from .env
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.models import (
    Base,
    IpoEvent,
    MarketSnapshot,
    MirofishSignal,
    NewsArticle,
    NrbRate,
    SimulationSeed,
    StockPrice,
    get_engine,
)

# ── Session helper ─────────────────────────────────────────────────────────────

def _session(db_url: Optional[str] = None) -> Session:
    engine = get_engine(db_url)
    return Session(engine)


# ── 1. upsert_market_snapshot ──────────────────────────────────────────────────

def upsert_market_snapshot(snapshot: dict, db_url: Optional[str] = None) -> None:
    """
    Insert or update a MarketSnapshot row keyed by date.

    `snapshot` is the dict returned by scrape_market_snapshot().
    """
    date_s = snapshot.get("as_of_date", datetime.now().strftime("%Y-%m-%d"))
    row = {
        "date":            date_s,
        "nepse_index":     snapshot.get("nepse_index", {}).get("close"),
        "sensitive_index": snapshot.get("sensitive_index", {}).get("close"),
        "float_index":     snapshot.get("float_index", {}).get("close"),
        "banking_index":   snapshot.get("banking_subindex", {}).get("close"),
        "turnover_npr":    snapshot.get("market_overview", {}).get("total_turnover_npr"),
        "traded_shares":   snapshot.get("market_overview", {}).get("total_traded_shares"),
        "total_transactions": snapshot.get("market_overview", {}).get("total_transactions"),
        "scrips_traded":   snapshot.get("market_overview", {}).get("total_scrips_traded"),
        "scraped_at":      datetime.utcnow(),
    }
    stmt = (
        pg_insert(MarketSnapshot)
        .values(**row)
        .on_conflict_do_update(
            index_elements=["date"],
            set_={k: v for k, v in row.items() if k != "date"},
        )
    )
    with _session(db_url) as sess:
        sess.execute(stmt)
        sess.commit()
    print(f"  [DB] market_snapshots upserted for {date_s}", flush=True)


# ── 2. bulk_insert_stocks ──────────────────────────────────────────────────────

def bulk_insert_stocks(stocks: list, db_url: Optional[str] = None) -> int:
    """
    Batch-insert StockDetail dicts / dataclass instances.
    Skips rows where (symbol, date) already exists.

    Returns number of rows actually inserted.
    """
    from dataclasses import asdict
    today = datetime.now().strftime("%Y-%m-%d")

    rows = []
    for s in stocks:
        d = asdict(s) if hasattr(s, "__dataclass_fields__") else s
        rows.append({
            "date":       d.get("scraped_at", today)[:10],
            "symbol":     d.get("symbol", ""),
            "ltp":        d.get("ltp"),
            "open":       d.get("open"),
            "high":       d.get("high"),
            "low":        d.get("low"),
            "close":      d.get("ltp"),       # use ltp as close
            "volume":     None,
            "pe_ratio":   d.get("pe_ratio"),
            "eps":        d.get("eps"),
            "market_cap": d.get("market_cap"),
            "scraped_at": datetime.utcnow(),
        })

    if not rows:
        return 0

    stmt = (
        pg_insert(StockPrice)
        .values(rows)
        .on_conflict_do_nothing(constraint="uq_stock_prices_symbol_date")
    )
    with _session(db_url) as sess:
        result = sess.execute(stmt)
        sess.commit()
    inserted = result.rowcount if result.rowcount >= 0 else len(rows)
    print(f"  [DB] stock_prices: {inserted} rows inserted (of {len(rows)})", flush=True)
    return inserted


# ── 3. insert_news_batch ───────────────────────────────────────────────────────

def insert_news_batch(articles: list[dict], db_url: Optional[str] = None) -> int:
    """
    Insert news articles, deduplicating by URL.

    Returns number of new rows inserted.
    """
    if not articles:
        return 0

    rows = []
    seen_urls: set[str] = set()
    for a in articles:
        url = a.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        pub = a.get("published_at", "")
        if isinstance(pub, str) and pub:
            try:
                pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except ValueError:
                pub_dt = datetime.utcnow()
        elif isinstance(pub, datetime):
            pub_dt = pub
        else:
            pub_dt = datetime.utcnow()

        rows.append({
            "published_at":    pub_dt,
            "title":           (a.get("title") or "")[:500],
            "url":             url,
            "source":          (a.get("source_name") or a.get("source") or "")[:100],
            "category":        (a.get("category") or "general_market")[:50],
            "body_excerpt":    (a.get("body") or a.get("body_excerpt") or "")[:2000],
            "sentiment_score": a.get("sentiment_score"),
            "inserted_at":     datetime.utcnow(),
        })

    if not rows:
        return 0

    stmt = (
        pg_insert(NewsArticle)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["url"])
    )
    with _session(db_url) as sess:
        result = sess.execute(stmt)
        sess.commit()
    inserted = result.rowcount if result.rowcount >= 0 else len(rows)
    print(f"  [DB] news_articles: {inserted} rows inserted", flush=True)
    return inserted


# ── 4. upsert_nrb_rates ────────────────────────────────────────────────────────

def upsert_nrb_rates(rates: dict, db_url: Optional[str] = None) -> None:
    """
    Upsert NRB rates keyed by date.

    `rates` is the dict returned by scrape_nrb_policy().
    """
    date_s = rates.get("as_of_date", datetime.now().strftime("%Y-%m-%d"))
    ir     = rates.get("interest_rates", {})
    rr     = rates.get("reserve_requirements", {})
    fx     = rates.get("forex", {}).get("current", {})
    cd     = rates.get("credit_deposit_ratio", {})

    row = {
        "date":                date_s,
        "bank_rate":           ir.get("bank_rate"),
        "repo_rate":           ir.get("repo_rate"),
        "reverse_repo":        ir.get("reverse_repo"),
        "crr":                 rr.get("crr_pct"),
        "slr":                 rr.get("slr_pct"),
        "usd_npr_buy":         fx.get("USD", {}).get("buy"),
        "usd_npr_sell":        fx.get("USD", {}).get("sell"),
        "eur_npr_buy":         fx.get("EUR", {}).get("buy"),
        "eur_npr_sell":        fx.get("EUR", {}).get("sell"),
        "inr_npr_buy":         fx.get("INR", {}).get("buy"),
        "inr_npr_sell":        fx.get("INR", {}).get("sell"),
        "cny_npr_buy":         fx.get("CNY", {}).get("buy"),
        "cny_npr_sell":        fx.get("CNY", {}).get("sell"),
        "credit_deposit_ratio": cd.get("ratio"),
        "scraped_at":          datetime.utcnow(),
    }
    stmt = (
        pg_insert(NrbRate)
        .values(**row)
        .on_conflict_do_update(
            index_elements=["date"],
            set_={k: v for k, v in row.items() if k != "date"},
        )
    )
    with _session(db_url) as sess:
        sess.execute(stmt)
        sess.commit()
    print(f"  [DB] nrb_rates upserted for {date_s}", flush=True)


# ── 5. save_seed ───────────────────────────────────────────────────────────────

def save_seed(seed: dict, db_url: Optional[str] = None) -> None:
    """Save a daily seed dict to simulation_seeds table."""
    date_s = seed.get("date", datetime.now().strftime("%Y-%m-%d"))
    row = {
        "date":                date_s,
        "seed_json":           seed,
        "brief_text":          seed.get("brief_text"),
        "simulation_question": seed.get("simulation_question"),
        "created_at":          datetime.utcnow(),
    }
    stmt = (
        pg_insert(SimulationSeed)
        .values(**row)
        .on_conflict_do_update(
            index_elements=["date"],
            set_={k: v for k, v in row.items() if k != "date"},
        )
    )
    with _session(db_url) as sess:
        sess.execute(stmt)
        sess.commit()
    print(f"  [DB] simulation_seeds upserted for {date_s}", flush=True)


# ── 6. get_seeds_for_backtest ──────────────────────────────────────────────────

def get_seeds_for_backtest(
    start_date: str,
    end_date: str,
    db_url: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve simulation seeds for a date range, ordered by date ascending.

    Parameters
    ----------
    start_date, end_date : "YYYY-MM-DD"

    Returns list of seed dicts (the seed_json JSONB column).
    """
    with _session(db_url) as sess:
        rows = (
            sess.query(SimulationSeed)
            .filter(SimulationSeed.date >= start_date)
            .filter(SimulationSeed.date <= end_date)
            .order_by(SimulationSeed.date.asc())
            .all()
        )
    return [r.seed_json for r in rows]


# ── backfill_history ───────────────────────────────────────────────────────────

_SS_PRIOR_PRICE_URL = "https://www.sharesansar.com/prior-price"
_SS_BASE            = "https://www.sharesansar.com"


def _ss_session_warm() -> tuple:
    """Return (requests.Session, xsrf_token) warmed on sharesansar."""
    import requests
    from urllib.parse import unquote

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(_SS_BASE, timeout=20)
    xsrf = unquote(s.cookies.get("XSRF-TOKEN", ""))
    return s, xsrf


def _fetch_historical_prices(
    session,
    symbol: str,
    from_date: str,
    to_date: str,
    xsrf: str,
) -> list[dict]:
    """
    Fetch historical daily prices for `symbol` from sharesansar DataTables endpoint.
    Returns list of raw row dicts.
    """
    # Warm up on the prior-price page with the symbol selected
    session.get(
        f"{_SS_BASE}/prior-price",
        params={"company": symbol},
        timeout=20,
    )
    rows = []
    start = 0
    length = 500
    while True:
        try:
            r = session.get(
                _SS_PRIOR_PRICE_URL,
                params={
                    "company":  symbol,
                    "from":     from_date,
                    "to":       to_date,
                    "draw":     1,
                    "start":    start,
                    "length":   length,
                },
                headers={
                    "Accept":           "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                    "X-XSRF-TOKEN":     xsrf,
                    "Referer":          f"{_SS_BASE}/prior-price",
                },
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            print(f"  [WARN] backfill {symbol} {from_date}: {exc}", file=sys.stderr)
            break

        data = payload.get("data", [])
        rows.extend(data)

        if len(data) < length:
            break
        start += length
        time.sleep(0.5)

    return rows


def backfill_history(
    start_date: str = "2022-01-01",
    end_date: Optional[str] = None,
    symbols: Optional[list[str]] = None,
    db_url: Optional[str] = None,
) -> None:
    """
    Scrape sharesansar.com historical price data and populate stock_prices.

    Parameters
    ----------
    start_date : "YYYY-MM-DD"  (default "2022-01-01")
    end_date   : "YYYY-MM-DD"  (default: today)
    symbols    : list of stock symbols (default: fetch top symbols from DB)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # If no symbols given, try to pull from existing stock_prices
    if not symbols:
        try:
            with _session(db_url) as sess:
                rows = sess.execute(
                    text("SELECT DISTINCT symbol FROM stock_prices LIMIT 100")
                ).fetchall()
                symbols = [r[0] for r in rows]
        except Exception:
            pass
    if not symbols:
        print("  [WARN] No symbols for backfill — pass symbols explicitly.", file=sys.stderr)
        return

    session, xsrf = _ss_session_warm()

    total_inserted = 0
    for sym in symbols:
        print(f"  [BACKFILL] {sym} {start_date} -> {end_date}...", flush=True)
        raw_rows = _fetch_historical_prices(session, sym, start_date, end_date, xsrf)

        db_rows = []
        for row in raw_rows:
            # sharesansar prior-price columns: s (date), ltp, open, high, low, qty, turnover
            d_str = row.get("published_date") or row.get("s", "")
            if len(d_str) >= 10:
                d_str = d_str[:10]
            else:
                continue

            def _n(v):
                try:
                    return float(str(v).replace(",", "").strip())
                except Exception:
                    return None

            db_rows.append({
                "date":       d_str,
                "symbol":     sym,
                "ltp":        _n(row.get("close") or row.get("ltp")),
                "open":       _n(row.get("open")),
                "high":       _n(row.get("high")),
                "low":        _n(row.get("low")),
                "close":      _n(row.get("close") or row.get("ltp")),
                "volume":     _n(row.get("quantity") or row.get("qty")),
                "pe_ratio":   None,
                "eps":        None,
                "market_cap": None,
                "scraped_at": datetime.utcnow(),
            })

        if not db_rows:
            print(f"  [BACKFILL] {sym}: 0 rows", flush=True)
            continue

        stmt = (
            pg_insert(StockPrice)
            .values(db_rows)
            .on_conflict_do_nothing(constraint="uq_stock_prices_symbol_date")
        )
        with _session(db_url) as sess:
            result = sess.execute(stmt)
            sess.commit()
        n = result.rowcount if result.rowcount >= 0 else len(db_rows)
        total_inserted += n
        print(f"  [BACKFILL] {sym}: {n} rows inserted", flush=True)
        time.sleep(1.0)

    print(f"  [BACKFILL] Done. Total rows inserted: {total_inserted}", flush=True)
