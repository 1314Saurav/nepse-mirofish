"""
backtest/historical_seeds.py
Historical seed regeneration for NEPSE MiroFish backtesting.
Reconstructs MiroFish signals for 2022-2024 without re-running
live simulations (too slow/expensive for 3 years of data).
Uses Claude API as a proxy to generate historically-accurate signals.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Claude proxy prompt for synthetic signal generation
# ---------------------------------------------------------------------------

SYNTHETIC_SIGNAL_PROMPT = """\
You are simulating the output of a 1000-agent MiroFish market sentiment \
simulation for the Nepal Stock Exchange (NEPSE) on {date}.

Given the following market conditions and news from {date}:

MARKET SNAPSHOT:
{market_snapshot}

NEWS ARTICLES ({news_count} articles):
{news_summary}

NRB RATES:
{nrb_rates}

Produce a JSON signal representing how 1000 Nepali market agents would \
collectively react. Agent types:
- Institutional brokers (200): focus on technical levels, broker reports
- Retail momentum traders (200): follow price/volume trends, social sentiment
- NRB policy watchers (150): interpret monetary policy signals
- Hydropower analysts (150): sector fundamentals, electricity generation data
- Political risk analysts (150): coalition stability, policy uncertainty
- Diaspora investors (150): remittance flows, USD/NPR, global risk appetite

Base your assessment on how these agent types would HISTORICALLY have \
reacted on {date}, given Nepal's actual economic conditions at that time.

Respond ONLY with valid JSON in exactly this format:
{{
  "date": "{date}",
  "bull_bear_score": <float -1.0 to +1.0>,
  "confidence_pct": <float 0 to 100>,
  "action": "<BUY|HOLD|SELL>",
  "sector_signals": {{
    "banking": <float -1 to +1>,
    "hydropower": <float -1 to +1>,
    "insurance": <float -1 to +1>,
    "finance": <float -1 to +1>,
    "microfinance": <float -1 to +1>,
    "manufacturing": <float -1 to +1>
  }},
  "key_themes": ["<theme1>", "<theme2>", "<theme3>"],
  "top_driver_agent_types": ["<type1>", "<type2>"],
  "source": "synthetic_claude"
}}
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_engine():
    try:
        from db.models import get_engine
        return get_engine()
    except Exception as exc:
        logger.warning("DB unavailable: %s", exc)
        return None


def _get_market_snapshot(d: date) -> dict:
    engine = _get_engine()
    if not engine:
        return {}
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT nepse_index, total_turnover, total_transactions, "
                    "advancing, declining FROM market_snapshots WHERE date = :d LIMIT 1"
                ),
                {"d": d},
            ).fetchone()
            if row:
                return {
                    "nepse_index": float(row[0] or 0),
                    "total_turnover_npr": float(row[1] or 0),
                    "transactions": int(row[2] or 0),
                    "advancing": int(row[3] or 0),
                    "declining": int(row[4] or 0),
                }
    except Exception as exc:
        logger.debug("Market snapshot fetch error: %s", exc)
    return {}


def _get_news_articles(d: date, lookback_days: int = 3) -> list[dict]:
    engine = _get_engine()
    if not engine:
        return []
    try:
        from sqlalchemy import text
        start = d - timedelta(days=lookback_days)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT title, source, published_at, sentiment_score "
                    "FROM news_articles "
                    "WHERE published_at BETWEEN :start AND :end "
                    "ORDER BY published_at DESC LIMIT 20"
                ),
                {"start": start, "end": d},
            ).fetchall()
            return [
                {"title": r[0], "source": r[1],
                 "published": str(r[2]), "sentiment": float(r[3] or 0)}
                for r in rows
            ]
    except Exception as exc:
        logger.debug("News fetch error: %s", exc)
    return []


def _get_nrb_rates(d: date) -> dict:
    engine = _get_engine()
    if not engine:
        return {}
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT repo_rate, bank_rate, crr, slr "
                    "FROM nrb_rates "
                    "WHERE effective_date <= :d ORDER BY effective_date DESC LIMIT 1"
                ),
                {"d": d},
            ).fetchone()
            if row:
                return {
                    "repo_rate": float(row[0] or 0),
                    "bank_rate": float(row[1] or 0),
                    "crr": float(row[2] or 0),
                    "slr": float(row[3] or 0),
                }
    except Exception as exc:
        logger.debug("NRB rates fetch error: %s", exc)
    return {}


def _existing_signal(d: date) -> Optional[dict]:
    """Check if a saved signal already exists for this date."""
    engine = _get_engine()
    if not engine:
        return None
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT signal_data FROM simulation_results "
                    "WHERE date = :d ORDER BY created_at DESC LIMIT 1"
                ),
                {"d": d},
            ).fetchone()
            if row and row[0]:
                data = json.loads(row[0]) if isinstance(row[0], str) else dict(row[0])
                return data
    except Exception:
        pass
    return None


def _save_signal(d: date, signal: dict) -> None:
    engine = _get_engine()
    if not engine:
        # Save to file as fallback
        out = Path("data/processed/historical_seeds")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"signal_{d.isoformat()}.json"
        with open(path, "w") as fh:
            json.dump(signal, fh, indent=2)
        return
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO simulation_results (date, signal_data, source) "
                    "VALUES (:d, :data, 'synthetic') "
                    "ON CONFLICT (date) DO NOTHING"
                ),
                {"d": d, "data": json.dumps(signal)},
            )
    except Exception as exc:
        logger.warning("Failed to save signal for %s: %s", d, exc)


# ---------------------------------------------------------------------------
# Claude API signal generator
# ---------------------------------------------------------------------------

def _generate_synthetic_signal(d: date, market: dict, news: list[dict], nrb: dict) -> Optional[dict]:
    """Call Claude API to generate a historically-accurate synthetic signal."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or "your_" in api_key:
        return _rule_based_signal(d, market, news, nrb)

    market_str = json.dumps(market, indent=2) if market else "No data available"
    news_summary = "\n".join(
        f"- [{a.get('source','')}] {a.get('title','')}"
        for a in news[:10]
    ) or "No news articles available"
    nrb_str = (
        f"Repo rate: {nrb.get('repo_rate', 'N/A')}%  "
        f"Bank rate: {nrb.get('bank_rate', 'N/A')}%  "
        f"CRR: {nrb.get('crr', 'N/A')}%"
    ) if nrb else "No NRB rate data"

    prompt = SYNTHETIC_SIGNAL_PROMPT.format(
        date=d.isoformat(),
        market_snapshot=market_str,
        news_count=len(news),
        news_summary=news_summary,
        nrb_rates=nrb_str,
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5",   # faster/cheaper for bulk generation
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Extract JSON
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        signal = json.loads(raw)
        signal["source"] = "synthetic_claude"
        signal["date"] = d.isoformat()
        return signal
    except Exception as exc:
        logger.warning("Claude signal generation failed for %s: %s", d, exc)
        return _rule_based_signal(d, market, news, nrb)


def _rule_based_signal(d: date, market: dict, news: list[dict], nrb: dict) -> dict:
    """
    Fallback: deterministic rule-based signal when Claude is unavailable.
    Uses market breadth and news sentiment as proxies.
    """
    adv = market.get("advancing", 0)
    dec = market.get("declining", 0)
    total = adv + dec

    # Market breadth score
    if total > 0:
        breadth = (adv - dec) / total
    else:
        breadth = 0.0

    # News sentiment average
    sentiments = [float(a.get("sentiment", 0)) for a in news if "sentiment" in a]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    # NRB stance (higher rates = bearish)
    repo = float(nrb.get("repo_rate", 7.0))
    nrb_score = -0.5 if repo > 8.0 else (0.3 if repo < 5.5 else 0.0)

    bull_bear = (breadth * 0.5 + avg_sentiment * 0.3 + nrb_score * 0.2)
    bull_bear = max(-1.0, min(1.0, bull_bear))
    confidence = min(80, abs(bull_bear) * 100 + 30)

    action = "BUY" if bull_bear > 0.3 else ("SELL" if bull_bear < -0.3 else "HOLD")

    return {
        "date": d.isoformat(),
        "bull_bear_score": round(bull_bear, 3),
        "score": round(bull_bear, 3),
        "confidence_pct": round(confidence, 1),
        "action": action,
        "sector_signals": {
            "banking": round(bull_bear * 0.9 + nrb_score * 0.3, 3),
            "hydropower": round(bull_bear * 1.1, 3),
            "insurance": round(bull_bear * 0.8, 3),
            "finance": round(bull_bear * 0.85 + nrb_score * 0.4, 3),
            "microfinance": round(bull_bear * 0.7, 3),
            "manufacturing": round(bull_bear * 0.6, 3),
        },
        "key_themes": ["market_breadth", "news_sentiment", "nrb_policy"],
        "top_driver_agent_types": ["institutional_broker", "nrb_policy_watcher"],
        "source": "rule_based_fallback",
    }


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

def regenerate_historical_seeds(
    start_date: str,
    end_date: str,
    force_regenerate: bool = False,
    use_claude: bool = True,
) -> dict:
    """
    For each trading day in range:
    1. Check if signal already exists in DB
    2. If missing (or force_regenerate): reconstruct from raw data
    3. Generate synthetic MiroFish signal
    4. Store in DB

    Returns coverage stats dict.
    """
    from backtest.calendar import get_trading_days
    trading_days = get_trading_days(start_date, end_date)

    stats = {
        "total_days": len(trading_days),
        "already_existed": 0,
        "generated": 0,
        "failed": 0,
        "no_data": 0,
    }

    logger.info(
        "Regenerating seeds for %d trading days (%s → %s)",
        len(trading_days), start_date, end_date,
    )

    for i, d in enumerate(trading_days):
        if i % 50 == 0:
            logger.info("Progress: %d/%d days", i, len(trading_days))

        # Check if already exists
        if not force_regenerate:
            existing = _existing_signal(d)
            if existing and existing.get("source") not in ("fallback", None):
                stats["already_existed"] += 1
                continue

        # Gather raw data
        market = _get_market_snapshot(d)
        news = _get_news_articles(d)
        nrb = _get_nrb_rates(d)

        # Check minimum data availability
        if not market and not news:
            logger.debug("No data available for %s — skipping", d)
            stats["no_data"] += 1
            continue

        # Generate signal
        try:
            signal = _generate_synthetic_signal(d, market, news, nrb) if use_claude \
                else _rule_based_signal(d, market, news, nrb)

            if signal:
                _save_signal(d, signal)
                stats["generated"] += 1
            else:
                stats["failed"] += 1
        except Exception as exc:
            logger.warning("Seed generation failed for %s: %s", d, exc)
            stats["failed"] += 1

    logger.info(
        "Seed regeneration complete: %d existed, %d generated, %d failed, %d no_data",
        stats["already_existed"], stats["generated"],
        stats["failed"], stats["no_data"],
    )
    return stats


def check_data_coverage(start_date: str, end_date: str) -> dict:
    """
    Report what % of trading days have sufficient data for backtesting.
    Checks: market snapshots, news articles, NRB rates, simulation seeds.
    """
    from backtest.calendar import get_trading_days
    trading_days = get_trading_days(start_date, end_date)
    engine = _get_engine()

    total = len(trading_days)
    coverage = {
        "total_trading_days": total,
        "market_snapshots": 0,
        "news_articles": 0,
        "nrb_rates": 0,
        "simulation_seeds": 0,
        "monthly_coverage": {},
    }

    if not engine:
        print("WARNING: Database not available. Cannot check coverage.")
        return coverage

    from sqlalchemy import text

    with engine.connect() as conn:
        for d in trading_days:
            month_key = d.strftime("%Y-%m")
            if month_key not in coverage["monthly_coverage"]:
                coverage["monthly_coverage"][month_key] = {
                    "days": 0, "snapshots": 0, "news": 0, "seeds": 0
                }
            coverage["monthly_coverage"][month_key]["days"] += 1

            # Market snapshot
            try:
                row = conn.execute(
                    text("SELECT 1 FROM market_snapshots WHERE date = :d LIMIT 1"),
                    {"d": d}
                ).fetchone()
                if row:
                    coverage["market_snapshots"] += 1
                    coverage["monthly_coverage"][month_key]["snapshots"] += 1
            except Exception:
                pass

            # News articles (at least 5 in last 3 days)
            try:
                count = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM news_articles "
                        "WHERE published_at BETWEEN :start AND :end"
                    ),
                    {"start": d - timedelta(days=3), "end": d},
                ).scalar()
                if (count or 0) >= 5:
                    coverage["news_articles"] += 1
                    coverage["monthly_coverage"][month_key]["news"] += 1
            except Exception:
                pass

            # NRB rates
            try:
                row = conn.execute(
                    text(
                        "SELECT 1 FROM nrb_rates "
                        "WHERE effective_date <= :d LIMIT 1"
                    ),
                    {"d": d},
                ).fetchone()
                if row:
                    coverage["nrb_rates"] += 1
            except Exception:
                pass

            # Simulation seed
            existing = _existing_signal(d)
            if existing and existing.get("source") not in ("fallback", None):
                coverage["simulation_seeds"] += 1
                coverage["monthly_coverage"][month_key]["seeds"] += 1

    # Compute percentages
    if total > 0:
        coverage["market_snapshots_pct"] = round(coverage["market_snapshots"] / total * 100, 1)
        coverage["news_articles_pct"] = round(coverage["news_articles"] / total * 100, 1)
        coverage["nrb_rates_pct"] = round(coverage["nrb_rates"] / total * 100, 1)
        coverage["simulation_seeds_pct"] = round(coverage["simulation_seeds"] / total * 100, 1)

    _print_coverage_report(coverage)
    return coverage


def _print_coverage_report(coverage: dict) -> None:
    """Print a formatted coverage report."""
    total = coverage["total_trading_days"]
    print(f"\n{'='*60}")
    print(f"  NEPSE Backtest Data Coverage Report")
    print(f"{'='*60}")
    print(f"  Total trading days: {total}")
    print(f"\n  {'Data Source':<25} {'Days':>6} {'Coverage':>9}")
    print(f"  {'-'*42}")

    items = [
        ("Market Snapshots", "market_snapshots"),
        ("News Articles (>=5)", "news_articles"),
        ("NRB Rate Data", "nrb_rates"),
        ("Simulation Seeds", "simulation_seeds"),
    ]
    for label, key in items:
        count = coverage.get(key, 0)
        pct = coverage.get(f"{key}_pct", 0.0)
        colour = "\033[92m" if pct >= 80 else ("\033[93m" if pct >= 60 else "\033[91m")
        reset = "\033[0m"
        print(f"  {label:<25} {count:>6} {colour}{pct:>8.1f}%{reset}")

    print(f"\n  {'Month':<10} {'Days':>5} {'Snapshots':>9} {'News':>6} {'Seeds':>7}")
    print(f"  {'-'*42}")
    for month, data in sorted(coverage.get("monthly_coverage", {}).items()):
        days = data["days"]
        snap_pct = data["snapshots"] / days * 100 if days else 0
        news_pct = data["news"] / days * 100 if days else 0
        seed_pct = data["seeds"] / days * 100 if days else 0
        flag = "!! " if snap_pct < 60 or seed_pct < 60 else "  "
        print(
            f"  {flag}{month:<10} {days:>5} "
            f"{snap_pct:>8.0f}% {news_pct:>5.0f}% {seed_pct:>6.0f}%"
        )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Historical seed management")
    parser.add_argument("--check-coverage", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if seed already exists")
    parser.add_argument("--no-claude", action="store_true",
                        help="Use rule-based fallback instead of Claude API")
    args = parser.parse_args()

    if args.check_coverage:
        check_data_coverage(args.start, args.end)

    if args.regenerate:
        stats = regenerate_historical_seeds(
            args.start, args.end,
            force_regenerate=args.force,
            use_claude=not args.no_claude,
        )
        print(f"\nRegeneration complete: {stats}")


if __name__ == "__main__":
    main()
