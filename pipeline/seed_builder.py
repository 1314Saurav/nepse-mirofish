"""
pipeline/seed_builder.py

Builds the daily MiroFish seed JSON and a 200-word market brief.

Loads today's scraped data files:
  - data/processed/YYYY-MM-DD_market snapshot (from nepse_market)   [may be named 2026-03-25_market.html or stored in raw]
  - data/processed/stocks_YYYY-MM-DD.json
  - data/processed/news_YYYY-MM-DD.json
  - data/processed/nrb_policy_YYYY-MM-DD.json
  - data/processed/ipo_calendar.json

Output
------
  data/seed/seed_YYYY-MM-DD.json   — structured seed dict
  Returns (seed_dict, brief_text)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR  = _ROOT / "data" / "processed"
SEED_DIR  = _ROOT / "data" / "seed"
SEED_DIR.mkdir(parents=True, exist_ok=True)

# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [WARN] Could not read {path}: {exc}", file=sys.stderr)
        return None


def _find_today_file(prefix: str, suffix: str = ".json") -> Optional[Path]:
    """Find data/processed/<prefix>_YYYY-MM-DD<suffix> for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    exact = PROC_DIR / f"{prefix}_{today}{suffix}"
    if exact.exists():
        return exact
    # Fall back to most recent file matching prefix
    candidates = sorted(PROC_DIR.glob(f"{prefix}_*{suffix}"), reverse=True)
    return candidates[0] if candidates else None


def load_today_data(date_str: Optional[str] = None) -> dict:
    """
    Load all available scraped data for `date_str` (default: today).

    Returns dict with keys: market, stocks, news, nrb, ipo.
    Missing files result in None values — seed builder handles gracefully.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    def _dated(name: str) -> Optional[Path]:
        p = PROC_DIR / f"{name}_{date_str}.json"
        return p if p.exists() else None

    # Market snapshot may be in raw/ as HTML (containing JSON) — fall back
    market_path = _dated("market") or _find_today_file("market")
    stocks_path = _dated("stocks") or _find_today_file("stocks")
    news_path   = _dated("news")   or _find_today_file("news")
    nrb_path    = _dated("nrb_policy") or _find_today_file("nrb_policy")
    ipo_path    = PROC_DIR / "ipo_calendar.json"

    return {
        "market": _load_json(market_path)  if market_path else None,
        "stocks": _load_json(stocks_path)  if stocks_path else None,
        "news":   _load_json(news_path)    if news_path   else None,
        "nrb":    _load_json(nrb_path)     if nrb_path    else None,
        "ipo":    _load_json(ipo_path)     if ipo_path.exists() else None,
    }


# ── Seed assembly ──────────────────────────────────────────────────────────────

def _build_market_summary(market: Optional[dict]) -> dict:
    if not market:
        return {}
    idx = market.get("nepse_index", {})
    mo  = market.get("market_overview", {})
    return {
        "nepse_index":        idx.get("close"),
        "nepse_point_change": idx.get("point_change"),
        "nepse_pct_change":   idx.get("pct_change"),
        "sensitive_index":    market.get("sensitive_index", {}).get("close"),
        "float_index":        market.get("float_index", {}).get("close"),
        "banking_index":      market.get("banking_subindex", {}).get("close"),
        "total_turnover_npr": mo.get("total_turnover_npr"),
        "total_traded_shares":mo.get("total_traded_shares"),
        "total_transactions": mo.get("total_transactions"),
        "scrips_traded":      mo.get("total_scrips_traded"),
        "top_gainers": market.get("top_gainers", [])[:5],
        "top_losers":  market.get("top_losers",  [])[:5],
    }


def _build_macro_context(nrb: Optional[dict]) -> dict:
    if not nrb:
        return {}
    ir  = nrb.get("interest_rates", {})
    rr  = nrb.get("reserve_requirements", {})
    fx  = nrb.get("forex", {}).get("current", {})
    cd  = nrb.get("credit_deposit_ratio", {})
    return {
        "bank_rate":            ir.get("bank_rate"),
        "repo_rate":            ir.get("repo_rate"),
        "reverse_repo_rate":    ir.get("reverse_repo"),
        "crr_pct":              rr.get("crr_pct"),
        "slr_pct":              rr.get("slr_pct"),
        "usd_npr":              fx.get("USD", {}).get("buy"),
        "eur_npr":              fx.get("EUR", {}).get("buy"),
        "inr_npr":              fx.get("INR", {}).get("buy"),
        "cny_npr":              fx.get("CNY", {}).get("buy"),
        "credit_deposit_ratio": cd.get("ratio"),
        "cd_ratio_as_of":       cd.get("as_of"),
    }


def _build_news_list(news: Optional[dict | list]) -> list[dict]:
    """Extract articles array from news payload."""
    if news is None:
        return []
    if isinstance(news, list):
        articles = news
    else:
        articles = news.get("articles", news.get("items", []))
    result = []
    for a in articles[:30]:           # cap at 30 in seed
        result.append({
            "title":        a.get("title", ""),
            "body_excerpt": (a.get("body", "") or "")[:500],
            "category":     a.get("category", "general_market"),
            "source":       a.get("source_name", a.get("source", "")),
            "url":          a.get("url", ""),
            "published_at": a.get("published_at", ""),
        })
    return result


def _build_ipo_events(ipo: Optional[dict]) -> list[str]:
    if not ipo:
        return []
    lines = []
    for item in ipo.get("open_ipos", []):
        lines.append(f"OPEN IPO: {item.get('company', item.get('name','?'))} "
                     f"@ NPR {item.get('issue_price','?')} "
                     f"(closes {item.get('close_date','?')})")
    for item in ipo.get("upcoming_ipos", []):
        lines.append(f"UPCOMING: {item.get('company', item.get('name','?'))} "
                     f"(opens {item.get('open_date','?')})")
    for item in ipo.get("bonus_right_shares", [])[:5]:
        company = item.get("company", item.get("issuer", "?"))
        lines.append(f"BONUS/RIGHT: {company} — {item.get('detail', item.get('announcementDetail',''))[:80]}")
    return lines


# ── Anthropic brief ────────────────────────────────────────────────────────────

def generate_market_brief(seed: dict) -> str:
    """
    Call claude-sonnet-4-6 to write a 200-word plain-English market brief
    from the seed data. Returns the brief text.
    Falls back to a rule-based summary if API key is missing.
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return _fallback_brief(seed)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        ms   = seed.get("market_summary", {})
        mc   = seed.get("macro_context", {})
        ipos = seed.get("ipo_events", [])

        prompt = f"""You are a Nepali stock market analyst. Write a concise 200-word plain-English
market brief for today ({seed['date']}) based on the data below. Focus on:
1. NEPSE index performance (direction + magnitude)
2. Key macro drivers (NRB rates, forex)
3. Notable IPO or sector events
4. 1-sentence forward-looking note

Data:
- NEPSE Index: {ms.get('nepse_index')} (change: {ms.get('nepse_pct_change')}%)
- Turnover: NPR {ms.get('total_turnover_npr'):,.0f} if {ms.get('total_turnover_npr')} else 'N/A'
- Repo Rate: {mc.get('repo_rate')}%  Bank Rate: {mc.get('bank_rate')}%
- USD/NPR: {mc.get('usd_npr')}  CD Ratio: {mc.get('credit_deposit_ratio')}%
- Top Gainers: {[g['symbol'] for g in ms.get('top_gainers', [])[:3]]}
- Top Losers:  {[l['symbol'] for l in ms.get('top_losers', [])[:3]]}
- IPO Events: {ipos[:3]}

Write exactly 200 words, no bullet points, plain prose suitable for a market briefing email."""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    except Exception as exc:
        print(f"  [WARN] Anthropic API call failed: {exc}", file=sys.stderr)
        return _fallback_brief(seed)


def _fallback_brief(seed: dict) -> str:
    """Rule-based brief when Anthropic API is unavailable."""
    ms   = seed.get("market_summary", {})
    mc   = seed.get("macro_context", {})
    ipos = seed.get("ipo_events", [])

    idx   = ms.get("nepse_index", "N/A")
    chg   = ms.get("nepse_pct_change", 0) or 0
    tone  = "gained" if chg >= 0 else "fell"
    turn  = ms.get("total_turnover_npr")
    turn_s = f"NPR {turn:,.0f}" if turn else "N/A"

    gainers = ", ".join(g["symbol"] for g in ms.get("top_gainers", [])[:3]) or "N/A"
    losers  = ", ".join(l["symbol"] for l in ms.get("top_losers",  [])[:3]) or "N/A"

    ipo_line = f" IPO activity: {ipos[0]}." if ipos else ""

    return (
        f"NEPSE {tone} {abs(chg):.2f}% to close at {idx} on {seed['date']}. "
        f"Market turnover stood at {turn_s}. "
        f"Top gainers included {gainers}; notable losers were {losers}. "
        f"NRB repo rate at {mc.get('repo_rate')}%, bank rate at {mc.get('bank_rate')}%. "
        f"USD/NPR exchange rate: {mc.get('usd_npr', 'N/A')}."
        f"{ipo_line} "
        f"[Auto-generated brief — set ANTHROPIC_API_KEY for AI-written summary.]"
    )


# ── Master function ────────────────────────────────────────────────────────────

def build_daily_seed(date_str: Optional[str] = None) -> tuple[dict, str]:
    """
    Build and save the daily MiroFish seed JSON + 200-word brief.

    Returns (seed_dict, brief_text).
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"  Loading scraped data for {date_str}...", flush=True)
    data = load_today_data(date_str)

    market_summary = _build_market_summary(data["market"])
    macro_context  = _build_macro_context(data["nrb"])
    news_articles  = _build_news_list(data["news"])
    ipo_events     = _build_ipo_events(data["ipo"])

    seed: dict = {
        "date":                date_str,
        "market_summary":      market_summary,
        "macro_context":       macro_context,
        "news_articles":       news_articles,
        "ipo_events":          ipo_events,
        "political_context":   "",        # manually editable
        "simulation_question": (
            "How might NEPSE move in the next 3 trading days given these conditions?"
        ),
    }

    print("  Generating market brief via Anthropic API...", flush=True)
    brief = generate_market_brief(seed)
    seed["brief_text"] = brief

    out_path = SEED_DIR / f"seed_{date_str}.json"
    out_path.write_text(
        json.dumps(seed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved -> {out_path}", flush=True)
    return seed, brief


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = p.parse_args()

    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    RESET = "\033[0m"

    print(f"\n{BOLD}Building daily MiroFish seed...{RESET}\n")
    seed, brief = build_daily_seed(args.date)

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Seed JSON{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    print(json.dumps(seed, indent=2, ensure_ascii=False)[:3000])
    if len(json.dumps(seed)) > 3000:
        print("  ... [truncated for display]")

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  200-Word Market Brief{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")
    print(f"\n{brief}\n")
