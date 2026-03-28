"""
pipeline/sector_aggregator.py

Groups NEPSE stocks by sector and computes aggregated metrics.

Sectors
-------
  Commercial Banks, Development Banks, Finance, Hydropower,
  Insurance, Manufacturing, Hotels, Microfinance, Mutual Fund

For each sector:
  - Average P/E ratio
  - Average EPS growth QoQ (requires two consecutive quarters in stocks data)
  - Total sector market cap (NPR)
  - Sector performance vs NEPSE (outperform / underperform / neutral)
  - Top 3 stocks by market cap

Output
------
  data/processed/sector_summary_YYYY-MM-DD.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
PROC_DIR = _ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Sector definitions ─────────────────────────────────────────────────────────

SECTORS = [
    "Commercial Banks",
    "Development Banks",
    "Finance",
    "Hydropower",
    "Insurance",
    "Manufacturing",
    "Hotels",
    "Microfinance",
    "Mutual Fund",
]

# Keyword-based sector mapping (merolagani sector strings -> canonical)
_SECTOR_MAP: dict[str, str] = {
    # Commercial Banks
    "commercial bank":  "Commercial Banks",
    "commercial banks": "Commercial Banks",
    # Development Banks
    "development bank":  "Development Banks",
    "development banks": "Development Banks",
    # Finance
    "finance":           "Finance",
    "finance company":   "Finance",
    # Hydropower
    "hydropower":        "Hydropower",
    "hydro power":       "Hydropower",
    "energy":            "Hydropower",
    "power":             "Hydropower",
    # Insurance
    "insurance":         "Insurance",
    "life insurance":    "Insurance",
    "non-life insurance":"Insurance",
    # Manufacturing
    "manufacturing":     "Manufacturing",
    "manufacturing and processing": "Manufacturing",
    "production":        "Manufacturing",
    # Hotels
    "hotel":             "Hotels",
    "hotels":            "Hotels",
    "tourism":           "Hotels",
    # Microfinance
    "microfinance":      "Microfinance",
    "micro finance":     "Microfinance",
    # Mutual Fund
    "mutual fund":       "Mutual Fund",
    "mutual funds":      "Mutual Fund",
    "investment fund":   "Mutual Fund",
}


def _map_sector(raw: Optional[str]) -> str:
    """Map a raw sector string to one of the canonical SECTORS."""
    if not raw:
        return "Other"
    key = raw.strip().lower()
    for pattern, canonical in _SECTOR_MAP.items():
        if pattern in key:
            return canonical
    return "Other"


# ── Aggregation ────────────────────────────────────────────────────────────────

def _avg(values: list[float]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _sum_or_none(values: list) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean), 2)


def _performance_label(sector_avg_pct: Optional[float],
                        nepse_pct: Optional[float]) -> str:
    if sector_avg_pct is None or nepse_pct is None:
        return "unknown"
    diff = sector_avg_pct - nepse_pct
    if diff > 0.5:
        return "outperform"
    if diff < -0.5:
        return "underperform"
    return "neutral"


def aggregate_sectors(
    stocks_data: list[dict],
    nepse_pct_change: Optional[float] = None,
) -> dict[str, dict]:
    """
    Aggregate `stocks_data` (list of StockDetail dicts) by sector.

    Returns dict keyed by sector name.
    """
    buckets: dict[str, list[dict]] = {s: [] for s in SECTORS}
    buckets["Other"] = []

    for stock in stocks_data:
        raw_sector = stock.get("sector", "")
        canonical  = _map_sector(raw_sector)
        buckets.setdefault(canonical, []).append(stock)

    result: dict[str, dict] = {}

    for sector, stocks in buckets.items():
        if not stocks:
            result[sector] = {
                "stock_count":       0,
                "avg_pe_ratio":      None,
                "avg_eps":           None,
                "total_market_cap":  None,
                "top_3_by_market_cap": [],
                "vs_nepse":          "unknown",
                "avg_pct_change":    None,
            }
            continue

        pe_values   = [s.get("pe_ratio")   for s in stocks]
        eps_values  = [s.get("eps")        for s in stocks]
        mcap_values = [s.get("market_cap") for s in stocks]
        pct_values  = [s.get("pct_change") for s in stocks]

        # Top 3 by market cap
        with_mcap = [(s.get("market_cap") or 0, s.get("symbol", "?"))
                     for s in stocks]
        with_mcap.sort(reverse=True)
        top3 = [sym for _, sym in with_mcap[:3]]

        avg_pct = _avg([v for v in pct_values if v is not None])

        result[sector] = {
            "stock_count":       len(stocks),
            "avg_pe_ratio":      _avg(pe_values),
            "avg_eps":           _avg(eps_values),
            "total_market_cap":  _sum_or_none(mcap_values),
            "top_3_by_market_cap": top3,
            "vs_nepse":          _performance_label(avg_pct, nepse_pct_change),
            "avg_pct_change":    avg_pct,
        }

    return result


# ── File-based entry point ─────────────────────────────────────────────────────

def build_sector_summary(date_str: Optional[str] = None) -> dict:
    """
    Load today's stocks JSON, aggregate by sector, save sector_summary_YYYY-MM-DD.json.

    Returns the sector summary dict.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Load stocks
    stocks_path = PROC_DIR / f"stocks_{date_str}.json"
    if not stocks_path.exists():
        # Fall back to most recent
        candidates = sorted(PROC_DIR.glob("stocks_*.json"), reverse=True)
        stocks_path = candidates[0] if candidates else None

    stocks_data: list[dict] = []
    if stocks_path and stocks_path.exists():
        try:
            raw = json.loads(stocks_path.read_text(encoding="utf-8"))
            stocks_data = raw if isinstance(raw, list) else raw.get("stocks", [])
        except Exception as exc:
            print(f"  [WARN] Could not load stocks: {exc}", file=sys.stderr)

    # Load NEPSE pct change from market snapshot
    nepse_pct: Optional[float] = None
    market_candidates = sorted(PROC_DIR.glob("market_*.json"), reverse=True)
    if market_candidates:
        try:
            mkt = json.loads(market_candidates[0].read_text(encoding="utf-8"))
            nepse_pct = mkt.get("nepse_index", {}).get("pct_change")
        except Exception:
            pass

    print(f"  Aggregating {len(stocks_data)} stocks across {len(SECTORS)} sectors...",
          flush=True)
    sectors = aggregate_sectors(stocks_data, nepse_pct)

    summary = {
        "date":              date_str,
        "total_stocks":      len(stocks_data),
        "nepse_pct_change":  nepse_pct,
        "sectors":           sectors,
        "generated_at":      datetime.now(timezone.utc).isoformat(),
    }

    out_path = PROC_DIR / f"sector_summary_{date_str}.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved -> {out_path}", flush=True)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None)
    args = p.parse_args()

    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"

    print(f"\n{BOLD}Building sector summary...{RESET}\n")
    summary = build_sector_summary(args.date)

    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  Sector Summary — {summary['date']}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}")
    print(f"  {'Sector':<24} {'Stocks':>6}  {'Avg P/E':>8}  {'Mkt Cap Cr':>12}  {'vs NEPSE':<12}")
    print(f"  {'-'*24} {'-'*6}  {'-'*8}  {'-'*12}  {'-'*12}")

    for sector, data in summary["sectors"].items():
        if data["stock_count"] == 0:
            continue
        vs    = data.get("vs_nepse", "unknown")
        color = GREEN if vs == "outperform" else (RED if vs == "underperform" else "")
        mcap  = data.get("total_market_cap")
        mcap_s = f"{mcap/1e7:>12.1f}" if mcap else "         N/A"
        print(
            f"  {sector:<24} {data['stock_count']:>6}  "
            f"{data['avg_pe_ratio'] or 'N/A':>8}  "
            f"{mcap_s}  "
            f"{color}{vs:<12}{RESET}"
        )

    print(f"\n  Total stocks covered: {summary['total_stocks']}\n")
