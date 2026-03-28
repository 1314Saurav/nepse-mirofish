"""
scrapers/nepse_market.py

Fetches live NEPSE market data from sharesansar.com and merolagani.com.

Sources
-------
* sharesansar.com  – main indices (NEPSE, Sensitive, Float) via AJAX,
                     sub-indices table (Banking), market summary stats
* merolagani.com   – JSON market-summary API for all 341 stocks
                     (used to derive top gainers / losers with volume)

Output
------
Returns a ``MarketSnapshot`` dict with a UTC timestamp.
Raw HTML / JSON is saved to data/raw/YYYY-MM-DD_market.html.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR  = _ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_RETRIES  = 3
RETRY_DELAY  = 2          # seconds between retries
REQUEST_TIMEOUT = 20      # seconds per HTTP call

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

# ── Low-level helpers ─────────────────────────────────────────────────────────

def _session(warm_url: str) -> requests.Session:
    """Return a Session pre-warmed at *warm_url* (acquires cookies)."""
    s = requests.Session()
    s.headers.update(_BROWSER_HEADERS)
    s.get(warm_url, timeout=REQUEST_TIMEOUT)
    return s


def _fetch(session: requests.Session, url: str, method: str = "GET",
           **kwargs) -> requests.Response:
    """HTTP request with up to MAX_RETRIES attempts and RETRY_DELAY back-off."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fn = session.post if method == "POST" else session.get
            resp = fn(url, timeout=REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(
        f"[{method}] {url} failed after {MAX_RETRIES} attempts — {last_err}"
    )


def _num(value) -> float | None:
    """Strip commas/whitespace and coerce to float."""
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _pct(close: float | None, point: float | None) -> float | None:
    """% change from close value and absolute point change."""
    if close is None or point is None:
        return None
    prev = close - point
    if prev == 0:
        return None
    return round(point / prev * 100, 2)


# ── Sharesansar fetchers ──────────────────────────────────────────────────────

def _ss_session() -> requests.Session:
    """Warm up a sharesansar session and return it (cookies + XSRF token)."""
    return _session("https://www.sharesansar.com/")


def _ss_xsrf(session: requests.Session) -> str:
    return unquote(session.cookies.get("XSRF-TOKEN", ""))


def fetch_ss_indices(session: requests.Session) -> dict:
    """
    POST home-indices to get NEPSE, Sensitive, Float index values.

    Returns dict keyed by index name → {"close", "point_change", "pct_change"}.
    """
    xsrf = _ss_xsrf(session)
    resp = _fetch(
        session,
        "https://www.sharesansar.com/home-indices",
        method="POST",
        headers={
            "X-XSRF-TOKEN": xsrf,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.sharesansar.com/",
        },
    )
    soup = BeautifulSoup(resp.text, "lxml")
    result: dict[str, dict] = {}
    for row in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        name = cells[0].strip()
        if name in ("Index", "Turnover", "NEPSE\nConfidence"):
            continue
        # Normalise multi-line names produced by JS whitespace
        name = re.sub(r"\s+", " ", name)
        close = _num(cells[1])
        point = _num(cells[2]) if len(cells) > 2 else None
        result[name] = {
            "close":        close,
            "point_change": point,
            "pct_change":   _pct(close, point),
        }
    return result


def fetch_ss_market_summary(session: requests.Session) -> dict:
    """
    GET market-summary page for total turnover, traded shares, etc.

    Returns flat dict of label → value strings (already in the page).
    """
    resp = _fetch(session, "https://www.sharesansar.com/market-summary")
    soup = BeautifulSoup(resp.text, "lxml")
    summary: dict[str, str] = {}
    for row in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) == 2:
            summary[cells[0]] = cells[1]
    return summary


def fetch_ss_banking_subindex(session: requests.Session) -> dict:
    """
    GET sharesansar homepage and parse the sub-indices table for Banking SubIndex.
    """
    resp = _fetch(session, "https://www.sharesansar.com/")
    soup = BeautifulSoup(resp.text, "lxml")
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Sub-Indices" not in headers:
            continue
        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if cells and cells[0] == "Banking SubIndex":
                return dict(zip(headers, cells))
    return {}


# ── Merolagani fetcher ────────────────────────────────────────────────────────

def fetch_ml_market_summary() -> dict:
    """
    Fetch the merolagani JSON market summary API.

    Returns the parsed JSON dict with keys: overall, turnover, sector, stock.
    """
    session = _session("https://merolagani.com/MarketSummary.aspx")
    resp = _fetch(
        session,
        "https://merolagani.com/handlers/webrequesthandler.ashx?type=market_summary",
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://merolagani.com/MarketSummary.aspx",
        },
    )
    return resp.json()


# ── MarketSnapshot builder ────────────────────────────────────────────────────

def scrape_market_snapshot() -> dict:
    """
    Scrape live NEPSE market data and return a ``MarketSnapshot`` dict.

    Saves combined raw data to data/raw/YYYY-MM-DD_market.html.
    """
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Fetch ──────────────────────────────────────────────────────────────────
    ss = _ss_session()

    ss_indices   = fetch_ss_indices(ss)
    ss_summary   = fetch_ss_market_summary(ss)
    ss_banking   = fetch_ss_banking_subindex(ss)
    ml_data      = fetch_ml_market_summary()

    # ── Save raw backup ────────────────────────────────────────────────────────
    raw_path = RAW_DIR / f"{today}_market.html"
    raw_path.write_text(
        "<!-- nepse-mirofish raw market snapshot -->\n"
        f"<!-- timestamp_utc: {timestamp_utc} -->\n\n"
        "<!-- === SHARESANSAR: INDICES === -->\n"
        + json.dumps(ss_indices, indent=2, ensure_ascii=False) + "\n\n"
        "<!-- === SHARESANSAR: MARKET SUMMARY === -->\n"
        + json.dumps(ss_summary, indent=2, ensure_ascii=False) + "\n\n"
        "<!-- === SHARESANSAR: BANKING SUB-INDEX === -->\n"
        + json.dumps(ss_banking, indent=2, ensure_ascii=False) + "\n\n"
        "<!-- === MEROLAGANI: MARKET SUMMARY JSON === -->\n"
        + json.dumps(ml_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Gainers / Losers ───────────────────────────────────────────────────────
    stocks = ml_data.get("stock", {}).get("detail", [])
    enriched: list[dict] = []
    for s in stocks:
        lp     = s.get("lp", 0)
        change = s.get("c", 0)
        prev   = lp - change
        pct    = round(change / prev * 100, 2) if prev else 0.0
        enriched.append({
            "symbol":     s["s"],
            "ltp":        lp,
            "change_pts": change,
            "change_pct": pct,
            "volume":     int(s.get("q", 0)),
        })

    top_gainers = sorted(enriched, key=lambda x: x["change_pct"], reverse=True)[:10]
    top_losers  = sorted(enriched, key=lambda x: x["change_pct"])[:10]

    # ── Extract named indices ──────────────────────────────────────────────────
    nepse     = ss_indices.get("NEPSE Index", {})
    sensitive = ss_indices.get("Sensitive Index", {})
    float_idx = ss_indices.get("Float Index", {})

    # Banking SubIndex from homepage sub-indices table
    banking = {
        "open":         _num(ss_banking.get("Open")),
        "close":        _num(ss_banking.get("Close")),
        "point_change": _num(ss_banking.get("Point")),
        "pct_change":   _num(ss_banking.get("% Change")),
    }

    # ── Market overview (merolagani is authoritative here) ─────────────────────
    overall = ml_data.get("overall", {})

    snapshot: dict = {
        "timestamp_utc": timestamp_utc,
        "as_of_date":    today,
        "sources":       ["sharesansar.com", "merolagani.com"],

        "nepse_index": {
            "close":        nepse.get("close"),
            "point_change": nepse.get("point_change"),
            "pct_change":   nepse.get("pct_change"),
        },
        "sensitive_index": {
            "close":        sensitive.get("close"),
            "point_change": sensitive.get("point_change"),
            "pct_change":   sensitive.get("pct_change"),
        },
        "float_index": {
            "close":        float_idx.get("close"),
            "point_change": float_idx.get("point_change"),
            "pct_change":   float_idx.get("pct_change"),
        },
        "banking_subindex": banking,

        "top_gainers": top_gainers,
        "top_losers":  top_losers,

        "market_overview": {
            "total_turnover_npr":  float(overall.get("t", 0)),
            "total_traded_shares": int(float(overall.get("q", 0))),
            "total_transactions":  int(float(overall.get("tn", 0))),
            "total_scrips_traded": int(float(overall.get("st", 0))),
        },

        "raw_backup_path": str(raw_path),
    }

    return snapshot


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Fetching NEPSE market snapshot …", flush=True)
    try:
        snapshot = scrape_market_snapshot()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Pretty-print
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    RESET  = "\033[0m"

    def section(title: str) -> None:
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{BOLD}{CYAN}  {title}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")

    def idx_line(label: str, d: dict) -> None:
        c   = d.get("close")
        pt  = d.get("point_change")
        pct = d.get("pct_change")
        colour = GREEN if (pt or 0) >= 0 else RED
        sign   = "+" if (pt or 0) >= 0 else ""
        print(f"  {label:<22} {c:>10,.2f}   {colour}{pt:+.2f}  ({sign}{pct:.2f}%){RESET}"
              if c and pt and pct else f"  {label:<22} N/A")

    print(f"\n{BOLD}NEPSE Market Snapshot{RESET}  [{snapshot['timestamp_utc']}]")

    section("Indices")
    idx_line("NEPSE Index",      snapshot["nepse_index"])
    idx_line("Sensitive Index",  snapshot["sensitive_index"])
    idx_line("Float Index",      snapshot["float_index"])
    bi = snapshot["banking_subindex"]
    if bi["close"]:
        sign = "+" if (bi["point_change"] or 0) >= 0 else ""
        colour = GREEN if (bi["point_change"] or 0) >= 0 else RED
        print(f"  {'Banking SubIndex':<22} {bi['close']:>10,.2f}   "
              f"{colour}{bi['point_change']:+.2f}  ({sign}{bi['pct_change']:.2f}%){RESET}")

    section("Market Overview")
    mo = snapshot["market_overview"]
    print(f"  Total Turnover (NPR)   {mo['total_turnover_npr']:>20,.2f}")
    print(f"  Total Traded Shares    {mo['total_traded_shares']:>20,}")
    print(f"  Total Transactions     {mo['total_transactions']:>20,}")
    print(f"  Scrips Traded          {mo['total_scrips_traded']:>20,}")

    section("Top 10 Gainers")
    print(f"  {'Symbol':<12} {'LTP':>8}  {'Change':>8}  {'Volume':>10}")
    print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*10}")
    for g in snapshot["top_gainers"]:
        print(f"  {GREEN}{g['symbol']:<12}{RESET} "
              f"{g['ltp']:>8,.1f}  "
              f"{GREEN}+{g['change_pct']:>6.2f}%{RESET}  "
              f"{g['volume']:>10,}")

    section("Top 10 Losers")
    print(f"  {'Symbol':<12} {'LTP':>8}  {'Change':>8}  {'Volume':>10}")
    print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*10}")
    for l in snapshot["top_losers"]:
        print(f"  {RED}{l['symbol']:<12}{RESET} "
              f"{l['ltp']:>8,.1f}  "
              f"{RED}{l['change_pct']:>7.2f}%{RESET}  "
              f"{l['volume']:>10,}")

    print(f"\n  Raw backup: {snapshot['raw_backup_path']}\n")
