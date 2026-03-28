"""
scrapers/nepse_stocks.py

Scrapes per-stock detail from merolagani.com/CompanyDetail.aspx.

Data fetched per symbol
-----------------------
* LTP, open, high, low, close (previous close used as 'close')
* 52-week high and low
* EPS, P/E ratio, book value, market cap
* Total listed shares, paidup value
* Latest dividend history (last 3 entries — cash + bonus)
* Promoter vs public shareholding: loaded via SignalR live push on the
  real site; not available from a static HTTP scrape — fields set to None.

Usage
-----
    from scrapers.nepse_stocks import scrape_all_stocks
    results = scrape_all_stocks(["NABIL", "NHPC", "NLICL"])

    # or run directly:
    python scrapers/nepse_stocks.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3
RETRY_DELAY     = 2
REQUEST_TIMEOUT = 20
MAX_WORKERS     = 5

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DividendEntry:
    fiscal_year: str
    cash_pct:    Optional[float]
    bonus_pct:   Optional[float]


@dataclass
class StockDetail:
    symbol:       str
    company_name: str
    sector:       str
    scraped_at:   str                        # ISO-8601 UTC timestamp

    # Price
    ltp:          Optional[float] = None    # last traded price
    open:         Optional[float] = None
    high:         Optional[float] = None
    low:          Optional[float] = None
    prev_close:   Optional[float] = None    # previous close (PClose)
    pct_change:   Optional[float] = None

    # 52-week range
    week52_high:  Optional[float] = None
    week52_low:   Optional[float] = None

    # Fundamentals
    eps:          Optional[float] = None
    eps_fy:       Optional[str]   = None
    pe_ratio:     Optional[float] = None
    book_value:   Optional[float] = None
    market_cap:   Optional[float] = None    # NPR

    # Share structure
    listed_shares: Optional[float] = None
    paidup_value:  Optional[float] = None   # face value per share (NPR)
    public_float:  Optional[float] = None   # not available via static scrape
    avg_volume_30d: Optional[float] = None

    # Shareholding — loaded via SignalR on real site; unavailable statically
    promoter_pct: Optional[float] = None
    public_pct:   Optional[float] = None

    # Dividend history (last 3 fiscal years)
    dividends:    list[DividendEntry] = field(default_factory=list)

    # Errors
    error:        Optional[str] = None


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _num(value) -> Optional[float]:
    """Strip commas/whitespace and coerce to float."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _fetch(session: requests.Session, url: str, method: str = "GET",
           **kwargs) -> requests.Response:
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
    raise RuntimeError(f"[{method}] {url} failed after {MAX_RETRIES} attempts — {last_err}")


def _session(warm_url: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    s.get(warm_url, timeout=REQUEST_TIMEOUT)
    return s


# ── Shared data loader ────────────────────────────────────────────────────────

def fetch_latest_market_ohlc() -> dict[str, dict]:
    """
    Fetch merolagani LatestMarket page and return a dict keyed by symbol.
    Each value has keys: open, high, low, ltp, pct_change, volume, prev_close.

    This is called ONCE and shared across all symbol scrapers.
    """
    s = _session("https://merolagani.com/LatestMarket.aspx")
    resp = _fetch(s, "https://merolagani.com/LatestMarket.aspx")
    soup = BeautifulSoup(resp.text, "lxml")

    ohlc: dict[str, dict] = {}
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        # Main table: Symbol, LTP, % Change, Open, High, Low, Qty, PClose
        if "Symbol" in headers and "Open" in headers and "High" in headers:
            col = {h: i for i, h in enumerate(headers)}
            for row in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if not cells or not cells[0]:
                    continue
                # Strip the <a> link text — cells[0] is the symbol text
                sym = cells[col["Symbol"]]

                def _cell(key: str) -> Optional[float]:
                    idx = col.get(key)
                    return _num(cells[idx]) if idx is not None and idx < len(cells) else None

                # NOTE: merolagani's LatestMarket column headers are mislabeled.
                # Actual data order: header"Open"=High, header"High"=Low,
                # header"Low"=Open.  Verified empirically against live prices.
                ohlc[sym] = {
                    "ltp":        _cell("LTP"),
                    "open":       _cell("Low"),      # header "Low"  → actual Open
                    "high":       _cell("Open"),     # header "Open" → actual High
                    "low":        _cell("High"),     # header "High" → actual Low
                    "pct_change": _cell("% Change"),
                    "volume":     _cell("Qty."),
                    "prev_close": _cell("PClose"),
                }
            break   # only need the first matching table

    return ohlc


# ── Per-symbol scraper ────────────────────────────────────────────────────────

def _parse_accordion(table) -> dict:
    """Parse the key-value accordion TABLE 0 on the company detail page."""
    data: dict[str, str] = {}
    for row in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        # Only take clean 2-cell rows (skip the nested dividend sub-table rows)
        if len(cells) == 2 and cells[0] and not cells[0].startswith("#"):
            data[cells[0]] = cells[1]
    return data


def _parse_dividends(tables) -> list[DividendEntry]:
    """
    Parse dividend (cash) and bonus tables from the company page.
    Returns last 3 unique fiscal years combined.
    """
    by_fy: dict[str, DividendEntry] = {}

    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        # Dividend table: #, Fiscal Year, Value  OR  #, Value, Fiscal Year
        if "Fiscal Year" not in headers and "Value" not in headers:
            continue

        fy_idx  = headers.index("Fiscal Year") if "Fiscal Year" in headers else None
        val_idx = headers.index("Value") if "Value" in headers else None
        if fy_idx is None or val_idx is None:
            continue

        # Determine if this is bonus or cash based on context
        # The page lists cash divs first then bonus divs
        # We detect by checking the header row's parent section text
        parent_text = (table.parent.get_text() if table.parent else "").lower()
        is_bonus = "bonus" in parent_text

        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) <= max(fy_idx, val_idx):
                continue
            raw_fy  = cells[fy_idx].strip("() ")
            raw_val = cells[val_idx].strip().replace("%", "")
            val = _num(raw_val)
            if not raw_fy or val is None:
                continue

            entry = by_fy.setdefault(raw_fy, DividendEntry(raw_fy, None, None))
            if is_bonus:
                entry.bonus_pct = val
            else:
                entry.cash_pct = val

    # Sort by FY descending and return last 3
    sorted_fys = sorted(by_fy.values(),
                        key=lambda e: e.fiscal_year, reverse=True)
    return sorted_fys[:3]


def _parse_company_detail_table(tables) -> dict:
    """Parse TABLE 4 (Symbol, Company Name, Sector, Listed Shares, …)."""
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Listed Shares" in headers or "Symbol" in headers:
            data: dict[str, str] = {}
            for row in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cells) == 2:
                    data[cells[0]] = cells[1]
            if "Listed Shares" in data:
                return data
    return {}


def scrape_stock(symbol: str,
                 ohlc_cache: dict[str, dict] | None = None) -> StockDetail:
    """
    Scrape all available data for *symbol* from merolagani.

    *ohlc_cache* is the pre-fetched LatestMarket dict (shared across threads).
    If None, LatestMarket is fetched independently for this symbol.
    """
    now = datetime.now(timezone.utc).isoformat()
    detail = StockDetail(symbol=symbol.upper(), company_name="", sector="",
                         scraped_at=now)

    try:
        sym = symbol.upper()
        url = f"https://merolagani.com/CompanyDetail.aspx?symbol={sym}"
        s   = _session(url)

        # ── 1. Company detail page ─────────────────────────────────────────
        resp = _fetch(s, url)
        soup = BeautifulSoup(resp.text, "lxml")
        tables = soup.find_all("table")

        if not tables:
            detail.error = "No tables found on company page"
            return detail

        # Accordion (key-value stats)
        accordion = _parse_accordion(tables[0])

        # Dividend history
        detail.dividends = _parse_dividends(tables)

        # Company info table
        info = _parse_company_detail_table(tables)

        # ── 2. Summary API ─────────────────────────────────────────────────
        api_resp = _fetch(
            s,
            f"https://merolagani.com/handlers/webrequesthandler.ashx"
            f"?type=get_company_summary&symbol={sym}",
            headers={"X-Requested-With": "XMLHttpRequest",
                     "Referer": url},
        )
        api = api_resp.json()

        # ── 3. OHLC from cache (or fresh LatestMarket fetch) ───────────────
        if ohlc_cache is None:
            ohlc_cache = fetch_latest_market_ohlc()
        ohlc = ohlc_cache.get(sym, {})

        # ── 4. Parse 52-week range  "562.00-471.00" ────────────────────────
        w52_raw = accordion.get("52 Weeks High - Low", "")
        w52_parts = w52_raw.split("-")
        w52_high = _num(w52_parts[0]) if len(w52_parts) >= 2 else None
        w52_low  = _num(w52_parts[-1]) if len(w52_parts) >= 2 else None

        # ── 5. Populate StockDetail ────────────────────────────────────────
        detail.company_name  = info.get("Company Name") or api.get("name", sym)
        detail.sector        = info.get("Sector") or accordion.get("Sector", "")

        detail.ltp           = _num(api.get("ltp")) or ohlc.get("ltp")
        detail.open          = ohlc.get("open")
        detail.high          = ohlc.get("high")
        detail.low           = ohlc.get("low")
        detail.prev_close    = ohlc.get("prev_close")
        detail.pct_change    = _num(api.get("percentChange")) or ohlc.get("pct_change")

        detail.week52_high   = _num(api.get("fiftyTwoHigh")) or w52_high
        detail.week52_low    = _num(api.get("fiftyTwoLow"))  or w52_low

        detail.eps           = _num(api.get("eps"))
        detail.eps_fy        = api.get("epsFY")
        detail.pe_ratio      = _num(api.get("peRatio"))
        detail.book_value    = _num(accordion.get("Book Value"))
        detail.market_cap    = _num(api.get("marketCap")) or _num(accordion.get("Market Capitalization"))

        detail.listed_shares   = _num(info.get("Listed Shares")) or _num(accordion.get("Shares Outstanding"))
        detail.paidup_value    = _num(info.get("Paidup Value"))
        detail.avg_volume_30d  = _num(accordion.get("30-Day Avg Volume"))

        # Promoter/public: NOT available via static scrape (requires SignalR)
        detail.promoter_pct = None
        detail.public_pct   = None

    except Exception as exc:
        detail.error = str(exc)

    return detail


# ── Concurrent multi-symbol scraper ──────────────────────────────────────────

def scrape_all_stocks(symbols: list[str]) -> list[StockDetail]:
    """
    Scrape *symbols* concurrently (up to MAX_WORKERS threads).

    Fetches LatestMarket OHLC once, then fans out symbol scrapes.
    Saves results to data/processed/stocks_YYYY-MM-DD.json and returns list.
    """
    print(f"  Fetching LatestMarket OHLC (shared cache)…", flush=True)
    ohlc_cache = fetch_latest_market_ohlc()
    print(f"  OHLC loaded for {len(ohlc_cache)} symbols.", flush=True)

    results: list[StockDetail] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(scrape_stock, sym, ohlc_cache): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                stock = future.result()
            except Exception as exc:
                stock = StockDetail(sym, "", "", datetime.now(timezone.utc).isoformat(),
                                    error=str(exc))
            status = f"ERROR: {stock.error}" if stock.error else f"LTP={stock.ltp}"
            print(f"  [{sym}] {status}", flush=True)
            results.append(stock)

    # Sort by symbol for deterministic output
    results.sort(key=lambda s: s.symbol)

    # ── Save to processed/ ─────────────────────────────────────────────────
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = PROCESSED_DIR / f"stocks_{today}.json"
    payload  = [asdict(r) for r in results]
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Saved {len(results)} records -> {out_path}")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    SYMBOLS = ["NABIL", "NHPC", "NLICL", "NLIC", "UPPER"]

    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    RESET  = "\033[0m"

    def hr(char: str = "-", width: int = 68) -> None:
        print(CYAN + char * width + RESET)

    print(f"\n{BOLD}Scraping stock details for: {', '.join(SYMBOLS)}{RESET}\n")

    results = scrape_all_stocks(SYMBOLS)

    for s in results:
        print()
        hr("=")
        err_tag = f"  {RED}ERROR: {s.error}{RESET}" if s.error else ""
        print(f"{BOLD}{CYAN}  {s.symbol}  —  {s.company_name}{RESET}  "
              f"[{s.sector}]{err_tag}")
        hr()

        if s.error:
            continue

        # Price block
        ltp_col = GREEN if (s.pct_change or 0) >= 0 else RED
        sign    = "+" if (s.pct_change or 0) >= 0 else ""
        print(f"  {'LTP':<24} {ltp_col}{s.ltp:>12,.2f}{RESET}   "
              f"{ltp_col}{sign}{s.pct_change:.2f}%{RESET}")
        for label, val in [("Open", s.open), ("High", s.high),
                            ("Low", s.low), ("Prev Close", s.prev_close)]:
            if val is not None:
                print(f"  {label:<24} {val:>12,.2f}")
        print(f"  {'52w High':<24} {s.week52_high:>12,.2f}" if s.week52_high else "")
        print(f"  {'52w Low':<24} {s.week52_low:>12,.2f}"  if s.week52_low  else "")

        # Fundamentals
        print()
        for label, val, fmt in [
            ("EPS",          s.eps,          f"{s.eps:.2f} (FY {s.eps_fy})" if s.eps else "N/A"),
            ("P/E Ratio",    s.pe_ratio,     f"{s.pe_ratio:.2f}" if s.pe_ratio else "N/A"),
            ("Book Value",   s.book_value,   f"{s.book_value:,.2f}" if s.book_value else "N/A"),
            ("Market Cap",   s.market_cap,
             f"NPR {s.market_cap/1e9:,.2f} Bn" if s.market_cap else "N/A"),
        ]:
            print(f"  {label:<24} {fmt}")

        # Share structure
        print()
        for label, val in [
            ("Listed Shares",   f"{s.listed_shares:,.0f}" if s.listed_shares else "N/A"),
            ("Paidup Value",    f"NPR {s.paidup_value:.2f}" if s.paidup_value else "N/A"),
            ("30d Avg Volume",  f"{s.avg_volume_30d:,.0f}" if s.avg_volume_30d else "N/A"),
            ("Promoter %",      f"{s.promoter_pct:.2f}%" if s.promoter_pct else
             f"{YELLOW}N/A (requires JS){RESET}"),
            ("Public %",        f"{s.public_pct:.2f}%" if s.public_pct else
             f"{YELLOW}N/A (requires JS){RESET}"),
        ]:
            print(f"  {label:<24} {val}")

        # Dividends
        if s.dividends:
            print()
            print(f"  {BOLD}Dividend History (last 3 FY){RESET}")
            print(f"  {'Fiscal Year':<14} {'Cash %':>8}  {'Bonus %':>8}")
            print(f"  {'-'*14} {'-'*8}  {'-'*8}")
            for d in s.dividends:
                cash_s  = f"{d.cash_pct:.2f}%" if d.cash_pct  is not None else "  n/a"
                bonus_s = f"{d.bonus_pct:.2f}%" if d.bonus_pct is not None else "  n/a"
                print(f"  {d.fiscal_year:<14} {cash_s:>8}  {bonus_s:>8}")

    hr("=")
    print()
    sys.exit(0)
