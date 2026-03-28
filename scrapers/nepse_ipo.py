"""
scrapers/nepse_ipo.py

Scrapes IPO, right-share, and bonus-share data from:
  - sharesansar.com  (AJAX/HMINS + DataTables JSON endpoints)
  - sebon.gov.np     (public regulatory tables)

Output  →  data/processed/ipo_calendar.json

Why this matters
----------------
IPO announcements are among the strongest sentiment drivers in NEPSE.
Oversubscription ratios, allotment dates, and listing prices consistently
cause broad index moves that MiroFish needs to account for.

Sections in ipo_calendar.json
------------------------------
  open_ipos          – currently open for application
  upcoming_ipos      – announced but not yet open (last 30 d SEBON approval)
  allotted_ipos      – recently allotted (last 60 days, with cut-off price)
  bonus_right_shares – bonus/right share announcements (last 30 days)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
RAW_DIR       = _ROOT / "data" / "raw"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3
RETRY_DELAY     = 2
REQUEST_TIMEOUT = 20

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_SS_BASE   = "https://www.sharesansar.com"
_SEBON_BASE = "https://sebon.gov.np"


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _fetch(session: requests.Session, url: str, method: str = "GET",
           **kwargs) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fn = session.post if method == "POST" else session.get
            r  = fn(url, timeout=REQUEST_TIMEOUT, **kwargs)
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"[{method}] {url} — {last}")


def _ss_session() -> tuple[requests.Session, str]:
    """
    Create a sharesansar session warmed on the homepage.
    Returns (session, xsrf_token).
    """
    s = requests.Session()
    s.headers.update({"User-Agent": _UA})
    _fetch(s, f"{_SS_BASE}/")           # sets XSRF-TOKEN cookie
    xsrf = unquote(s.cookies.get("XSRF-TOKEN", ""))
    return s, xsrf


def _strip_tags(html_str: str) -> str:
    return re.sub(r"<[^>]+>", "", str(html_str)).strip()


def _num(value) -> Optional[float]:
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_date(s: str) -> Optional[str]:
    """Normalise a date string to ISO-8601; return None if unparseable."""
    if not s or s.strip().lower() in ("coming soon", "n/a", "", "-"):
        return None
    s = s.strip().split("[")[0].strip()   # strip "[Closed]" suffix
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s   # return as-is if no format matched


def _within_days(date_str: Optional[str], days: int) -> bool:
    """True if date_str is within the last *days* calendar days."""
    if not date_str:
        return False
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        cutoff = datetime.now() - timedelta(days=days)
        return d >= cutoff
    except ValueError:
        return False


def _parse_hmins_table(html: str) -> list[dict]:
    """Parse a sharesansar *-hmins HTML fragment into a list of row dicts."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        return []
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows: list[dict] = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        rows.append(dict(zip(headers, cells)))
    return rows


# ── Sharesansar scrapers ──────────────────────────────────────────────────────

def fetch_existing_issues(session: requests.Session, xsrf: str) -> list[dict]:
    """
    POST existing-issues-hmins → all currently announced / open IPOs.

    Returns normalised list of {symbol, company, units, issue_price,
    open_date, close_date, status, is_open}.
    """
    r = _fetch(session, f"{_SS_BASE}/existing-issues-hmins",
               method="POST",
               headers={"X-XSRF-TOKEN": xsrf, "X-Requested-With": "XMLHttpRequest",
                        "Referer": f"{_SS_BASE}/existing-issues"})
    rows = _parse_hmins_table(r.text)
    today = datetime.now().date()
    result = []
    for row in rows:
        open_d  = _parse_date(row.get("Opening Date", ""))
        close_d = _parse_date(row.get("Closing Date", ""))
        # Determine if currently open
        try:
            is_open = (open_d is not None and close_d is not None and
                       datetime.strptime(open_d, "%Y-%m-%d").date() <= today <=
                       datetime.strptime(close_d, "%Y-%m-%d").date())
        except (ValueError, TypeError):
            is_open = False

        result.append({
            "symbol":       row.get("Symbol", ""),
            "company":      row.get("Company", ""),
            "units":        _num(row.get("Units")),
            "issue_price":  _num(row.get("Price")),
            "open_date":    open_d,
            "close_date":   close_d,
            "status":       row.get("Status", "").strip(),
            "is_open":      is_open,
        })
    return result


def fetch_upcoming_issues(session: requests.Session, xsrf: str) -> list[dict]:
    """
    POST upcoming-issue-hmins → IPOs approved / in pipeline, not yet open.
    """
    r = _fetch(session, f"{_SS_BASE}/upcoming-issue-hmins",
               method="POST",
               headers={"X-XSRF-TOKEN": xsrf, "X-Requested-With": "XMLHttpRequest",
                        "Referer": f"{_SS_BASE}/upcoming-issue"})
    rows = _parse_hmins_table(r.text)
    return [
        {
            "symbol":        row.get("Symbol", ""),
            "company":       row.get("Company", ""),
            "units":         _num(row.get("Units")),
            "sector":        row.get("Sector", ""),
            "issue_manager": row.get("Remark", ""),
        }
        for row in rows
    ]


def fetch_allotted_ipos(session: requests.Session, xsrf: str,
                        days: int = 60) -> list[dict]:
    """
    POST auction-hmins → recently allotted IPOs (cut-off price auctions).
    Filters to entries whose closing date is within *days* calendar days.
    """
    r = _fetch(session, f"{_SS_BASE}/auction-hmins",
               method="POST",
               headers={"X-XSRF-TOKEN": xsrf, "X-Requested-With": "XMLHttpRequest",
                        "Referer": f"{_SS_BASE}/existing-issues"})
    rows = _parse_hmins_table(r.text)
    result = []
    for row in rows:
        close_d = _parse_date(row.get("Closing Date", ""))
        if not _within_days(close_d, days):
            continue
        result.append({
            "symbol":          row.get("Symbol", ""),
            "company":         row.get("Company", ""),
            "units":           _num(row.get("Units")),
            "total_allotted":  _num(row.get("Total Alloted")),
            "close_date":      close_d,
            "cutoff_price":    _num(row.get("Cut-off Price")),
            "status":          row.get("Status", "").strip(),
        })
    return result


def fetch_bonus_right_shares(session: requests.Session, days: int = 30) -> list[dict]:
    """
    GET sharesansar/proposed-dividend DataTables endpoint.

    Returns bonus-share and right-share announcements within *days* days.

    NOTE: The endpoint's ``duration`` param uses the server's internal
    ``published_date`` (batch-refresh date), which can lag by a day, making
    ``duration=N`` unreliable.  We always request ``duration=365`` and apply
    our own ``announcement_date`` cutoff instead.
    """
    # Warm up on the dividend page before the DataTable request
    _fetch(session, f"{_SS_BASE}/proposed-dividend")

    r = _fetch(
        session,
        f"{_SS_BASE}/proposed-dividend",
        params={
            "type": "LATEST", "duration": "365",   # broad fetch; we filter below
            "draw": 1, "start": 0, "length": 500,
        },
        headers={
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{_SS_BASE}/proposed-dividend",
        },
    )
    data = r.json()
    result = []
    for item in data.get("data", []):
        ann_date = _parse_date(item.get("announcement_date"))
        if not _within_days(ann_date, days):
            continue
        sym = _strip_tags(item.get("symbol", ""))
        co  = _strip_tags(item.get("companyname", ""))
        result.append({
            "symbol":            sym,
            "company":           co,
            "bonus_pct":         _num(item.get("bonus_share")),
            "cash_dividend_pct": _num(item.get("cash_dividend")),
            "total_dividend_pct":_num(item.get("total_dividend")),
            "fiscal_year":       item.get("year"),
            "announcement_date": ann_date,
            "book_close_date":   _parse_date(item.get("bookclose_date")),
            "bonus_listing_date":_parse_date(item.get("bonus_listing_date")),
            "ltp":               _num(item.get("close")),
        })
    return result


# ── SEBON scraper ─────────────────────────────────────────────────────────────

def _sebon_table(page_url: str) -> list[dict]:
    """
    Fetch a SEBON document-list page and return rows as
    {title, date, english_pdf_url, nepali_pdf_url}.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": _UA})
    r = _fetch(s, page_url)
    soup = BeautifulSoup(r.text, "lxml")

    rows: list[dict] = []
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Title" not in headers:
            continue
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            title  = cells[0].get_text(strip=True)
            date_s = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            en_url = (cells[2].find("a")["href"]
                      if len(cells) > 2 and cells[2].find("a") else None)
            rows.append({
                "title":   title,
                "date":    _parse_date(date_s),
                "pdf_url": en_url,
            })
        break  # only first matching table
    return rows


def fetch_sebon_ipo_pipeline(days: int = 30) -> list[dict]:
    """
    Scrape SEBON IPO pipeline & approved lists.
    Returns document entries within *days* calendar days.
    """
    entries: list[dict] = []
    sources = [
        ("ipo_approved",    f"{_SEBON_BASE}/ipo-approved"),
        ("ipo_pipeline",    f"{_SEBON_BASE}/ipo-pipeline"),
        ("right_approved",  f"{_SEBON_BASE}/right-share-approved"),
        ("right_pipeline",  f"{_SEBON_BASE}/right-share-pipeline"),
    ]
    for category, url in sources:
        try:
            rows = _sebon_table(url)
        except Exception as exc:
            print(f"  SEBON [{category}] error: {exc}")
            continue
        for row in rows:
            if _within_days(row["date"], days):
                entries.append({**row, "category": category})
    return entries


def fetch_sebon_bonus_shares(days: int = 30) -> list[dict]:
    """
    Scrape SEBON bonus-share-registered document list.
    Returns recent entries (within *days*) as metadata records.
    Note: full company details are in the PDFs; titles carry the BS date.
    """
    rows = _sebon_table(f"{_SEBON_BASE}/bonus-share-segistered")
    return [
        {**r, "category": "bonus_share_registered"}
        for r in rows
        if _within_days(r["date"], days)
    ]


# ── Master scrape function ────────────────────────────────────────────────────

def scrape_ipo_calendar() -> dict:
    """
    Fetch all IPO / corporate action data and return a unified calendar dict.
    Saves output to data/processed/ipo_calendar.json.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    today     = datetime.now().strftime("%Y-%m-%d")

    print("  [1/5] Initialising sharesansar session...", flush=True)
    session, xsrf = _ss_session()

    print("  [2/5] Fetching open + existing issues...", flush=True)
    all_existing  = fetch_existing_issues(session, xsrf)
    open_ipos     = [i for i in all_existing if i["is_open"]]
    # "upcoming from sharesansar" = announced but dates say Coming Soon
    ss_upcoming   = [i for i in all_existing if not i["is_open"]]

    print("  [3/5] Fetching pipeline / approved issues...", flush=True)
    pipeline      = fetch_upcoming_issues(session, xsrf)

    print("  [4/5] Fetching allotted IPOs (last 60 days)...", flush=True)
    allotted      = fetch_allotted_ipos(session, xsrf, days=60)

    print("  [5/5] Fetching bonus/right share announcements + SEBON docs...",
          flush=True)
    bonus_right   = fetch_bonus_right_shares(session, days=30)
    sebon_docs    = fetch_sebon_ipo_pipeline(days=30)
    sebon_bonus   = fetch_sebon_bonus_shares(days=30)

    # ── Merge upcoming: sharesansar pipeline + sebon docs ─────────────────
    # De-duplicate by symbol where possible
    known_symbols = {p["symbol"] for p in pipeline if p["symbol"]}
    for item in ss_upcoming:
        if item["symbol"] not in known_symbols:
            pipeline.append({
                "symbol":        item["symbol"],
                "company":       item["company"],
                "units":         item["units"],
                "issue_price":   item["issue_price"],
                "open_date":     item["open_date"],
                "close_date":    item["close_date"],
                "sector":        "",
                "issue_manager": "",
                "source":        "sharesansar_existing",
            })
            known_symbols.add(item["symbol"])

    # ── Build calendar ─────────────────────────────────────────────────────
    calendar = {
        "timestamp_utc":     timestamp,
        "as_of_date":        today,
        "sources":           ["sharesansar.com", "sebon.gov.np"],

        "open_ipos": open_ipos,

        "upcoming_ipos": pipeline,

        "allotted_ipos": allotted,

        "bonus_right_shares":    bonus_right,
        "sebon_recent_docs":     sebon_docs,
        "sebon_bonus_registered": sebon_bonus,

        "summary": {
            "open_count":           len(open_ipos),
            "upcoming_count":       len(pipeline),
            "allotted_60d_count":   len(allotted),
            "bonus_right_30d_count": len(bonus_right),
            "sebon_docs_30d_count":  len(sebon_docs),
            "sebon_bonus_30d_count": len(sebon_bonus),
        },
    }

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "ipo_calendar.json"
    out_path.write_text(json.dumps(calendar, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return calendar, out_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    RESET  = "\033[0m"

    def section(title: str) -> None:
        print(f"\n{CYAN}{'='*64}{RESET}")
        print(f"{BOLD}{CYAN}  {title}{RESET}")
        print(f"{CYAN}{'='*64}{RESET}")

    print(f"\n{BOLD}Scraping IPO calendar...{RESET}\n")

    try:
        calendar, out_path = scrape_ipo_calendar()
    except Exception as exc:
        print(f"{RED}ERROR: {exc}{RESET}")
        sys.exit(1)

    s = calendar["summary"]

    section("Summary")
    for label, key in [
        ("Currently open IPOs",         "open_count"),
        ("Upcoming IPOs",               "upcoming_count"),
        ("Allotted IPOs (last 60d)",    "allotted_60d_count"),
        ("Bonus/right anncts (30d)",    "bonus_right_30d_count"),
        ("SEBON docs (30d)",            "sebon_docs_30d_count"),
        ("SEBON bonus registered (30d)","sebon_bonus_30d_count"),
    ]:
        val = s[key]
        colour = GREEN if val > 0 else YELLOW
        print(f"  {label:<34} {colour}{val:>4}{RESET}")

    section(f"Open IPOs ({s['open_count']})")
    if calendar["open_ipos"]:
        for i in calendar["open_ipos"]:
            print(f"  {GREEN}{i['symbol']:<12}{RESET} {i['company']}")
            print(f"    Units: {i['units']:>12,.0f}   Price: NPR {i['issue_price']}")
            print(f"    Open:  {i['open_date']}   Close: {i['close_date']}")
    else:
        print(f"  {YELLOW}No IPOs currently open.{RESET}")

    section(f"Upcoming IPOs ({s['upcoming_count']})")
    for i in calendar["upcoming_ipos"][:10]:
        sym = i.get("symbol", "")
        co  = i.get("company", "")
        u   = i.get("units")
        sec = i.get("sector", "")
        im  = i.get("issue_manager", "")
        print(f"  {CYAN}{sym:<12}{RESET} {co[:45]}")
        print(f"    Units: {f'{u:>,.0f}' if u else 'N/A':>12}   Sector: {sec}   Manager: {im[:30]}")
    if len(calendar["upcoming_ipos"]) > 10:
        print(f"  ... and {len(calendar['upcoming_ipos'])-10} more")

    section(f"Allotted IPOs — last 60 days ({s['allotted_60d_count']})")
    if calendar["allotted_ipos"]:
        print(f"  {'Symbol':<12} {'Company':<38} {'Close Date':<12} {'Cutoff':>8}")
        print(f"  {'-'*12} {'-'*38} {'-'*12} {'-'*8}")
        for i in calendar["allotted_ipos"]:
            cutoff = f"NPR {i['cutoff_price']:.2f}" if i["cutoff_price"] else "  TBD"
            print(f"  {i['symbol']:<12} {i['company'][:38]:<38} "
                  f"{i['close_date'] or 'N/A':<12} {cutoff:>12}")
    else:
        print(f"  {YELLOW}No allotted IPOs found in last 60 days.{RESET}")

    section(f"Bonus / Right Share Announcements — last 30 days ({s['bonus_right_30d_count']})")
    if calendar["bonus_right_shares"]:
        print(f"  {'Symbol':<10} {'Bonus%':>6}  {'Cash%':>6}  {'Book Close':<12}  {'Listing':<12}")
        print(f"  {'-'*10} {'-'*6}  {'-'*6}  {'-'*12}  {'-'*12}")
        for i in calendar["bonus_right_shares"]:
            bonus = f"{i['bonus_pct']:.2f}" if i["bonus_pct"] else "  n/a"
            cash  = f"{i['cash_dividend_pct']:.2f}" if i["cash_dividend_pct"] else "  n/a"
            print(f"  {i['symbol']:<10} {bonus:>6}  {cash:>6}  "
                  f"{i['book_close_date'] or 'N/A':<12}  {i['bonus_listing_date'] or 'N/A':<12}")
    else:
        print(f"  {YELLOW}No bonus/right share announcements in last 30 days.{RESET}")

    section(f"SEBON Recent Documents — last 30 days ({s['sebon_docs_30d_count']})")
    for doc in calendar["sebon_recent_docs"]:
        print(f"  [{doc['date']}] [{doc['category']}] {doc['title']}")

    section(f"SEBON Bonus Share Registered — last 30 days ({s['sebon_bonus_30d_count']})")
    for doc in calendar["sebon_bonus_registered"]:
        print(f"  [{doc['date']}] {doc['title']}")

    print(f"\n  Saved -> {out_path}\n")
