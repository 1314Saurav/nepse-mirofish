"""
scrapers/nrb_policy.py

Scrapes Nepal Rastra Bank (nrb.org.np) for:
  - Latest monetary policy document (PDF link + date)
  - Interest rate corridor: bank rate, repo rate, reverse repo rate
  - CRR and SLR requirements
  - Latest BFI circulars (last 10) — title, date, url
  - Forex rates (USD, EUR, INR, CNY) — current + 30-day trend
  - Credit-to-deposit ratio

PDFs are downloaded to data/raw/nrb_pdfs/ and first 3 pages extracted
via pdfplumber, saved as <stem>_text.txt alongside the PDF.

Output
------
Returns an NrbSnapshot dict.
Saves to data/processed/nrb_policy_YYYY-MM-DD.json.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, unquote

import pdfplumber
import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR     = _ROOT / "data" / "raw"
PDF_DIR     = RAW_DIR / "nrb_pdfs"
PROC_DIR    = _ROOT / "data" / "processed"

for _d in (PDF_DIR, PROC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
NRB_BASE        = "https://www.nrb.org.np"
FOREX_API       = f"{NRB_BASE}/api/forex/v1/rates"
MAX_RETRIES     = 3
RETRY_DELAY     = 2
REQUEST_TIMEOUT = 20

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Low-level helpers ──────────────────────────────────────────────────────────

def _get(url: str, **kwargs) -> Optional[requests.Response]:
    """GET with retry; returns None on all failures."""
    kw = {"timeout": REQUEST_TIMEOUT, "headers": _HEADERS, **kwargs}
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, **kw)
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    print(f"  [WARN] GET {url} failed: {last}", file=sys.stderr)
    return None


def _num(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


# ── PDF helpers ────────────────────────────────────────────────────────────────

def _download_pdf(url: str, stem: str) -> Optional[Path]:
    """Download a PDF to PDF_DIR/<stem>.pdf; return path or None."""
    pdf_path = PDF_DIR / f"{stem}.pdf"
    if pdf_path.exists():
        return pdf_path
    resp = _get(url)
    if resp is None:
        return None
    pdf_path.write_bytes(resp.content)
    return pdf_path


def _extract_pdf_text(pdf_path: Path, pages: int = 3) -> str:
    """Extract first `pages` pages from a PDF; save as <stem>_text.txt."""
    txt_path = pdf_path.with_name(pdf_path.stem + "_text.txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="replace")
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            chunks = []
            for pg in pdf.pages[:pages]:
                text = pg.extract_text() or ""
                chunks.append(text)
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(chunks)
    except Exception as exc:
        full_text = f"[PDF extraction error: {exc}]"
    txt_path.write_text(full_text, encoding="utf-8")
    return full_text


# ── Forex rates ────────────────────────────────────────────────────────────────

_FOREX_TARGETS = {"USD", "EUR", "INR", "CNY"}

def fetch_forex_rates(days: int = 30) -> dict:
    """
    Fetch forex rates from the NRB JSON API for the last `days` days.

    Returns:
        {
          "current": {"USD": {"buy": ..., "sell": ...}, ...},
          "trend":   {"USD": [{"date": "...", "buy": ..., "sell": ...}, ...], ...}
        }
    """
    today = datetime.now(timezone.utc).date()
    from_date = (today - timedelta(days=days)).isoformat()
    to_date   = today.isoformat()

    resp = _get(
        FOREX_API,
        params={
            "page": 1,
            "per_page": days + 5,   # slight buffer
            "from": from_date,
            "to": to_date,
        },
    )
    if resp is None:
        return {"current": {}, "trend": {}}

    try:
        payload = resp.json()
    except Exception:
        return {"current": {}, "trend": {}}

    # NRB API: data.payload is a list of {date, rates: [{currency:{iso3,...}, buy, sell, unit}, ...]}
    entries = payload.get("data", {}).get("payload", [])

    trend: dict[str, list[dict]] = {sym: [] for sym in _FOREX_TARGETS}
    current: dict[str, dict] = {}

    for entry in sorted(entries, key=lambda e: e.get("date", ""), reverse=False):
        date_str = entry.get("date", "")
        for rate in entry.get("rates", []):
            iso3 = rate.get("currency", {}).get("iso3", "").upper()
            if iso3 not in _FOREX_TARGETS:
                continue
            unit = _num(rate.get("unit", 1)) or 1
            buy  = round((_num(rate.get("buy"))  or 0) / unit, 4)
            sell = round((_num(rate.get("sell")) or 0) / unit, 4)
            trend[iso3].append({"date": date_str, "buy": buy, "sell": sell})

    # Most recent entry = current
    for sym in _FOREX_TARGETS:
        if trend[sym]:
            current[sym] = trend[sym][-1]

    return {"current": current, "trend": trend}


# ── Monetary policy document ───────────────────────────────────────────────────

def fetch_monetary_policy_doc() -> dict:
    """
    Scrape the NRB monetary policy page for the latest PDF link and date.

    Returns:
        {"title": ..., "url": ..., "date": ..., "pdf_path": ..., "text_excerpt": ...}
    """
    candidates = [
        f"{NRB_BASE}/monetary-policy/",
        f"{NRB_BASE}/contents/monetary-policy/",
        f"{NRB_BASE}/publications/monetary-policy/",
    ]
    for url in candidates:
        resp = _get(url)
        if resp is None:
            continue

        # If it was served as a PDF directly
        ct = resp.headers.get("content-type", "")
        if "pdf" in ct:
            today_str = datetime.now().strftime("%Y-%m-%d")
            stem = f"monetary_policy_{today_str}"
            pdf_path = PDF_DIR / f"{stem}.pdf"
            pdf_path.write_bytes(resp.content)
            text = _extract_pdf_text(pdf_path)
            return {
                "title":        "Monetary Policy (direct PDF)",
                "url":          url,
                "date":         today_str,
                "pdf_path":     str(pdf_path),
                "text_excerpt": text[:2000],
            }

        soup = BeautifulSoup(resp.text, "lxml")
        # Look for PDF links in the page
        pdf_links = [
            a for a in soup.find_all("a", href=True)
            if a["href"].lower().endswith(".pdf")
        ]
        if not pdf_links:
            continue

        # Prefer links with "monetary" or "policy" in text/href
        preferred = [
            a for a in pdf_links
            if any(kw in (a.get_text() + a["href"]).lower()
                   for kw in ("monetary", "policy", "मौद्रिक"))
        ]
        link = (preferred or pdf_links)[0]
        href = link["href"]
        if not href.startswith("http"):
            href = urljoin(NRB_BASE, href)

        title = link.get_text(strip=True) or "Monetary Policy"
        # Try to find a date near the link
        date_str = _nearby_date(link) or datetime.now().strftime("%Y-%m-%d")

        stem     = re.sub(r"[^\w]", "_", title[:40].lower()).strip("_")
        pdf_path = _download_pdf(href, f"monetary_policy_{stem}")
        text     = _extract_pdf_text(pdf_path) if pdf_path else ""

        return {
            "title":        title,
            "url":          href,
            "date":         date_str,
            "pdf_path":     str(pdf_path) if pdf_path else None,
            "text_excerpt": text[:2000],
        }

    return {"title": None, "url": None, "date": None, "pdf_path": None, "text_excerpt": None}


def _nearby_date(tag) -> Optional[str]:
    """Try to find a date string near a BeautifulSoup tag."""
    text = ""
    for parent in (tag.parent, tag.parent.parent if tag.parent else None):
        if parent:
            text += parent.get_text(" ", strip=True)
    match = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", text)
    if match:
        return match.group(1).replace("/", "-")
    return None


# ── Interest rate corridor ─────────────────────────────────────────────────────

# Key rate pages to try
_RATE_PAGES = [
    f"{NRB_BASE}/",                              # main page has indicators sidebar
    f"{NRB_BASE}/category/monetary-operations/",
    f"{NRB_BASE}/monetary-policy/key-rates/",
    f"{NRB_BASE}/contents/monetary-policy/",
]

_RATE_LABELS = {
    "bank_rate":    ["bank rate", "bank-rate", "bankrate"],
    "repo_rate":    ["repo rate", "repurchase rate", "repo"],
    "reverse_repo": ["reverse repo", "reverse repurchase", "deposit collection"],
    "crr":          ["crr", "cash reserve ratio", "cash reserve"],
    "slr":          ["slr", "statutory liquidity ratio"],
}


def _parse_rates_from_html(html: str) -> dict:
    """Parse a rate table from NRB HTML; returns dict of found rates."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, Optional[float]] = {k: None for k in _RATE_LABELS}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = [td.get_text(strip=True).lower() for td in row.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            label_cell = cells[0]
            val_cell   = cells[-1]
            for key, keywords in _RATE_LABELS.items():
                if found[key] is not None:
                    continue
                if any(kw in label_cell for kw in keywords):
                    found[key] = _num(re.sub(r"[^\d.]", "", val_cell))

    # Also scan free text for patterns like "Bank Rate: 7.0%"
    for key, keywords in _RATE_LABELS.items():
        if found[key] is not None:
            continue
        for kw in keywords:
            m = re.search(
                rf"{re.escape(kw)}\s*[:\-–]\s*([\d.]+)\s*%",
                html, re.IGNORECASE,
            )
            if m:
                found[key] = _num(m.group(1))
                break

    return found


def fetch_interest_rate_corridor() -> dict:
    """
    Try multiple NRB pages to find bank rate, repo rate, reverse repo, CRR, SLR.

    Returns dict with those keys (values may be None if not found).
    """
    result: dict[str, Optional[float]] = {k: None for k in _RATE_LABELS}

    for url in _RATE_PAGES:
        resp = _get(url)
        if resp is None:
            continue
        ct = resp.headers.get("content-type", "")
        if "pdf" in ct:
            continue

        parsed = _parse_rates_from_html(resp.text)
        for key, val in parsed.items():
            if result[key] is None and val is not None:
                result[key] = val

        if all(v is not None for v in result.values()):
            break

    # Hard-coded fallback for known 2025/26 NRB policy rates
    # (updated when scraped values are not parseable)
    defaults = {
        "bank_rate":    7.0,
        "repo_rate":    6.5,
        "reverse_repo": 4.5,
        "crr":          4.0,
        "slr":          12.0,
    }
    for key, default in defaults.items():
        if result[key] is None:
            result[key] = default
            print(f"  [INFO] {key} not scraped; using policy default {default}", file=sys.stderr)

    return {
        "bank_rate":    result["bank_rate"],
        "repo_rate":    result["repo_rate"],
        "reverse_repo": result["reverse_repo"],
        "crr":          result["crr"],
        "slr":          result["slr"],
        "note":         "Values from live scrape where available; policy defaults otherwise.",
    }


# ── BFI Circulars ──────────────────────────────────────────────────────────────

_CIRCULAR_URLS = [
    f"{NRB_BASE}/category/circulars/?department=bfr",
    f"{NRB_BASE}/category/circulars/",
    f"{NRB_BASE}/bfrd/bfi-directives/",
    f"{NRB_BASE}/bfrd/notices-and-circulars/",
]


def fetch_bfi_circulars(limit: int = 10) -> list[dict]:
    """
    Scrape the NRB BFI circulars/directives page for the latest `limit` entries.

    NRB website is WordPress-based; circulars appear as article/post elements.
    Returns list of {"title": ..., "date": ..., "url": ...}
    """
    for url in _CIRCULAR_URLS:
        resp = _get(url)
        if resp is None:
            continue
        ct = resp.headers.get("content-type", "")
        if "pdf" in ct:
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        items: list[dict] = []

        # Strategy 1: NRB category pages use <ul><li><a>Title</a> date, size</li></ul>
        # Find the main content area to avoid nav <li> elements
        main = (
            soup.find("main") or
            soup.find("div", id=re.compile(r"content|main|primary", re.I)) or
            soup.find("div", class_=re.compile(r"content|main|primary|entry", re.I)) or
            soup
        )
        for li in main.find_all("li"):
            link_tag = li.find("a", href=True)
            if not link_tag:
                continue
            href  = link_tag["href"]
            if not href.startswith("http"):
                href = urljoin(NRB_BASE, href)
            # Skip navigation and category links
            if any(skip in href for skip in ["/category/", "/#", "/page/",
                                              "/about", "/bod", "/departments",
                                              "/provincial", "/information-officers",
                                              "/organogram", "/principal"]):
                continue
            if href.rstrip("/") in (NRB_BASE, NRB_BASE + "/"):
                continue
            title = link_tag.get_text(strip=True)
            if len(title) < 8:
                continue
            # Extract date from text nodes after the link
            li_text = li.get_text(" ", strip=True)
            date_m  = re.search(r"(20\d{2}[-/]\d{2}[-/]\d{2}|\d{1,2}[- ]\w+[- ]20\d{2})", li_text)
            date_str = date_m.group(1) if date_m else ""
            items.append({"title": title[:200], "date": date_str, "url": href})
            if len(items) >= limit:
                break

        # Strategy 2: table rows
        if not items:
            for table in soup.find_all("table"):
                for row in table.find_all("tr")[1:]:
                    cells    = row.find_all("td")
                    link_tag = row.find("a", href=True)
                    if not link_tag:
                        continue
                    href  = link_tag["href"]
                    if not href.startswith("http"):
                        href = urljoin(NRB_BASE, href)
                    title     = link_tag.get_text(strip=True)
                    date_text = cells[-1].get_text(strip=True) if cells else ""
                    if title and len(title) >= 8:
                        items.append({"title": title[:200], "date": date_text, "url": href})
                        if len(items) >= limit:
                            break
                if items:
                    break

        if items:
            return items[:limit]

    return []


# ── Credit-to-deposit ratio ────────────────────────────────────────────────────

_CD_URLS = [
    f"{NRB_BASE}/",                                              # sidebar indicators
    f"{NRB_BASE}/category/banking-and-financial-statistics/",
    f"{NRB_BASE}/statistics/",
]


def fetch_credit_deposit_ratio() -> dict:
    """
    Try to scrape the latest CD ratio from NRB pages.

    The NRB main page has a sidebar/widget "Indicators" section with CD ratio.
    Returns {"ratio": float|None, "as_of": str|None, "source_url": str|None}
    """
    for url in _CD_URLS:
        resp = _get(url)
        if resp is None:
            continue
        ct = resp.headers.get("content-type", "")
        if "pdf" in ct:
            continue

        html = resp.text
        soup = BeautifulSoup(html, "lxml")

        # Strategy 1: look for indicator widget/table with "CD Ratio" label
        for tag in soup.find_all(string=re.compile(r"cd\s*ratio|credit.{0,5}deposit", re.I)):
            parent = tag.parent
            if parent is None:
                continue
            # Value might be in next sibling td/span or a nearby element
            next_sib = parent.find_next_sibling()
            if next_sib:
                val = _num(re.sub(r"[^\d.]", "", next_sib.get_text()))
                if val is not None and 50 < val < 130:
                    return {
                        "ratio":      val,
                        "as_of":      datetime.now().strftime("%Y-%m-%d"),
                        "source_url": url,
                    }
            # Try same row td
            row = parent.find_parent("tr")
            if row:
                cells = row.find_all("td")
                for cell in cells:
                    val = _num(re.sub(r"[^\d.]", "", cell.get_text()))
                    if val is not None and 50 < val < 130:
                        return {
                            "ratio":      val,
                            "as_of":      datetime.now().strftime("%Y-%m-%d"),
                            "source_url": url,
                        }

        # Strategy 2: regex scan
        patterns = [
            r"cd\s*ratio\s*[:\-–]?\s*([\d.]+)\s*%?",
            r"credit.{0,10}deposit.{0,20}([\d.]+)\s*%",
        ]
        for pat in patterns:
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                val = _num(m.group(1))
                if val is not None and 50 < val < 130:
                    return {
                        "ratio":      val,
                        "as_of":      datetime.now().strftime("%Y-%m-%d"),
                        "source_url": url,
                    }

    return {
        "ratio":      None,
        "as_of":      None,
        "source_url": None,
        "note":       "CD ratio not found in scraped pages; check NRB stats manually.",
    }


# ── Master scraper ─────────────────────────────────────────────────────────────

def scrape_nrb_policy() -> dict:
    """
    Scrape all NRB policy data and return an NrbSnapshot dict.
    Saves to data/processed/nrb_policy_YYYY-MM-DD.json.
    """
    today     = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now(timezone.utc).isoformat()

    print("  Fetching forex rates (30-day)...", flush=True)
    forex = fetch_forex_rates(days=30)

    print("  Fetching monetary policy document...", flush=True)
    mp_doc = fetch_monetary_policy_doc()

    print("  Fetching interest rate corridor...", flush=True)
    rates = fetch_interest_rate_corridor()

    print("  Fetching BFI circulars...", flush=True)
    circulars = fetch_bfi_circulars(limit=10)

    print("  Fetching credit/deposit ratio...", flush=True)
    cd_ratio = fetch_credit_deposit_ratio()

    snapshot = {
        "timestamp_utc":     timestamp,
        "as_of_date":        today,
        "source":            "nrb.org.np",

        "interest_rates": {
            "bank_rate":    rates["bank_rate"],
            "repo_rate":    rates["repo_rate"],
            "reverse_repo": rates["reverse_repo"],
            "note":         rates.get("note"),
        },
        "reserve_requirements": {
            "crr_pct": rates["crr"],
            "slr_pct": rates["slr"],
        },
        "forex": forex,

        "monetary_policy_doc": mp_doc,
        "bfi_circulars":       circulars,
        "credit_deposit_ratio": cd_ratio,
    }

    out_path = PROC_DIR / f"nrb_policy_{today}.json"
    out_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved -> {out_path}", flush=True)
    return snapshot


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"

    print(f"\n{BOLD}Fetching NRB policy snapshot...{RESET}\n")
    snap = scrape_nrb_policy()

    def section(t: str) -> None:
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{BOLD}{CYAN}  {t}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")

    section("Interest Rate Corridor")
    ir = snap["interest_rates"]
    rr = snap["reserve_requirements"]
    print(f"  Bank Rate         : {ir['bank_rate']} %")
    print(f"  Repo Rate         : {ir['repo_rate']} %")
    print(f"  Reverse Repo Rate : {ir['reverse_repo']} %")
    print(f"  CRR               : {rr['crr_pct']} %")
    print(f"  SLR               : {rr['slr_pct']} %")

    section("Forex Rates (Current)")
    for sym, vals in snap["forex"].get("current", {}).items():
        print(f"  {sym}/NPR  buy={vals.get('buy'):.4f}  sell={vals.get('sell'):.4f}")

    section("Monetary Policy Document")
    mp = snap["monetary_policy_doc"]
    print(f"  Title  : {mp['title']}")
    print(f"  URL    : {mp['url']}")
    print(f"  Date   : {mp['date']}")
    if mp.get("pdf_path"):
        print(f"  PDF    : {mp['pdf_path']}")

    section("BFI Circulars (last 10)")
    for c in snap["bfi_circulars"]:
        print(f"  [{c['date'][:10]}]  {c['title'][:70]}")
        print(f"             {c['url']}")

    section("Credit / Deposit Ratio")
    cd = snap["credit_deposit_ratio"]
    val = cd.get("ratio")
    print(f"  CD Ratio : {val} %  (as of {cd.get('as_of')})")
    if cd.get("note"):
        print(f"  Note     : {cd['note']}")

    print(f"\n  Snapshot -> data/processed/nrb_policy_{snap['as_of_date']}.json\n")
