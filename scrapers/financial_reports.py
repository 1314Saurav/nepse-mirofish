"""
scrapers/financial_reports.py

Scrapes quarterly financial reports for NEPSE-listed companies from
sharesansar.com/financial-analysis, downloads PDFs to
data/raw/financial_reports/{SYMBOL}/, and parses key financial metrics
with pdfplumber + regex.

SEBON report formatting note
-----------------------------
Nepali financial reports use comma-separated numbers but with a different
convention: the first group has 3 digits, then subsequent groups have 2
digits (South Asian numbering system).
  e.g.  1,23,45,678  = 12,345,678
The regex and parser handle both formats.

Output
------
  list[FinancialSummary]  — one entry per downloaded quarter.
  Saves to data/processed/financials/{SYMBOL}_financials.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
RAW_DIR    = _ROOT / "data" / "raw" / "financial_reports"
PROC_DIR   = _ROOT / "data" / "processed" / "financials"

for _d in (RAW_DIR, PROC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
_SS_BASE        = "https://www.sharesansar.com"
MAX_RETRIES     = 3
RETRY_DELAY     = 2
REQUEST_TIMEOUT = 25

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class FinancialSummary:
    symbol:            str
    quarter:           str            # e.g. "Q1 2081/82"
    period_end_date:   Optional[str]  = None
    scraped_at:        str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Income statement
    net_profit_loss_cr:      Optional[float] = None    # NPR crore
    operating_revenue_cr:    Optional[float] = None
    total_expenses_cr:       Optional[float] = None
    net_interest_income_cr:  Optional[float] = None    # banks / BFIs

    # Per-share
    eps:                     Optional[float] = None
    book_value_per_share:    Optional[float] = None

    # Ratios
    roe_pct:                 Optional[float] = None    # Return on Equity %
    roa_pct:                 Optional[float] = None    # Return on Assets %
    npl_ratio_pct:           Optional[float] = None    # NPL % (banks only)

    # Source
    pdf_url:                 Optional[str]   = None
    pdf_path:                Optional[str]   = None
    parse_warnings:          list[str]       = field(default_factory=list)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _session(warm_url: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(warm_url, timeout=REQUEST_TIMEOUT)
    except Exception:
        pass
    return s


def _fetch(session: requests.Session, url: str, **kwargs) -> Optional[requests.Response]:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    print(f"  [WARN] {url}: {last}", file=sys.stderr)
    return None


# ── Number parsing ─────────────────────────────────────────────────────────────

def _parse_nepali_number(text: str) -> Optional[float]:
    """
    Parse a number from Nepali financial report text.
    Handles:
      - South Asian comma format: 1,23,45,678
      - Standard format:          1,234,567
      - Negative in parentheses:  (1,23,456)
      - Lakhs / Crore labels
    Returns value in original units (caller converts to crore if needed).
    """
    t = text.strip()
    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    t = re.sub(r"[,\s]", "", t)
    try:
        val = float(t)
        return -val if negative else val
    except ValueError:
        return None


def _to_crore(val: Optional[float], unit_hint: str = "") -> Optional[float]:
    """Convert value to NPR crore based on unit hint in surrounding text."""
    if val is None:
        return None
    uh = unit_hint.lower()
    if "crore" in uh or "करोड" in uh:
        return round(val, 4)
    if "lakh" in uh or "lac" in uh or "लाख" in uh:
        return round(val / 100, 4)
    if "thousand" in uh or "हजार" in uh:
        return round(val / 10_000, 4)
    # Heuristic: if raw value > 1_000_000, probably in NPR units
    if abs(val) > 1_000_000:
        return round(val / 10_000_000, 4)   # NPR -> crore
    return round(val, 4)


# ── PDF scraping ───────────────────────────────────────────────────────────────

_PDF_PATTERNS: dict[str, list[str]] = {
    "net_profit_loss_cr": [
        r"net\s+profit\s*(?:/\s*loss)?\s*[:\-–]?\s*([\(\d,.\s]+)",
        r"खुद\s+मुनाफा\s*[:\-–]?\s*([\(\d,.\s]+)",
        r"profit\s+(?:after\s+tax|for\s+the\s+period)\s*[:\-–]?\s*([\(\d,.\s]+)",
    ],
    "eps": [
        r"earnings?\s+per\s+share\s*[:\-–]?\s*([\(\d,.]+)",
        r"\beps\b\s*[:\-–]?\s*([\(\d,.]+)",
        r"प्रति\s+सेयर\s+आय\s*[:\-–]?\s*([\(\d,.]+)",
    ],
    "net_interest_income_cr": [
        r"net\s+interest\s+income\s*[:\-–]?\s*([\(\d,.\s]+)",
        r"ब्याज\s+आम्दानी.{0,30}([\(\d,.\s]+)",
    ],
    "roe_pct": [
        r"return\s+on\s+(?:equity|shareholders)\s*[:\-–]?\s*([\d.]+)\s*%?",
        r"\broe\b\s*[:\-–]?\s*([\d.]+)",
    ],
    "roa_pct": [
        r"return\s+on\s+assets?\s*[:\-–]?\s*([\d.]+)\s*%?",
        r"\broa\b\s*[:\-–]?\s*([\d.]+)",
    ],
    "npl_ratio_pct": [
        r"non.performing\s+loan\s*[:\-–]?\s*([\d.]+)\s*%?",
        r"\bnpl\b.{0,15}([\d.]+)\s*%?",
        r"खराब\s+कर्जा.{0,20}([\d.]+)\s*%?",
    ],
    "operating_revenue_cr": [
        r"(?:total\s+)?operating\s+(?:income|revenue)\s*[:\-–]?\s*([\(\d,.\s]+)",
        r"operating\s+profit\s*[:\-–]?\s*([\(\d,.\s]+)",
    ],
    "total_expenses_cr": [
        r"total\s+(?:expenses?|expenditure)\s*[:\-–]?\s*([\(\d,.\s]+)",
        r"(?:operating\s+)?costs?\s*[:\-–]?\s*([\(\d,.\s]+)",
    ],
    "book_value_per_share": [
        r"book\s+value\s+per\s+share\s*[:\-–]?\s*([\d,.]+)",
        r"net\s+worth\s+per\s+share\s*[:\-–]?\s*([\d,.]+)",
    ],
}


def _parse_financials_from_text(text: str, symbol: str) -> dict:
    """Apply regex patterns to extracted PDF text; return parsed values."""
    results: dict[str, Optional[float]] = {k: None for k in _PDF_PATTERNS}
    warnings: list[str] = []
    text_lower = text.lower()

    # Detect unit context (look for unit declaration near table headers)
    unit_hint = ""
    m_unit = re.search(r"(?:amount\s+in|rs\.?\s+in|npr\s+in|figures?\s+in)\s+(\w+)", text_lower)
    if m_unit:
        unit_hint = m_unit.group(1)

    for field_name, patterns in _PDF_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text_lower, re.IGNORECASE | re.MULTILINE)
            if m:
                raw = m.group(1).strip()
                val = _parse_nepali_number(raw)
                if val is not None:
                    if field_name in ("eps", "roe_pct", "roa_pct", "npl_ratio_pct",
                                      "book_value_per_share"):
                        results[field_name] = round(val, 4)
                    else:
                        results[field_name] = _to_crore(val, unit_hint)
                    break
        else:
            warnings.append(f"{field_name}: not found in PDF text")

    return {"values": results, "warnings": warnings}


# ── sharesansar scraper ────────────────────────────────────────────────────────

def _ss_xsrf(session: requests.Session) -> str:
    return unquote(session.cookies.get("XSRF-TOKEN", ""))


def fetch_report_pdf_links(symbol: str, quarters: int = 4) -> list[dict]:
    """
    Scrape sharesansar.com/financial-analysis for PDF report links.

    Returns list of {"quarter": ..., "url": ..., "date": ...}
    """
    session = _session(f"{_SS_BASE}/financial-analysis")
    xsrf    = _ss_xsrf(session)

    # Try DataTables endpoint first
    resp = _fetch(
        session,
        f"{_SS_BASE}/financial-analysis",
        params={"symbol": symbol},
        headers={
            "Accept":             "application/json",
            "X-Requested-With":   "XMLHttpRequest",
            "X-XSRF-TOKEN":       xsrf,
            "Referer":            f"{_SS_BASE}/financial-analysis",
        },
    )

    pdf_entries: list[dict] = []

    # Try parsing as HTML page with links
    for attempt_url in [
        f"{_SS_BASE}/financial-analysis/{symbol.lower()}",
        f"{_SS_BASE}/company/{symbol.lower()}",
    ]:
        r = _fetch(session, attempt_url)
        if r is None:
            continue
        soup = BeautifulSoup(r.text, "lxml")

        # Look for links that say "Financial Report" or end in .pdf
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower() or "financial" in href.lower():
                if not href.startswith("http"):
                    href = urljoin(_SS_BASE, href)
                text = a.get_text(strip=True)
                # Try to find quarter label near the link
                parent_text = a.parent.get_text(" ", strip=True) if a.parent else ""
                quarter_m = re.search(
                    r"(Q[1-4]|First|Second|Third|Fourth|Annual).{0,20}(20[78]\d)",
                    parent_text, re.IGNORECASE
                )
                quarter = quarter_m.group(0) if quarter_m else text[:30]
                date_m = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", parent_text)
                date   = date_m.group(0) if date_m else ""
                pdf_entries.append({"quarter": quarter, "url": href, "date": date})

        if pdf_entries:
            break

    return pdf_entries[:quarters]


def _download_report_pdf(symbol: str, url: str, quarter: str) -> Optional[Path]:
    """Download a financial report PDF to RAW_DIR/{symbol}/."""
    sym_dir = RAW_DIR / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    safe_q  = re.sub(r"[^\w]", "_", quarter)[:40]
    path    = sym_dir / f"{symbol.upper()}_{safe_q}.pdf"
    if path.exists():
        return path
    r = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return None
    path.write_bytes(r.content)
    return path


def _extract_text_from_pdf(pdf_path: Path, max_pages: int = 10) -> str:
    """Extract text from a PDF (up to max_pages); cache as .txt."""
    txt_path = pdf_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="replace")
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages_text = []
            for pg in pdf.pages[:max_pages]:
                t = pg.extract_text() or ""
                pages_text.append(t)
            text = "\n".join(pages_text)
    except Exception as exc:
        text = f"[ERROR extracting PDF: {exc}]"
    txt_path.write_text(text, encoding="utf-8")
    return text


# ── Master function ────────────────────────────────────────────────────────────

def scrape_financial_reports(symbol: str, quarters: int = 4) -> list[FinancialSummary]:
    """
    Scrape + parse quarterly financial reports for `symbol`.

    Downloads PDFs and extracts financial metrics.
    Saves to data/processed/financials/{SYMBOL}_financials.json.
    """
    symbol = symbol.upper()
    print(f"  [{symbol}] Fetching report PDF links...", flush=True)
    links = fetch_report_pdf_links(symbol, quarters)

    if not links:
        print(f"  [{symbol}] No report PDFs found.", file=sys.stderr)
        return []

    summaries: list[FinancialSummary] = []
    for entry in links:
        pdf_url = entry["url"]
        quarter = entry.get("quarter", "unknown")
        date_s  = entry.get("date", "")

        print(f"  [{symbol}] Downloading: {quarter} -> {pdf_url[:80]}", flush=True)
        pdf_path = _download_report_pdf(symbol, pdf_url, quarter)
        if pdf_path is None:
            fs = FinancialSummary(symbol=symbol, quarter=quarter,
                                  period_end_date=date_s, pdf_url=pdf_url)
            fs.parse_warnings.append("PDF download failed")
            summaries.append(fs)
            continue

        print(f"  [{symbol}] Extracting text from {pdf_path.name}...", flush=True)
        text   = _extract_text_from_pdf(pdf_path)
        parsed = _parse_financials_from_text(text, symbol)
        vals   = parsed["values"]

        fs = FinancialSummary(
            symbol=symbol,
            quarter=quarter,
            period_end_date=date_s,
            pdf_url=pdf_url,
            pdf_path=str(pdf_path),
            net_profit_loss_cr=vals["net_profit_loss_cr"],
            operating_revenue_cr=vals["operating_revenue_cr"],
            total_expenses_cr=vals["total_expenses_cr"],
            net_interest_income_cr=vals["net_interest_income_cr"],
            eps=vals["eps"],
            book_value_per_share=vals["book_value_per_share"],
            roe_pct=vals["roe_pct"],
            roa_pct=vals["roa_pct"],
            npl_ratio_pct=vals["npl_ratio_pct"],
            parse_warnings=parsed["warnings"],
        )
        summaries.append(fs)

    # Save
    out_path = PROC_DIR / f"{symbol}_financials.json"
    out_path.write_text(
        json.dumps([asdict(s) for s in summaries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  [{symbol}] Saved -> {out_path}", flush=True)
    return summaries


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    import argparse
    parser = argparse.ArgumentParser(description="Scrape NEPSE financial reports")
    parser.add_argument("symbols", nargs="+", help="Stock symbols e.g. NABIL EBL")
    parser.add_argument("--quarters", type=int, default=4)
    args = parser.parse_args()

    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    RESET = "\033[0m"

    for sym in args.symbols:
        print(f"\n{BOLD}== {sym} Financial Reports =={RESET}")
        results = scrape_financial_reports(sym, args.quarters)
        for fs in results:
            print(f"\n  Quarter     : {fs.quarter}")
            print(f"  Period End  : {fs.period_end_date}")
            print(f"  Net Profit  : {fs.net_profit_loss_cr} Cr NPR")
            print(f"  EPS         : {fs.eps}")
            print(f"  ROE         : {fs.roe_pct} %")
            print(f"  ROA         : {fs.roa_pct} %")
            print(f"  NPL         : {fs.npl_ratio_pct} %")
            print(f"  NII         : {fs.net_interest_income_cr} Cr NPR")
            if fs.parse_warnings:
                print(f"  {CYAN}Warnings: {len(fs.parse_warnings)} fields not parsed{RESET}")
