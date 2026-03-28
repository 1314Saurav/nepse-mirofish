"""
pipeline/event_calendar.py
NEPSE market event calendar.
Tracks NRB monetary policy, Nepal government budget, IPO listings,
book close dates, and other high-impact market events.

Sources (when scrapers are available):
  - sharesansar.com  — book close / dividend events
  - sebon.gov.np     — IPO / FPO announcements
  - nrb.org.np       — monetary policy calendar
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib import request as urllib_request
from urllib.error import URLError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static high-impact event definitions
# (updated manually; scraper auto-populates dynamic events)
# ---------------------------------------------------------------------------

# Nepal fiscal year starts mid-July; budget is typically presented in May/June
HIGH_IMPACT_EVENTS: dict[str, dict] = {
    "nrb_monetary_policy": {
        "name": "NRB Monetary Policy Announcement",
        "impact": "HIGH",
        "description": "Nepal Rastra Bank sets repo rate, CRR, SLR; moves banking sector",
        "typical_months": [7, 1],        # mid-July (start FY) + mid-January (mid-term review)
        "days_caution_before": 3,
        "days_caution_after": 2,
        "affected_sectors": ["banking", "finance", "insurance"],
    },
    "government_budget": {
        "name": "Nepal Government Budget",
        "impact": "HIGH",
        "description": "Annual budget presentation in parliament; major market mover",
        "typical_months": [5, 6],
        "days_caution_before": 5,
        "days_caution_after": 3,
        "affected_sectors": ["all"],
    },
    "book_close": {
        "name": "Book Close (Dividend)",
        "impact": "MEDIUM",
        "description": "Record date for dividend eligibility; sharp price drop after",
        "days_caution_before": 2,
        "days_caution_after": 1,
        "affected_sectors": ["symbol_specific"],
    },
    "ipo_listing": {
        "name": "IPO / FPO Listing",
        "impact": "MEDIUM",
        "description": "New share listing on NEPSE; affects sector sentiment",
        "days_caution_before": 1,
        "days_caution_after": 1,
        "affected_sectors": ["sector_specific"],
    },
    "quarterly_results": {
        "name": "Corporate Quarterly Results",
        "impact": "MEDIUM",
        "description": "Q1/Q2/Q3/Q4 financials; especially banking sector AGMs",
        "typical_months": [10, 1, 4, 7],
        "days_caution_before": 1,
        "days_caution_after": 1,
        "affected_sectors": ["banking"],
    },
    "nepse_holiday": {
        "name": "NEPSE Market Holiday",
        "impact": "LOW",
        "description": "Dashain, Tihar, Holi, public holidays — market closed",
        "days_caution_before": 0,
        "days_caution_after": 0,
        "affected_sectors": ["all"],
    },
    "rights_share": {
        "name": "Rights Share Offering",
        "impact": "MEDIUM",
        "description": "Dilutive rights issue; affects ex-rights price",
        "days_caution_before": 2,
        "days_caution_after": 2,
        "affected_sectors": ["symbol_specific"],
    },
}

# ---------------------------------------------------------------------------
# Hard-coded known events for current cycle (update as needed)
# ---------------------------------------------------------------------------

KNOWN_EVENTS_2025_2026: list[dict] = [
    {
        "name": "NRB Mid-Term Monetary Policy Review",
        "date": "2026-01-15",
        "impact": "HIGH",
        "event_type": "nrb_monetary_policy",
        "affected_sectors": ["banking", "finance"],
        "source": "static",
    },
    {
        "name": "Nepal Government Budget FY 2082/83",
        "date": "2026-05-29",
        "impact": "HIGH",
        "event_type": "government_budget",
        "affected_sectors": ["all"],
        "source": "static",
    },
    {
        "name": "NRB Monetary Policy FY 2082/83",
        "date": "2026-07-16",
        "impact": "HIGH",
        "event_type": "nrb_monetary_policy",
        "affected_sectors": ["banking", "finance", "insurance"],
        "source": "static",
    },
]


# ---------------------------------------------------------------------------
# Dynamic scraper helpers
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 8) -> Optional[str]:
    """Simple HTTP GET; returns text or None on failure."""
    try:
        req = urllib_request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (MiroFish/1.0; NEPSE calendar)"},
        )
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (URLError, Exception) as exc:
        logger.debug("Fetch failed for %s: %s", url, exc)
        return None


def _scrape_sharesansar_book_close() -> list[dict]:
    """
    Attempt to scrape upcoming book close dates from sharesansar.com.
    Returns empty list on failure (graceful degradation).
    """
    events: list[dict] = []
    html = _fetch_url("https://www.sharesansar.com/book-close")
    if not html:
        return events

    # Very simple regex scrape — site structure may change
    # Pattern: symbol, book close date in table rows
    pattern = re.compile(
        r'<td[^>]*>\s*([A-Z]{2,10})\s*</td>.*?'
        r'<td[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*</td>',
        re.DOTALL,
    )
    for match in pattern.finditer(html):
        symbol, close_date = match.group(1), match.group(2)
        try:
            d = date.fromisoformat(close_date)
            if d >= date.today():
                events.append({
                    "name": f"Book Close: {symbol}",
                    "date": close_date,
                    "impact": "MEDIUM",
                    "event_type": "book_close",
                    "affected_sectors": ["symbol_specific"],
                    "symbol": symbol,
                    "source": "sharesansar",
                })
        except ValueError:
            pass

    logger.info("Scraped %d book close events from sharesansar", len(events))
    return events


def _scrape_sebon_ipos() -> list[dict]:
    """
    Attempt to scrape upcoming IPO/FPO listings from sebon.gov.np.
    Returns empty list on failure.
    """
    events: list[dict] = []
    html = _fetch_url("https://www.sebon.gov.np/prospectus")
    if not html:
        return events

    # Simple scrape for company names and dates
    pattern = re.compile(
        r'([A-Za-z\s]+(?:Ltd|Limited|Bank|Finance|Insurance))'
        r'.*?(\d{4}-\d{2}-\d{2})',
        re.DOTALL,
    )
    seen: set[str] = set()
    for match in pattern.finditer(html[:50000]):
        name = match.group(1).strip()[:40]
        list_date = match.group(2)
        key = f"{name}:{list_date}"
        if key in seen:
            continue
        seen.add(key)
        try:
            d = date.fromisoformat(list_date)
            if d >= date.today():
                events.append({
                    "name": f"IPO Listing: {name}",
                    "date": list_date,
                    "impact": "MEDIUM",
                    "event_type": "ipo_listing",
                    "affected_sectors": ["sector_specific"],
                    "source": "sebon",
                })
        except ValueError:
            pass

    logger.info("Scraped %d IPO events from sebon.gov.np", len(events))
    return events[:10]  # cap


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def get_upcoming_events(
    days_ahead: int = 30,
    include_scraped: bool = True,
) -> list[dict]:
    """
    Return events occurring within the next `days_ahead` days.
    Merges static + scraped events; deduplicates; sorts by date.

    Parameters
    ----------
    days_ahead      : how many calendar days to look ahead
    include_scraped : whether to attempt live scraping (requires internet)
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)

    # Start with static known events
    all_events: list[dict] = list(KNOWN_EVENTS_2025_2026)

    # Try scrapers (silently ignore failures)
    if include_scraped:
        try:
            all_events.extend(_scrape_sharesansar_book_close())
        except Exception as exc:
            logger.debug("sharesansar scrape skipped: %s", exc)
        try:
            all_events.extend(_scrape_sebon_ipos())
        except Exception as exc:
            logger.debug("sebon scrape skipped: %s", exc)

    # Filter to window
    upcoming: list[dict] = []
    seen_keys: set[str] = set()
    for ev in all_events:
        try:
            ev_date = date.fromisoformat(str(ev.get("date", "")))
        except ValueError:
            continue
        if today <= ev_date <= cutoff:
            key = f"{ev.get('event_type','')}:{ev_date}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            days_away = (ev_date - today).days
            upcoming.append({
                **ev,
                "date": ev_date.isoformat(),
                "days_away": days_away,
            })

    upcoming.sort(key=lambda e: e["date"])
    return upcoming


def get_events_for_date(target_date: Optional[date] = None) -> list[dict]:
    """Return events on or within caution window of a specific date."""
    target_date = target_date or date.today()
    # Look back/ahead 5 days to catch caution windows
    window_events = get_upcoming_events(days_ahead=10)
    relevant: list[dict] = []
    for ev in window_events:
        ev_date = date.fromisoformat(ev["date"])
        event_type = ev.get("event_type", "")
        template = HIGH_IMPACT_EVENTS.get(event_type, {})
        caution_before = template.get("days_caution_before", 1)
        caution_after = template.get("days_caution_after", 1)
        window_start = ev_date - timedelta(days=caution_before)
        window_end = ev_date + timedelta(days=caution_after)
        if window_start <= target_date <= window_end:
            relevant.append(ev)
    return relevant


def adjust_signal_for_events(
    signal: dict,
    events: Optional[list[dict]] = None,
    target_date: Optional[date] = None,
) -> dict:
    """
    Dampen or block trading signals when high-impact events are imminent.

    Parameters
    ----------
    signal      : composite signal dict from signal_combiner.combine_signals()
    events      : list of events (if None, auto-fetches for today)
    target_date : date to check (default: today)

    Returns
    -------
    Modified signal dict with `event_adjusted=True` and adjusted position_size_pct.
    """
    if events is None:
        events = get_events_for_date(target_date)

    if not events:
        return {**signal, "event_adjusted": False}

    # Find highest-impact event
    has_high = any(e.get("impact") == "HIGH" for e in events)
    has_medium = any(e.get("impact") == "MEDIUM" for e in events)

    adjustments: list[str] = []
    adjusted = dict(signal)

    if has_high:
        # Reduce position size by 50% and force WATCH
        orig_size = adjusted.get("position_size_pct", 10.0)
        adjusted["position_size_pct"] = round(orig_size * 0.5, 2)
        if adjusted.get("action") == "BUY":
            adjusted["action"] = "WATCH"
            adjustments.append("HIGH-impact event: action downgraded BUY→WATCH")
        adjustments.append(
            f"HIGH-impact event: position size halved → {adjusted['position_size_pct']:.1f}%"
        )
    elif has_medium:
        # Reduce position size by 25%
        orig_size = adjusted.get("position_size_pct", 10.0)
        adjusted["position_size_pct"] = round(orig_size * 0.75, 2)
        adjustments.append(
            f"MEDIUM-impact event: position size reduced → {adjusted['position_size_pct']:.1f}%"
        )

    event_names = [e.get("name", "") for e in events[:3]]
    adjusted["event_adjusted"] = True
    adjusted["event_adjustment_reasons"] = adjustments
    adjusted["nearby_events"] = event_names

    return adjusted


def save_event_cache(output_dir: str = "data/processed") -> str:
    """Fetch and cache upcoming events to JSON. Returns path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    events = get_upcoming_events(days_ahead=60)
    today_str = date.today().isoformat()
    path = Path(output_dir) / f"events_{today_str}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"fetched": today_str, "events": events}, fh, indent=2)
    logger.info("Event cache saved → %s  (%d events)", path, len(events))
    return str(path)


def format_events_terminal(events: list[dict]) -> str:
    """Return a simple text table of events."""
    if not events:
        return "No upcoming events.\n"
    lines = [f"\n{'='*55}", "  📅  UPCOMING MARKET EVENTS", f"{'='*55}"]
    for ev in events:
        impact_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
            ev.get("impact", "LOW"), "⚪"
        )
        lines.append(
            f"  {impact_icon} {ev.get('date',''):<12}  {ev.get('name',''):<30}"
            f"  ({ev.get('days_away', '?')}d)"
        )
    lines.append(f"{'='*55}\n")
    return "\n".join(lines)
