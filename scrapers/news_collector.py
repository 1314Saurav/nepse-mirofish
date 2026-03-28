"""
scrapers/news_collector.py

Collects NEPSE-relevant financial news from Nepali RSS feeds
and attempts to scrape public Twitter/X discourse via nitter.

Sources
-------
RSS feeds (user-specified, with verified fallbacks):
  Onlinekhabar_business  – onlinekhabar.com business/economy
  Ratopati_arthatantra   – ratopati.com economy (fallback: general feed)
  Karobar_daily          – karobardaily.com (Nepali financial daily)
  Nepal_monitor          – nepalmonitor.org (network dependent)
  Sharesansar_news       – sharesansar.com news (fallback: category page scrape)

Twitter/X via nitter:
  Public nitter instances are probed in order. All are currently unreliable
  (the nitter network has degraded significantly). The function returns an
  empty list with a status note if none are reachable.

Output  →  data/processed/news_YYYY-MM-DD.json
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_WORDS_BODY  = 800
MAX_RETRIES     = 2
RETRY_DELAY     = 1
REQUEST_TIMEOUT = 12

_UA_FEED    = "Mozilla/5.0 (compatible; feedparser/6; +https://github.com/nepse-mirofish)"
_UA_BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Primary and fallback RSS feed URLs
RSS_SOURCES: dict[str, list[str]] = {
    "Onlinekhabar_business": [
        "https://www.onlinekhabar.com/category/business/feed",
        "https://www.onlinekhabar.com/feed",            # fallback: all topics
    ],
    "Ratopati_arthatantra": [
        "https://www.ratopati.com/category/arthatantra/feed",
        "https://ratopati.com/feed",
        "https://www.ratopati.com/feed",
    ],
    "Karobar_daily": [
        "https://karobardaily.com/feed",                # 301 → followed automatically
    ],
    "Nepal_monitor": [
        "https://nepalmonitor.org/reports/rss",
    ],
    "Sharesansar_news": [
        "https://www.sharesansar.com/newsfeeds",
        "https://www.sharesansar.com/feed",
        "https://www.sharesansar.com/category/latest/feed",
    ],
}

# Per-domain CSS selectors for article body (tried in order)
BODY_SELECTORS: dict[str, list[str]] = {
    "onlinekhabar.com": ["article .entry-content", ".entry-content", "article"],
    "karobardaily.com": [".content-area", ".entry-content", "article", "main"],
    "ratopati.com":     [".entry-content", ".post-content", "article"],
    "setopati.com":     ["#content", ".entry-content", "article"],
    "sharesansar.com":  [".entry-content", ".post-content", ".td-post-content", "article"],
    "nepalmonitor.org": [".article-body", ".entry-content", "article"],
    "_default":         [
        "article .entry-content", ".entry-content", ".post-content",
        ".article-content", ".content-area", ".td-post-content",
        ".single-content", ".news-body", "article", "#content",
    ],
}

# Nitter public instances (tried in order)
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.poast.org",
    "https://nitter.tiekoetter.com",
    "https://nitter.catsarch.com",
    "https://nitter.woodland.cafe",
    "https://nitter.moomoo.me",
]

# Classification keyword map  (order matters: first match wins)
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("IPO", [
        "ipo", "fpo", "right share", "right issue", "book close", "allotment",
        "oversubscribed", "prospectus", "issue open", "issue close", "listing",
        "sebon approved",
    ]),
    ("policy/NRB", [
        "nrb", "nepal rastra bank", "monetary policy", "interest rate",
        "bank rate", "cash reserve", "crr", "slr", "inflation", "cpi",
        "foreign exchange", "remittance policy",
    ]),
    ("banking", [
        "bank", "nabil", "nbbl", "bok", "nic asia", "ncc", "sanima",
        "everest bank", "prabhu bank", "sbi", "kumari", "citizen", "mega",
        "sunrise bank", "global ime", "nepal investment", "deposit", "loan",
        "npa", "bad loan", "credit",
    ]),
    ("hydro/energy", [
        "hydropower", "hydro", "mw", "megawatt", "nea", "upper tamakoshi",
        "energy", "electricity", "power", "khola", "river", "dam", "ppa",
        "grid", "solar",
    ]),
    ("insurance", [
        "insurance", "nlic", "nlicl", "life insurance", "non-life",
        "general insurance", "beema", "premium", "claim",
    ]),
    ("manufacturing", [
        "manufacturing", "factory", "industry", "production", "cement",
        "steel", "textile", "pharmaceutical", "consumer goods",
    ]),
    ("general_market", [
        "nepse", "stock market", "share market", "turnover", "index",
        "share price", "listed", "market cap", "brokerage", "floorsheet",
        "dividend", "bonus share", "agm",
    ]),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _url_id(url: str) -> str:
    """Stable short ID for deduplication."""
    return hashlib.md5(url.strip().encode()).hexdigest()[:12]


def _fetch(url: str, session: Optional[requests.Session] = None,
           **kwargs) -> requests.Response:
    s = session or requests.Session()
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = s.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"GET {url} failed: {last}")


def _parse_pub_date(entry: feedparser.FeedParserDict) -> Optional[datetime]:
    """Parse feedparser entry date → timezone-aware UTC datetime."""
    for field in ("published", "updated", "created"):
        raw = entry.get(field)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
        tpl = entry.get(f"{field}_parsed")
        if tpl:
            try:
                return datetime(*tpl[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def _is_recent(dt: Optional[datetime], hours: int = 24) -> bool:
    if dt is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt >= cutoff


def _classify(title: str, body: str) -> str:
    text = (title + " " + body).lower()
    for category, keywords in CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            return category
    return "general_market"


def _extract_body(url: str, session: requests.Session) -> str:
    """Fetch article page and extract body text (first MAX_WORDS_BODY words)."""
    try:
        r = _fetch(url, session=session,
                   headers={"User-Agent": _UA_BROWSER,
                             "Accept": "text/html,*/*",
                             "Referer": url})
        soup = BeautifulSoup(r.text, "lxml")

        # Remove noise elements
        for tag in soup.find_all(["script", "style", "nav", "header",
                                   "footer", "aside", "noscript"]):
            tag.decompose()

        # Pick best selector set for this domain
        domain = re.sub(r"^www\.", "", re.sub(r"https?://", "", url).split("/")[0])
        selectors = BODY_SELECTORS.get(domain, BODY_SELECTORS["_default"])

        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                words = el.get_text(" ", strip=True).split()
                if len(words) >= 20:
                    return " ".join(words[:MAX_WORDS_BODY])

        # Last resort: largest <p> cluster
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        combined = " ".join(paras)
        words = combined.split()
        return " ".join(words[:MAX_WORDS_BODY]) if len(words) >= 20 else ""

    except Exception:
        return ""


# ── Feed scrapers ─────────────────────────────────────────────────────────────

def _parse_feed(source_name: str, urls: list[str]) -> list[dict]:
    """
    Try each URL in *urls* until one returns entries.
    Returns list of raw feedparser entries annotated with source_name and feed_url.
    """
    for url in urls:
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": _UA_FEED})
            if feed.entries:
                return [(e, source_name, url) for e in feed.entries]
        except Exception:
            continue
    return []


def scrape_rss_feeds(hours: int = 24) -> list[dict]:
    """
    Fetch all RSS feeds, filter to articles published within *hours* hours,
    fetch full article bodies, classify, and return deduplicated list.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": _UA_BROWSER})

    # ── Fetch all feeds concurrently ──────────────────────────────────────
    raw_entries: list[tuple] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_parse_feed, name, urls): name
            for name, urls in RSS_SOURCES.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                entries = future.result()
                raw_entries.extend(entries)
                print(f"  [{name}] {len(entries)} entries fetched", flush=True)
            except Exception as exc:
                print(f"  [{name}] ERROR: {exc}", flush=True)

    # ── Filter to recent, build articles ─────────────────────────────────
    seen_ids: set[str] = set()
    articles: list[dict] = []

    def _build(args: tuple) -> Optional[dict]:
        entry, source_name, feed_url = args
        url = entry.get("link", "")
        if not url:
            return None
        uid = _url_id(url)
        if uid in seen_ids:
            return None

        pub_dt = _parse_pub_date(entry)
        if not _is_recent(pub_dt, hours):
            return None

        title   = entry.get("title", "").strip()
        summary = BeautifulSoup(entry.get("summary", ""), "lxml").get_text(" ", strip=True)
        body    = _extract_body(url, session) or summary
        words   = body.split()[:MAX_WORDS_BODY]

        return {
            "_uid":       uid,
            "source":     source_name,
            "title":      title,
            "url":        url,
            "published_at": pub_dt.isoformat() if pub_dt else None,
            "body_words": len(words),
            "body":       " ".join(words),
            "category":   _classify(title, " ".join(words)),
        }

    # Fetch bodies concurrently
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures2 = [pool.submit(_build, args) for args in raw_entries]
        for future in as_completed(futures2):
            try:
                art = future.result()
                if art and art["_uid"] not in seen_ids:
                    seen_ids.add(art["_uid"])
                    articles.append(art)
            except Exception:
                pass

    # Remove internal dedup key
    for art in articles:
        art.pop("_uid", None)

    articles.sort(key=lambda a: a.get("published_at") or "", reverse=True)
    return articles


# ── Twitter/Nitter scraper ────────────────────────────────────────────────────

def scrape_nepse_tweets(query: str = "#NEPSE OR #Nepal_stocks",
                        max_tweets: int = 50) -> dict:
    """
    Scrape recent tweets matching *query* from public nitter instances.

    Returns:
        {"status": "ok"|"unavailable", "note": str, "tweets": list[dict]}

    NOTE: Public nitter instances are currently unreliable.  All known public
    mirrors have degraded availability.  This function probes each instance and
    returns an empty list with status="unavailable" if none respond with
    parseable tweet content.  The architecture is in place to activate
    automatically when a working instance is detected.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": _UA_BROWSER})

    encoded = requests.utils.quote(query)

    for base in NITTER_INSTANCES:
        url = f"{base}/search?q={encoded}&f=tweets"
        try:
            r = s.get(url, timeout=8, allow_redirects=True)
            if r.status_code != 200 or len(r.text) < 500:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            # nitter tweet selectors
            tweet_cards = (soup.find_all(class_="tweet-card") or
                           soup.find_all(class_="tweet-content") or
                           soup.find_all(attrs={"data-tweet-id": True}))

            if not tweet_cards:
                continue

            tweets: list[dict] = []
            for card in tweet_cards[:max_tweets]:
                text_el   = (card.find(class_="tweet-content") or
                             card.find(class_="content") or card)
                text      = text_el.get_text(" ", strip=True)
                date_el   = card.find(class_="tweet-date") or card.find("a", class_="date")
                date_str  = date_el.get_text(strip=True) if date_el else ""
                user_el   = (card.find(class_="username") or
                             card.find(class_="fullname"))
                username  = user_el.get_text(strip=True) if user_el else ""
                link_el   = card.find("a", href=re.compile(r"/status/"))
                tweet_url = (f"{base}{link_el['href']}"
                             if link_el and link_el.get("href") else "")

                if text:
                    tweets.append({
                        "username":  username,
                        "text":      text[:280],
                        "date":      date_str,
                        "tweet_url": tweet_url,
                    })

            if tweets:
                return {
                    "status":   "ok",
                    "note":     f"scraped from {base}",
                    "instance": base,
                    "query":    query,
                    "tweets":   tweets,
                }

        except Exception:
            continue

    return {
        "status":  "unavailable",
        "note": (
            "All probed nitter instances returned empty/unreachable responses. "
            "Public nitter mirrors have degraded significantly in 2025-2026. "
            "Update NITTER_INSTANCES in this file with a working instance, "
            "or use the official Twitter/X API (requires credentials)."
        ),
        "query":   query,
        "tweets":  [],
    }


# ── Master collector ──────────────────────────────────────────────────────────

def collect_news(hours: int = 24) -> dict:
    """
    Collect all news and social data. Save to processed/news_YYYY-MM-DD.json.
    Returns the full payload dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"  Fetching RSS feeds (last {hours}h)...", flush=True)
    articles = scrape_rss_feeds(hours=hours)

    print(f"  Probing nitter for NEPSE tweets...", flush=True)
    twitter  = scrape_nepse_tweets()

    # Count per category
    from collections import Counter
    cat_counts = dict(Counter(a["category"] for a in articles))

    payload = {
        "timestamp_utc":  timestamp,
        "as_of_date":     today,
        "window_hours":   hours,
        "article_count":  len(articles),
        "per_category":   cat_counts,
        "twitter":        twitter,
        "articles":       articles,
    }

    out_path = PROCESSED_DIR / f"news_{today}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload, out_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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

    print(f"\n{BOLD}Collecting NEPSE news (last 24 hours)...{RESET}\n")

    try:
        payload, out_path = collect_news(hours=24)
    except Exception as exc:
        print(f"{RED}ERROR: {exc}{RESET}")
        sys.exit(1)

    section("Article Count by Category")
    for cat, count in sorted(payload["per_category"].items(),
                              key=lambda x: -x[1]):
        bar = GREEN + "#" * min(count, 40) + RESET
        print(f"  {cat:<20} {count:>3}  {bar}")

    section(f"Articles ({payload['article_count']} total)")
    for art in payload["articles"]:
        pub = art.get("published_at", "")[:16]
        cat = art.get("category", "")
        src = art.get("source", "")
        title = art.get("title", "")
        print(f"  [{pub}] {CYAN}[{cat:<16}]{RESET} {YELLOW}[{src}]{RESET}")
        print(f"    {title[:75]}")

    section("Twitter / X (#NEPSE)")
    tw = payload["twitter"]
    if tw["status"] == "ok":
        print(f"  {GREEN}Status: {tw['status']}  via {tw['instance']}{RESET}")
        for t in tw["tweets"][:10]:
            print(f"  @{t['username']}  [{t['date']}]")
            print(f"    {t['text'][:100]}")
    else:
        print(f"  {YELLOW}Status: {tw['status']}{RESET}")
        print(f"  {tw['note']}")

    print(f"\n  Saved -> {out_path}\n")
