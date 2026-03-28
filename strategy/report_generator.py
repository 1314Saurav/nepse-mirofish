"""
strategy/report_generator.py
Weekly strategy report generator using Anthropic Claude API.
Aggregates the week's signals, portfolio performance, and MiroFish insights
into a structured markdown report, then optionally sends via Telegram.

Schedule: Friday 18:00 NST (Nepal Standard Time = UTC+5:45)
Run:  python -m strategy.report_generator
      make weekly-report
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data gatherers
# ---------------------------------------------------------------------------

def _gather_week_data(week_ending: date) -> dict:
    """Collect all data needed for the weekly report."""
    week_start = week_ending - timedelta(days=6)
    data: dict = {
        "week_start": week_start.isoformat(),
        "week_ending": week_ending.isoformat(),
        "regime": "UNKNOWN",
        "regime_history": [],
        "portfolio_summary": {},
        "weekly_trades": [],
        "watchlist_snapshot": [],
        "sector_rotation": [],
        "top_signals": [],
        "events_next_week": [],
    }

    # Regime
    try:
        from strategy.regime_detector import detect_regime
        regime_result = detect_regime()
        data["regime"] = regime_result.get("regime", "UNKNOWN") if isinstance(regime_result, dict) else str(regime_result)
        data["regime_confidence"] = regime_result.get("confidence", 0.5) if isinstance(regime_result, dict) else 0.5
    except Exception as exc:
        logger.warning("Could not load regime: %s", exc)

    # Portfolio
    try:
        from strategy.portfolio import load_portfolio
        p = load_portfolio()
        data["portfolio_summary"] = p.get_summary()
        data["weekly_trades"] = [
            t for t in p.get_trade_log()
            if t.get("trade_date", "") >= week_start.isoformat()
        ]
    except Exception as exc:
        logger.warning("Could not load portfolio: %s", exc)

    # Watchlist (latest saved)
    try:
        from strategy.watchlist import load_latest_watchlist
        data["watchlist_snapshot"] = load_latest_watchlist()[:10]
    except Exception as exc:
        logger.warning("Could not load watchlist: %s", exc)

    # Sector rotation (latest saved)
    try:
        from strategy.sector_rotation import get_rotation_signal
        sr = get_rotation_signal(save=False)
        data["sector_rotation"] = sr.get("ranked_sectors", [])[:6]
    except Exception as exc:
        logger.warning("Could not load sector rotation: %s", exc)

    # Upcoming events
    try:
        from pipeline.event_calendar import get_upcoming_events
        data["events_next_week"] = get_upcoming_events(days_ahead=7)
    except Exception as exc:
        logger.warning("Could not load events: %s", exc)

    return data


def _build_prompt(data: dict) -> str:
    """Construct the LLM prompt for report generation."""
    portfolio = data.get("portfolio_summary", {})
    trades = data.get("weekly_trades", [])
    watchlist = data.get("watchlist_snapshot", [])
    sectors = data.get("sector_rotation", [])
    events = data.get("events_next_week", [])

    trade_lines = ""
    for t in trades[:10]:
        pnl = t.get("pnl_pct", 0)
        trade_lines += (
            f"  - {t.get('action','?')} {t.get('symbol','')} "
            f"@ NPR {t.get('price',0):.2f} × {t.get('shares',0)} shares"
        )
        if t.get("action") == "SELL":
            trade_lines += f" | PnL: {pnl:+.2f}% | Reason: {t.get('exit_reason','')}"
        trade_lines += "\n"

    watchlist_lines = ""
    for w in watchlist[:8]:
        watchlist_lines += (
            f"  - {w.get('symbol',''):<8} Score={w.get('score',0):.0f} "
            f"Tier={w.get('tier','')} RSI={w.get('rsi',50):.0f} "
            f"Vol×={w.get('vol_ratio',1):.1f} MF={w.get('mirofish_score',0):+.2f}\n"
        )

    sector_lines = "\n".join(
        f"  {i+1}. {s.get('sector',''):<15} score={s.get('combined_score',0):.3f}"
        for i, s in enumerate(sectors)
    )

    event_lines = "\n".join(
        f"  - {e.get('date','')}: {e.get('name','')} [{e.get('impact','')}]"
        for e in events[:5]
    ) or "  None"

    return f"""You are a senior NEPSE (Nepal Stock Exchange) quantitative analyst and portfolio manager.

Generate a comprehensive weekly strategy report for the week of {data['week_start']} to {data['week_ending']}.

## INPUT DATA

**Market Regime:** {data['regime']} (confidence: {data.get('regime_confidence', 0.5):.0%})

**Portfolio Performance:**
- Initial Cash: NPR {portfolio.get('initial_cash', 500000):,.0f}
- Portfolio Value: NPR {portfolio.get('total_portfolio_value', 500000):,.0f}
- Total PnL: {portfolio.get('total_pnl_pct', 0):+.2f}%
- Realised PnL: NPR {portfolio.get('realised_pnl_npr', 0):,.0f}
- Open Positions: {portfolio.get('open_positions_count', 0)}
- Win Rate: {portfolio.get('win_rate', 0):.1f}%
- Total Trades: {portfolio.get('total_trades', 0)}

**Weekly Trades:**
{trade_lines or '  No trades this week'}

**Top Watchlist (End of Week):**
{watchlist_lines or '  No data'}

**Sector Rotation Ranking:**
{sector_lines or '  No data'}

**Upcoming Events (Next 7 Days):**
{event_lines}

## REPORT FORMAT

Please generate a report with these exact sections:

### 1. Weekly Market Summary (3-4 sentences)
Summarise the week's market regime, major drivers, and overall sentiment.

### 2. Portfolio Performance Review
Analyse the week's trades and portfolio metrics. What worked? What didn't?
Include specific commentary on each closed trade's exit quality.

### 3. Regime Analysis & Outlook
Interpret the current regime ({data['regime']}) for next week.
Probability of regime change? Key triggers to watch.

### 4. Top Sector Opportunities
Based on the sector rotation ranking, highlight 2-3 sectors with the best
risk/reward for next week. Name specific stocks to watch within those sectors.

### 5. Risk Factors & Event Calendar
Discuss the upcoming events and their potential market impact.
Include any global/regional macro risks relevant to Nepal markets.

### 6. Strategy Recommendations for Next Week
Concrete, actionable recommendations:
- Which Tier-A watchlist stocks to prioritise (max 3)
- Position sizing guidance given current regime
- Key entry/exit levels to watch
- Any open positions that should be reviewed

### 7. One-Sentence Market Call
End with a single bold sentence summarising your overall market stance for next week.

Constraints:
- Be specific to NEPSE context (T+3 settlement, circuit breakers, low liquidity)
- Use NPR for all currency figures
- Keep total length under 800 words
- Be direct and actionable, not generic
"""


# ---------------------------------------------------------------------------
# Claude API caller
# ---------------------------------------------------------------------------

def _call_claude_api(prompt: str) -> Optional[str]:
    """Call Anthropic Claude API and return the response text."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your_") or api_key.startswith("sk-ant-api03-REPLACE"):
        logger.warning("ANTHROPIC_API_KEY not configured")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text if message.content else None
    except ImportError:
        logger.warning("anthropic package not installed; trying requests fallback")
    except Exception as exc:
        logger.error("Claude API error: %s", exc)
        return None

    # Fallback: raw HTTP via requests / urllib
    try:
        import json as _json
        import urllib.request as _req

        payload = _json.dumps({
            "model": "claude-opus-4-5",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        http_req = _req.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with _req.urlopen(http_req, timeout=60) as resp:
            result = _json.loads(resp.read())
            return result["content"][0]["text"]
    except Exception as exc:
        logger.error("Claude API fallback error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def generate_weekly_report(
    week_ending: Optional[date] = None,
    save: bool = True,
    send_telegram: bool = True,
    output_dir: str = "data/reports",
) -> Optional[str]:
    """
    Generate the weekly NEPSE strategy report.

    Parameters
    ----------
    week_ending    : date of the last trading day of the week (default: today)
    save           : persist the report to a markdown file
    send_telegram  : send a summary to Telegram if configured
    output_dir     : directory for saved reports

    Returns
    -------
    The report text (markdown string) or None on failure.
    """
    week_ending = week_ending or date.today()
    logger.info("Generating weekly report for week ending %s", week_ending)

    # 1. Gather data
    data = _gather_week_data(week_ending)

    # 2. Build prompt
    prompt = _build_prompt(data)

    # 3. Call Claude
    report_text = _call_claude_api(prompt)

    if not report_text:
        # Fallback: generate a basic template report without LLM
        logger.warning("LLM unavailable — generating template report")
        report_text = _fallback_report(data)

    # 4. Wrap with metadata header
    full_report = f"""# MiroFish Weekly NEPSE Report
**Week:** {data['week_start']} → {data['week_ending']}
**Regime:** {data['regime']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M NST')}

---

{report_text}

---
*Generated by MiroFish strategy engine with Claude AI analysis*
"""

    # 5. Save
    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f"weekly_report_{data['week_ending']}.md"
        path = Path(output_dir) / filename
        path.write_text(full_report, encoding="utf-8")
        logger.info("Report saved → %s", path)
        print(f"Report saved → {path}")

    # 6. Telegram summary
    if send_telegram:
        _send_telegram_summary(data, report_text)

    # 7. Print to terminal
    print(full_report)
    return full_report


def _fallback_report(data: dict) -> str:
    """Minimal report when LLM is unavailable."""
    portfolio = data.get("portfolio_summary", {})
    trades = data.get("weekly_trades", [])
    sectors = data.get("sector_rotation", [])
    events = data.get("events_next_week", [])

    top_sector = sectors[0].get("sector", "N/A") if sectors else "N/A"
    trade_count = len(trades)
    pnl = portfolio.get("total_pnl_pct", 0)
    win_rate = portfolio.get("win_rate", 0)

    return f"""### 1. Weekly Market Summary
Market regime for the week was **{data['regime']}**. Full AI analysis unavailable — LLM API not configured.

### 2. Portfolio Performance Review
- Total PnL: **{pnl:+.2f}%**
- Trades executed: {trade_count}
- Win rate: {win_rate:.1f}%

### 3. Regime Analysis & Outlook
Current regime: **{data['regime']}**. Monitor for regime shift triggers.

### 4. Top Sector Opportunities
Leading sector by momentum: **{top_sector}**. Review sector rotation rankings for full list.

### 5. Risk Factors & Event Calendar
Upcoming events: {len(events)} event(s) in next 7 days. Review event_calendar for details.

### 6. Strategy Recommendations for Next Week
Review Tier-A watchlist stocks. Maintain position sizing per current regime guidelines.

### 7. One-Sentence Market Call
**No AI market call available — configure ANTHROPIC_API_KEY for full report generation.**
"""


def _send_telegram_summary(data: dict, report_text: str) -> None:
    """Send a condensed Telegram summary of the weekly report."""
    try:
        from strategy.watchlist import send_telegram
    except ImportError:
        return

    portfolio = data.get("portfolio_summary", {})
    pnl = portfolio.get("total_pnl_pct", 0)
    pnl_icon = "📈" if pnl >= 0 else "📉"

    # Extract the market call (last bold sentence) from the report
    import re
    bold_sentences = re.findall(r'\*\*(.+?)\*\*', report_text)
    market_call = bold_sentences[-1] if bold_sentences else "See full report"

    msg = (
        f"📊 *MiroFish Weekly Report* — {data['week_ending']}\n\n"
        f"Regime: `{data['regime']}`\n"
        f"{pnl_icon} Portfolio PnL: `{pnl:+.2f}%`\n"
        f"Win Rate: `{portfolio.get('win_rate', 0):.1f}%`\n\n"
        f"*Market Call:* _{market_call}_"
    )
    send_telegram(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate weekly NEPSE strategy report")
    parser.add_argument("--date", help="Week ending date (YYYY-MM-DD); default: today")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--output-dir", default="data/reports")
    args = parser.parse_args()

    week_ending = date.fromisoformat(args.date) if args.date else date.today()
    generate_weekly_report(
        week_ending=week_ending,
        save=not args.no_save,
        send_telegram=not args.no_telegram,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
