"""
strategy/dashboard.py
Two dashboard modes:
  1. Rich terminal dashboard  — live updating tables in the terminal
  2. Lightweight HTML auto-refresh — writes dashboard.html every N seconds

Run:
  python -m strategy.dashboard            # terminal (default)
  python -m strategy.dashboard --html     # HTML mode
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loader helpers
# ---------------------------------------------------------------------------

def _load_latest_watchlist() -> list[dict]:
    try:
        from strategy.watchlist import load_latest_watchlist
        return load_latest_watchlist()
    except Exception:
        return []


def _load_portfolio_summary() -> dict:
    try:
        from strategy.portfolio import load_portfolio
        p = load_portfolio()
        return p.get_summary()
    except Exception:
        return {}


def _load_regime() -> dict:
    try:
        from strategy.regime_detector import detect_regime
        result = detect_regime()
        return result if isinstance(result, dict) else {"regime": str(result)}
    except Exception:
        return {"regime": "UNKNOWN"}


def _load_events() -> list[dict]:
    try:
        from pipeline.event_calendar import get_upcoming_events
        return get_upcoming_events(days_ahead=7)
    except Exception:
        return []


def _load_sector_rotation() -> list[dict]:
    try:
        from strategy.sector_rotation import get_rotation_signal
        result = get_rotation_signal()
        return result.get("ranked_sectors", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# TERMINAL DASHBOARD (rich)
# ---------------------------------------------------------------------------

def _build_terminal_dashboard() -> None:
    """Render a rich terminal dashboard using the `rich` library."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text
        from rich import box
        from rich.live import Live
        from rich.layout import Layout
    except ImportError:
        logger.error("Install 'rich': pip install rich")
        return

    console = Console()

    def build_layout() -> Layout:
        watchlist = _load_latest_watchlist()
        portfolio = _load_portfolio_summary()
        regime_info = _load_regime()
        events = _load_events()
        sectors = _load_sector_rotation()

        regime = regime_info.get("regime", "UNKNOWN")
        regime_color = {
            "BULL": "green", "EARLY_BULL": "green",
            "BEAR": "red", "CAPITULATION": "red",
            "SIDEWAYS": "yellow", "CONSOLIDATION": "yellow",
            "RECOVERY": "cyan",
        }.get(regime, "white")

        # ── Header ──────────────────────────────────────────────────────────
        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        header = Panel(
            Text(
                f"  🐟  MIROFISH NEPSE DASHBOARD   {now_str}   "
                f"Regime: [{regime_color}]{regime}[/{regime_color}]",
                justify="center",
            ),
            style="bold blue",
            box=box.DOUBLE_EDGE,
        )

        # ── Watchlist table ──────────────────────────────────────────────────
        wl_table = Table(
            title="📋 Daily Watchlist",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold cyan",
        )
        wl_table.add_column("#", width=3)
        wl_table.add_column("Symbol", width=8)
        wl_table.add_column("Score", width=6, justify="right")
        wl_table.add_column("Tier", width=6)
        wl_table.add_column("RSI", width=5, justify="right")
        wl_table.add_column("Vol×", width=5, justify="right")
        wl_table.add_column("MF", width=6, justify="right")
        wl_table.add_column("Action", width=7)

        tier_style = {"A": "bold green", "B": "yellow", "C": "dim cyan", "AVOID": "red"}
        for i, w in enumerate(watchlist[:10], 1):
            tier = w.get("tier", "C")
            style = tier_style.get(tier, "white")
            wl_table.add_row(
                str(i),
                w.get("symbol", ""),
                f"{w.get('score', 0):.1f}",
                f"[{style}]{tier}[/{style}]",
                f"{w.get('rsi', 50):.1f}",
                f"{w.get('vol_ratio', 1):.1f}",
                f"{w.get('mirofish_score', 0):+.2f}",
                w.get("action", ""),
            )
        if not watchlist:
            wl_table.add_row("–", "No data", "–", "–", "–", "–", "–", "–")

        # ── Portfolio table ──────────────────────────────────────────────────
        pf_table = Table(
            title="💼 Portfolio",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
        )
        pf_table.add_column("Symbol", width=8)
        pf_table.add_column("Entry", width=8, justify="right")
        pf_table.add_column("Current", width=8, justify="right")
        pf_table.add_column("Shares", width=6, justify="right")
        pf_table.add_column("PnL %", width=8, justify="right")
        pf_table.add_column("Strategy", width=16)

        for pos in portfolio.get("open_positions", []):
            pnl = pos.get("unrealised_pnl_pct", 0.0)
            pnl_style = "green" if pnl >= 0 else "red"
            pf_table.add_row(
                pos.get("symbol", ""),
                f"{pos.get('entry_price', 0):.2f}",
                f"{pos.get('current_price', 0):.2f}",
                str(pos.get("shares", 0)),
                f"[{pnl_style}]{pnl:+.2f}%[/{pnl_style}]",
                pos.get("strategy", ""),
            )
        if not portfolio.get("open_positions"):
            pf_table.add_row("–", "–", "–", "–", "No open positions", "–")

        # ── Portfolio summary panel ──────────────────────────────────────────
        total_pnl = portfolio.get("total_pnl_pct", 0.0)
        pnl_c = "green" if total_pnl >= 0 else "red"
        summary_text = (
            f"Cash: NPR {portfolio.get('cash_remaining', 0):,.0f}   "
            f"Portfolio: NPR {portfolio.get('total_portfolio_value', 0):,.0f}   "
            f"PnL: [{pnl_c}]{total_pnl:+.2f}%[/{pnl_c}]   "
            f"Win rate: {portfolio.get('win_rate', 0):.1f}%   "
            f"Trades: {portfolio.get('total_trades', 0)}"
        )
        pf_summary = Panel(Text.from_markup(summary_text), title="Summary", style="magenta")

        # ── Sector rotation ──────────────────────────────────────────────────
        sec_table = Table(
            title="🔄 Sector Rotation",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold yellow",
        )
        sec_table.add_column("Rank", width=5, justify="right")
        sec_table.add_column("Sector", width=14)
        sec_table.add_column("Score", width=7, justify="right")

        for i, sec in enumerate(sectors[:6], 1):
            sec_table.add_row(
                str(i),
                sec.get("sector", ""),
                f"{sec.get('combined_score', 0):.3f}",
            )
        if not sectors:
            sec_table.add_row("–", "No data", "–")

        # ── Events ──────────────────────────────────────────────────────────
        ev_table = Table(
            title="📅 Upcoming Events",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold red",
        )
        ev_table.add_column("Date", width=11)
        ev_table.add_column("Event", width=28)
        ev_table.add_column("Impact", width=8)

        impact_style = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "dim white"}
        for ev in events[:5]:
            imp = ev.get("impact", "LOW")
            sty = impact_style.get(imp, "white")
            ev_table.add_row(
                str(ev.get("date", "")),
                ev.get("name", ""),
                f"[{sty}]{imp}[/{sty}]",
            )
        if not events:
            ev_table.add_row("–", "No upcoming events", "–")

        # ── Assemble layout ──────────────────────────────────────────────────
        layout = Layout()
        layout.split_column(
            Layout(header, size=3),
            Layout(name="middle"),
            Layout(pf_summary, size=3),
        )
        layout["middle"].split_row(
            Layout(wl_table),
            Layout(name="right_col"),
        )
        layout["right_col"].split_column(
            Layout(pf_table),
            Layout(sec_table),
            Layout(ev_table),
        )
        return layout

    try:
        with Live(build_layout(), refresh_per_second=0.2, screen=True) as live:
            while True:
                time.sleep(30)
                live.update(build_layout())
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")


# ---------------------------------------------------------------------------
# HTML DASHBOARD
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta http-equiv="refresh" content="{refresh}"/>
  <title>MiroFish NEPSE Dashboard</title>
  <style>
    body {{ font-family: 'Courier New', monospace; background:#0d1117; color:#c9d1d9; margin:0; padding:16px; }}
    h1 {{ color:#58a6ff; font-size:1.3em; }}
    h2 {{ color:#f0883e; font-size:1.0em; margin-top:20px; }}
    table {{ border-collapse:collapse; width:100%; margin-bottom:16px; font-size:0.85em; }}
    th {{ background:#161b22; color:#8b949e; padding:6px 10px; text-align:left; border-bottom:1px solid #30363d; }}
    td {{ padding:5px 10px; border-bottom:1px solid #21262d; }}
    tr:hover td {{ background:#161b22; }}
    .tier-A {{ color:#3fb950; font-weight:bold; }}
    .tier-B {{ color:#d29922; }}
    .tier-C {{ color:#58a6ff; }}
    .tier-AVOID {{ color:#f85149; }}
    .pos {{ color:#3fb950; }}
    .neg {{ color:#f85149; }}
    .regime {{ padding:4px 10px; border-radius:4px; font-weight:bold; }}
    .regime-BULL {{ background:#1a3a2a; color:#3fb950; }}
    .regime-BEAR {{ background:#3a1a1a; color:#f85149; }}
    .regime-SIDEWAYS {{ background:#3a3a1a; color:#d29922; }}
    .regime-RECOVERY {{ background:#1a2a3a; color:#58a6ff; }}
    .header-bar {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }}
    .meta {{ color:#8b949e; font-size:0.8em; }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  </style>
</head>
<body>
<div class="header-bar">
  <h1>🐟 MiroFish NEPSE Dashboard</h1>
  <span class="meta">Updated: {updated} &nbsp;|&nbsp; Auto-refresh: {refresh}s</span>
</div>

<div style="margin-bottom:12px;">
  Regime: <span class="regime regime-{regime}">{regime}</span>
  &nbsp;|&nbsp; Cash: NPR {cash:,} &nbsp;|&nbsp; Portfolio: NPR {portfolio_value:,}
  &nbsp;|&nbsp; PnL: <span class="{pnl_cls}">{pnl_pct:+.2f}%</span>
  &nbsp;|&nbsp; Win Rate: {win_rate:.1f}%
</div>

<div class="two-col">
<div>
<h2>📋 Daily Watchlist</h2>
<table>
  <tr><th>#</th><th>Symbol</th><th>Score</th><th>Tier</th><th>RSI</th><th>Vol×</th><th>MF</th><th>Action</th></tr>
  {watchlist_rows}
</table>
</div>

<div>
<h2>💼 Open Positions</h2>
<table>
  <tr><th>Symbol</th><th>Entry</th><th>Shares</th><th>PnL %</th><th>Strategy</th></tr>
  {positions_rows}
</table>

<h2>🔄 Sector Rotation</h2>
<table>
  <tr><th>Rank</th><th>Sector</th><th>Score</th></tr>
  {sectors_rows}
</table>

<h2>📅 Upcoming Events</h2>
<table>
  <tr><th>Date</th><th>Event</th><th>Impact</th></tr>
  {events_rows}
</table>
</div>
</div>

</body>
</html>
"""


def _build_html_dashboard(refresh: int = 60) -> str:
    watchlist = _load_latest_watchlist()
    portfolio = _load_portfolio_summary()
    regime_info = _load_regime()
    events = _load_events()
    sectors = _load_sector_rotation()

    regime = regime_info.get("regime", "UNKNOWN")

    # Watchlist rows
    wl_rows = []
    for i, w in enumerate(watchlist[:10], 1):
        tier = w.get("tier", "C")
        wl_rows.append(
            f"<tr><td>{i}</td><td><b>{w.get('symbol','')}</b></td>"
            f"<td>{w.get('score',0):.1f}</td>"
            f"<td class='tier-{tier}'>{tier}</td>"
            f"<td>{w.get('rsi',50):.1f}</td>"
            f"<td>{w.get('vol_ratio',1):.1f}</td>"
            f"<td>{w.get('mirofish_score',0):+.2f}</td>"
            f"<td>{w.get('action','')}</td></tr>"
        )
    if not wl_rows:
        wl_rows = ["<tr><td colspan='8'>No watchlist data</td></tr>"]

    # Positions rows
    pos_rows = []
    for pos in portfolio.get("open_positions", []):
        pnl = pos.get("unrealised_pnl_pct", 0.0)
        cls = "pos" if pnl >= 0 else "neg"
        pos_rows.append(
            f"<tr><td><b>{pos.get('symbol','')}</b></td>"
            f"<td>{pos.get('entry_price',0):.2f}</td>"
            f"<td>{pos.get('shares',0)}</td>"
            f"<td class='{cls}'>{pnl:+.2f}%</td>"
            f"<td>{pos.get('strategy','')}</td></tr>"
        )
    if not pos_rows:
        pos_rows = ["<tr><td colspan='5'>No open positions</td></tr>"]

    # Sector rows
    sec_rows = []
    for i, sec in enumerate(sectors[:6], 1):
        sec_rows.append(
            f"<tr><td>{i}</td><td>{sec.get('sector','')}</td>"
            f"<td>{sec.get('combined_score',0):.3f}</td></tr>"
        )
    if not sec_rows:
        sec_rows = ["<tr><td colspan='3'>No data</td></tr>"]

    # Events rows
    ev_rows = []
    impact_cls = {"HIGH": "neg", "MEDIUM": "", "LOW": "meta"}
    for ev in events[:5]:
        imp = ev.get("impact", "LOW")
        cls = impact_cls.get(imp, "")
        ev_rows.append(
            f"<tr><td>{ev.get('date','')}</td>"
            f"<td>{ev.get('name','')}</td>"
            f"<td class='{cls}'>{imp}</td></tr>"
        )
    if not ev_rows:
        ev_rows = ["<tr><td colspan='3'>No upcoming events</td></tr>"]

    total_pnl = portfolio.get("total_pnl_pct", 0.0)

    return _HTML_TEMPLATE.format(
        refresh=refresh,
        updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        regime=regime,
        cash=int(portfolio.get("cash_remaining", 0)),
        portfolio_value=int(portfolio.get("total_portfolio_value", 0)),
        pnl_pct=total_pnl,
        pnl_cls="pos" if total_pnl >= 0 else "neg",
        win_rate=portfolio.get("win_rate", 0.0),
        watchlist_rows="\n  ".join(wl_rows),
        positions_rows="\n  ".join(pos_rows),
        sectors_rows="\n  ".join(sec_rows),
        events_rows="\n  ".join(ev_rows),
    )


def run_html_dashboard(
    output_path: str = "data/dashboard.html",
    refresh_seconds: int = 60,
) -> None:
    """Write dashboard HTML on a loop; open it in any browser."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"HTML dashboard → {output_path}  (refresh every {refresh_seconds}s)")
    print("Open the file in your browser. Press Ctrl+C to stop.\n")
    try:
        while True:
            html = _build_html_dashboard(refresh=refresh_seconds)
            Path(output_path).write_text(html, encoding="utf-8")
            logger.debug("Dashboard written → %s", output_path)
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="MiroFish NEPSE Dashboard")
    parser.add_argument("--html", action="store_true",
                        help="Run HTML auto-refresh mode (default: terminal)")
    parser.add_argument("--output", default="data/dashboard.html",
                        help="HTML output path (only used with --html)")
    parser.add_argument("--refresh", type=int, default=60,
                        help="Refresh interval in seconds (default: 60)")
    args = parser.parse_args()

    if args.html:
        run_html_dashboard(output_path=args.output, refresh_seconds=args.refresh)
    else:
        _build_terminal_dashboard()


if __name__ == "__main__":
    main()
