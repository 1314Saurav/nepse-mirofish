"""
dashboard/app.py
NEPSE MiroFish — Web Analytics Dashboard (FastAPI)

Serves a full single-page analytical dashboard at http://localhost:8080.
Replaces Telegram notifications entirely — everything the user needs is in
the browser: overview KPIs, watchlist/signals, portfolio, market analysis,
and weekly reports.

Run:
    python dashboard/app.py
    python dashboard/app.py --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path & load .env
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="NEPSE MiroFish Dashboard", version="1.0.0")

# ---------------------------------------------------------------------------
# NST helpers
# ---------------------------------------------------------------------------

NST = timezone(timedelta(hours=5, minutes=45))


def nst_now() -> datetime:
    return datetime.now(tz=NST)


def is_market_open() -> bool:
    """NEPSE trades Sun–Thu 11:00–15:00 NST."""
    now = nst_now()
    if now.weekday() in (4, 5):  # Friday = 4, Saturday = 5
        return False
    return (now.hour == 11 and now.minute >= 0) or (
        12 <= now.hour < 15
    ) or (now.hour == 15 and now.minute == 0)


def is_nepse_trading_day() -> bool:
    """NEPSE trades Sunday–Thursday."""
    return nst_now().weekday() not in (4, 5)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

DATA_DIR = _ROOT / "data"
PT_DIR = DATA_DIR / "paper_trading"
NOTIF_DIR = DATA_DIR / "notifications"
REPORTS_DIR = DATA_DIR / "processed" / "paper_trading"


def _load_json(path: Path) -> Any:
    """Load a JSON file gracefully; return None if missing/corrupt."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _active_session() -> Optional[str]:
    """Return the active paper-trading session ID."""
    txt = PT_DIR / "active_session.txt"
    if txt.exists():
        content = txt.read_text(encoding="utf-8").strip()
        if content:
            return content
    # Fall back to most recently modified sub-directory
    try:
        subdirs = sorted(
            [d for d in PT_DIR.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if subdirs:
            return subdirs[0].name
    except Exception:
        pass
    return None


def _session_dir() -> Optional[Path]:
    sid = _active_session()
    if sid:
        p = PT_DIR / sid
        if p.is_dir():
            return p
    return None


def _load_state() -> dict:
    sd = _session_dir()
    if sd:
        state = _load_json(sd / "state.json")
        if state:
            return state
    return {
        "paper_trade_id": "demo",
        "capital": 1_000_000.0,
        "starting_capital": 1_000_000.0,
        "session_start": "2025-01-01",
        "positions": {},
        "trade_log": [],
    }


def _load_accuracy() -> dict:
    sd = _session_dir()
    if sd:
        acc = _load_json(sd / "accuracy_report.json")
        if acc:
            return acc
    return {}


def _load_snapshots() -> list[dict]:
    sd = _session_dir()
    snaps = []
    if sd:
        snap_dir = sd / "snapshots"
        if not snap_dir.exists():
            snap_dir = sd  # snapshots may live in session root
        for f in sorted(snap_dir.glob("snapshot_*.json")):
            d = _load_json(f)
            if d:
                snaps.append(d)
    return snaps


def _load_latest_watchlist() -> list[dict]:
    sd = _session_dir()
    if sd:
        wl_files = sorted(sd.glob("watchlist_*.json"), reverse=True)
        if wl_files:
            data = _load_json(wl_files[0])
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "watchlist" in data:
                return data["watchlist"]
    return []


def _load_notifications(limit: int = 50) -> list[dict]:
    path = NOTIF_DIR / "alerts.json"
    if not path.exists():
        return []
    entries = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    entries.append({"ts": "", "msg": line})
    except Exception:
        return []
    return entries[-limit:]


def _list_reports() -> list[dict]:
    if not REPORTS_DIR.exists():
        return []
    reports = []
    for f in sorted(REPORTS_DIR.glob("weekly_review_week*.md"), reverse=True):
        wn = f.stem.replace("weekly_review_week", "")
        reports.append({"week_num": wn, "filename": f.name, "path": str(f)})
    return reports


# ---------------------------------------------------------------------------
# API routes — data endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    sid = _active_session()
    return {
        "status": "ok",
        "nst_time": nst_now().isoformat(),
        "market_open": is_market_open(),
        "trading_day": is_nepse_trading_day(),
        "active_session": sid,
        "data_dir_exists": DATA_DIR.exists(),
    }


@app.get("/api/overview")
async def api_overview():
    state = _load_state()
    accuracy = _load_accuracy()
    snaps = _load_snapshots()
    notifs = _load_notifications(10)

    # Portfolio value
    capital = float(state.get("capital", 1_000_000))
    starting = float(state.get("starting_capital", 1_000_000))
    positions = state.get("positions", {})
    pos_value = sum(
        float(p.get("entry_price", 0)) * int(p.get("qty", 0))
        for p in positions.values()
    )
    portfolio_value = capital + pos_value
    total_return_pct = (portfolio_value / starting - 1) * 100 if starting else 0.0

    # Today's P&L from snapshots
    today_pnl = 0.0
    if len(snaps) >= 2:
        today_pnl = snaps[-1].get("return_pct", 0.0) - snaps[-2].get("return_pct", 0.0)
    elif snaps:
        today_pnl = snaps[-1].get("return_pct", 0.0)

    # Market regime from latest snapshot
    regime = "UNKNOWN"
    mf_score = 0.0
    if snaps:
        regime = snaps[-1].get("regime", "SIDEWAYS")
        mf_score = snaps[-1].get("mirofish_score", 0.0)

    # Signal accuracy
    total_correct = accuracy.get("total_correct", 0)
    total_evaluated = accuracy.get("total_evaluated", 1)
    signal_accuracy = round(total_correct / max(total_evaluated, 1) * 100, 1)

    # Top signals from watchlist
    watchlist = _load_latest_watchlist()
    top_signals = [
        {
            "symbol": w.get("symbol", ""),
            "action": w.get("action", "WATCH"),
            "score": round(float(w.get("score", w.get("combined_score", 0))), 2),
            "target": w.get("target_price", w.get("target", 0)),
            "stop_loss": w.get("stop_loss", w.get("stop", 0)),
            "confidence": w.get("confidence", w.get("confidence_pct", 0)),
        }
        for w in watchlist[:10]
    ]

    return {
        "nst_time": nst_now().strftime("%Y-%m-%d %H:%M:%S NST"),
        "market_open": is_market_open(),
        "trading_day": is_nepse_trading_day(),
        "portfolio_value": round(portfolio_value, 2),
        "portfolio_value_formatted": f"NPR {portfolio_value:,.0f}",
        "today_pnl_pct": round(today_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "regime": regime,
        "mirofish_score": round(mf_score, 3),
        "signal_accuracy": signal_accuracy,
        "top_signals": top_signals,
        "notifications": notifs,
        "session_id": state.get("paper_trade_id", "unknown"),
        "open_positions": len(positions),
    }


@app.get("/api/watchlist")
async def api_watchlist():
    watchlist = _load_latest_watchlist()
    snaps = _load_snapshots()

    # Signal history: last 30 days accuracy from snapshots
    signal_history = []
    for s in snaps[-30:]:
        signal_history.append({
            "date": s.get("date", ""),
            "regime": s.get("regime", "UNKNOWN"),
            "mirofish_score": round(float(s.get("mirofish_score", 0)), 3),
        })

    return {
        "watchlist": watchlist,
        "signal_history": signal_history,
        "last_updated": nst_now().isoformat(),
    }


@app.get("/api/portfolio")
async def api_portfolio():
    state = _load_state()
    snaps = _load_snapshots()

    capital = float(state.get("capital", 0))
    starting = float(state.get("starting_capital", 1_000_000))
    positions_raw = state.get("positions", {})
    trade_log = state.get("trade_log", [])

    # Enrich positions
    positions = []
    total_invested = 0.0
    for sym, pos in positions_raw.items():
        ep = float(pos.get("entry_price", 0))
        qty = int(pos.get("qty", 0))
        invested = ep * qty
        total_invested += invested
        # Approximate current price from latest snapshot if available
        current_price = ep  # fallback
        pnl_npr = (current_price - ep) * qty
        pnl_pct = (current_price / ep - 1) * 100 if ep else 0.0
        entry_date_str = pos.get("entry_date", "")
        days_held = 0
        try:
            from datetime import date as _date
            ed = _date.fromisoformat(entry_date_str)
            days_held = (nst_now().date() - ed).days
        except Exception:
            pass
        positions.append({
            "symbol": sym,
            "qty": qty,
            "entry_price": round(ep, 2),
            "current_price": round(current_price, 2),
            "pnl_npr": round(pnl_npr, 2),
            "pnl_pct": round(pnl_pct, 2),
            "days_held": days_held,
            "strategy": pos.get("strategy", "momentum"),
            "stop_loss": round(float(pos.get("stop_loss", 0)), 2),
            "target_price": round(float(pos.get("target_price", 0)), 2),
            "mirofish_score": round(float(pos.get("mirofish_score", 0)), 3),
        })

    # Equity curve from snapshots
    equity_curve = [
        {
            "date": s.get("date", ""),
            "portfolio_value": round(float(s.get("portfolio_value", starting)), 2),
            "cash": round(float(s.get("cash", capital)), 2),
        }
        for s in snaps
    ]

    # Unrealised P&L
    unrealised_pnl = sum(p["pnl_npr"] for p in positions)
    portfolio_value = capital + total_invested

    return {
        "positions": positions,
        "equity_curve": equity_curve,
        "cash_balance": round(capital, 2),
        "cash_balance_formatted": f"NPR {capital:,.0f}",
        "total_invested": round(total_invested, 2),
        "total_invested_formatted": f"NPR {total_invested:,.0f}",
        "unrealised_pnl": round(unrealised_pnl, 2),
        "unrealised_pnl_formatted": f"NPR {unrealised_pnl:,.0f}",
        "portfolio_value": round(portfolio_value, 2),
        "trade_history": trade_log[-20:],
    }


@app.get("/api/signals")
async def api_signals():
    sd = _session_dir()
    history = []
    if sd:
        sig_path = sd / "signal_history.json"
        raw = _load_json(sig_path)
        if isinstance(raw, list):
            history = raw[-30:]

    accuracy = _load_accuracy()
    return {
        "signal_history": history,
        "accuracy_report": accuracy,
        "last_updated": nst_now().isoformat(),
    }


@app.get("/api/notifications")
async def api_notifications():
    notifs = _load_notifications(50)
    return {"notifications": notifs, "count": len(notifs)}


@app.get("/api/reports")
async def api_reports():
    return {"reports": _list_reports()}


@app.get("/api/report/{week_num}")
async def api_report(week_num: str):
    path = REPORTS_DIR / f"weekly_review_week{week_num}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    content = path.read_text(encoding="utf-8")
    return {"week_num": week_num, "content": content}


@app.post("/api/run-cycle")
async def api_run_cycle(background_tasks: BackgroundTasks):
    def _run():
        try:
            from paper_trading.engine import PaperTradingEngine
            sid = _active_session()
            engine = PaperTradingEngine(paper_trade_id=sid)
            result = engine.run_daily_cycle()
            logger.info("Daily cycle completed: %s", result)
        except Exception as exc:
            logger.error("Daily cycle failed: %s", exc)

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Daily analysis cycle triggered in background"}


@app.post("/api/weekly-report")
async def api_weekly_report(background_tasks: BackgroundTasks):
    def _run():
        try:
            from paper_trading.weekly_review import WeeklyReviewGenerator
            gen = WeeklyReviewGenerator()
            gen.generate_review()
            logger.info("Weekly report generated")
        except Exception as exc:
            logger.error("Weekly report failed: %s", exc)

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Weekly report generation triggered in background"}


# ---------------------------------------------------------------------------
# HTML dashboard (single page)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEPSE MiroFish Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --bg2: #161b22;
    --bg3: #1c2128;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --green: #00ff88;
    --green-dim: #00c86a;
    --red: #ff4d4f;
    --yellow: #ffd700;
    --blue: #58a6ff;
    --purple: #bc8cff;
    --orange: #f0883e;
    --card-shadow: 0 4px 24px rgba(0,0,0,0.4);
    --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    min-height: 100vh;
    font-size: 14px;
  }

  /* ── Sidebar ── */
  #sidebar {
    width: 220px;
    min-width: 220px;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 0;
    position: sticky;
    top: 0;
    height: 100vh;
    z-index: 100;
  }
  #sidebar-logo {
    padding: 20px 18px 16px;
    border-bottom: 1px solid var(--border);
  }
  #sidebar-logo h1 {
    font-size: 16px;
    font-weight: 700;
    color: var(--green);
    letter-spacing: 0.5px;
  }
  #sidebar-logo p {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
  }
  #sidebar nav { flex: 1; padding: 12px 0; }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    cursor: pointer;
    border-radius: 0;
    transition: background 0.15s, color 0.15s;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 500;
    border-left: 3px solid transparent;
    user-select: none;
  }
  .nav-item:hover { background: var(--bg3); color: var(--text); }
  .nav-item.active {
    background: rgba(0,255,136,0.08);
    color: var(--green);
    border-left-color: var(--green);
  }
  .nav-icon { font-size: 16px; width: 20px; text-align: center; }
  #sidebar-footer {
    padding: 14px 18px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-muted);
  }
  #sidebar-footer #nst-clock {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
  }
  #market-status-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .badge-open { background: rgba(0,255,136,0.15); color: var(--green); }
  .badge-closed { background: rgba(255,77,79,0.15); color: var(--red); }
  #refresh-countdown {
    margin-top: 8px;
    font-size: 11px;
    color: var(--text-muted);
  }

  /* ── Main content ── */
  #main {
    flex: 1;
    overflow-y: auto;
    min-width: 0;
  }
  .tab-panel { display: none; padding: 24px; max-width: 1400px; }
  .tab-panel.active { display: block; }

  /* ── Section title ── */
  .section-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
  }
  .section-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 20px;
  }

  /* ── KPI Cards ── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .kpi-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--card-shadow);
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, var(--green));
  }
  .kpi-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }
  .kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    margin-bottom: 6px;
    font-variant-numeric: tabular-nums;
  }
  .kpi-sub {
    font-size: 12px;
    color: var(--text-muted);
  }
  .kpi-positive { color: var(--green) !important; }
  .kpi-negative { color: var(--red) !important; }

  /* ── Cards / panels ── */
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--card-shadow);
    margin-bottom: 20px;
  }
  .card-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .card-title span { color: var(--text-muted); font-size: 11px; font-weight: 400; }

  /* ── Tables ── */
  .tbl-wrap { overflow-x: auto; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  th {
    text-align: left;
    padding: 10px 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  td {
    padding: 9px 12px;
    border-bottom: 1px solid rgba(48,54,61,0.5);
    color: var(--text);
    white-space: nowrap;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .row-buy td { border-left: 2px solid var(--green); }
  .row-sell td { border-left: 2px solid var(--red); }
  .row-watch td { border-left: 2px solid var(--yellow); }
  .row-hold td { border-left: 2px solid var(--blue); }

  /* ── Badges ── */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.4px;
  }
  .badge-buy   { background: rgba(0,255,136,0.15); color: var(--green); }
  .badge-sell  { background: rgba(255,77,79,0.15); color: var(--red); }
  .badge-watch { background: rgba(255,215,0,0.15); color: var(--yellow); }
  .badge-hold  { background: rgba(88,166,255,0.15); color: var(--blue); }
  .badge-bull  { background: rgba(0,255,136,0.15); color: var(--green); }
  .badge-bear  { background: rgba(255,77,79,0.15); color: var(--red); }
  .badge-side  { background: rgba(255,215,0,0.12); color: var(--yellow); }
  .badge-unk   { background: rgba(139,148,158,0.15); color: var(--text-muted); }

  /* ── Chart containers ── */
  .chart-wrap { position: relative; height: 260px; }
  .chart-wrap-lg { position: relative; height: 340px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) {
    .two-col { grid-template-columns: 1fr; }
    .three-col { grid-template-columns: 1fr; }
  }

  /* ── Notification feed ── */
  .notif-feed { max-height: 300px; overflow-y: auto; }
  .notif-item {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(48,54,61,0.4);
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }
  .notif-item:last-child { border-bottom: none; }
  .notif-ts {
    font-size: 10px;
    color: var(--text-muted);
    white-space: nowrap;
    margin-top: 2px;
    min-width: 100px;
  }
  .notif-msg {
    font-size: 12px;
    color: var(--text);
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* ── Buttons ── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 18px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
    text-decoration: none;
  }
  .btn-primary {
    background: var(--green);
    color: #0d1117;
  }
  .btn-primary:hover { background: var(--green-dim); }
  .btn-secondary {
    background: var(--bg3);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover { border-color: var(--green); color: var(--green); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── Loading spinner ── */
  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--green);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-overlay {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: var(--text-muted);
    gap: 10px;
  }

  /* ── Reports ── */
  .report-list { display: flex; flex-direction: column; gap: 8px; }
  .report-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--bg3);
    border-radius: 8px;
    border: 1px solid var(--border);
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .report-item:hover { border-color: var(--green); }
  .report-content {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-top: 20px;
    max-height: 600px;
    overflow-y: auto;
    line-height: 1.7;
  }
  .report-content h1, .report-content h2, .report-content h3 {
    color: var(--green);
    margin: 16px 0 8px;
  }
  .report-content p { color: var(--text); margin-bottom: 10px; }
  .report-content code {
    background: var(--bg3);
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 12px;
  }
  .report-content pre {
    background: var(--bg3);
    padding: 14px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 12px 0;
  }
  .report-content table { margin: 12px 0; }
  .report-content th, .report-content td {
    border: 1px solid var(--border);
    padding: 7px 12px;
  }
  .report-content blockquote {
    border-left: 3px solid var(--green);
    padding-left: 14px;
    color: var(--text-muted);
    margin: 10px 0;
  }

  /* ── Overview header ── */
  .overview-header {
    background: linear-gradient(135deg, var(--bg2) 0%, rgba(0,255,136,0.04) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }
  .overview-header-left h2 {
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
  }
  .overview-header-left p {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 4px;
  }
  #header-time {
    font-size: 28px;
    font-weight: 700;
    color: var(--green);
    font-variant-numeric: tabular-nums;
  }

  /* ── Score bar ── */
  .score-bar-wrap { display: flex; align-items: center; gap: 8px; }
  .score-bar {
    flex: 1;
    height: 6px;
    background: var(--bg3);
    border-radius: 3px;
    overflow: hidden;
  }
  .score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
  }
  .score-bar-fill.high  { background: var(--green); }
  .score-bar-fill.med   { background: var(--yellow); }
  .score-bar-fill.low   { background: var(--red); }

  /* ── Accuracy table colors ── */
  .acc-good { color: var(--green); font-weight: 600; }
  .acc-ok   { color: var(--yellow); font-weight: 600; }
  .acc-bad  { color: var(--red); font-weight: 600; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--text-muted);
  }
  .empty-state p { margin-top: 8px; font-size: 13px; }
</style>
</head>
<body>

<!-- ═══════════════════════════ SIDEBAR ═══════════════════════════ -->
<div id="sidebar">
  <div id="sidebar-logo">
    <h1>🐟 MiroFish</h1>
    <p>NEPSE Analytics Dashboard</p>
  </div>
  <nav id="sidenav">
    <div class="nav-item active" data-tab="overview">
      <span class="nav-icon">📊</span> Overview
    </div>
    <div class="nav-item" data-tab="watchlist">
      <span class="nav-icon">🎯</span> Watchlist &amp; Signals
    </div>
    <div class="nav-item" data-tab="portfolio">
      <span class="nav-icon">💼</span> Portfolio
    </div>
    <div class="nav-item" data-tab="analysis">
      <span class="nav-icon">📈</span> Market Analysis
    </div>
    <div class="nav-item" data-tab="reports">
      <span class="nav-icon">📋</span> Reports
    </div>
  </nav>
  <div id="sidebar-footer">
    <div id="nst-clock">--:--:-- NST</div>
    <span id="market-status-badge" class="badge-closed">CLOSED</span>
    <div id="refresh-countdown">Auto-refresh in 30s</div>
  </div>
</div>

<!-- ═══════════════════════════ MAIN ═══════════════════════════ -->
<div id="main">

  <!-- ──────── Tab 1: Overview ──────── -->
  <div id="tab-overview" class="tab-panel active">
    <div class="overview-header">
      <div class="overview-header-left">
        <h2>Market Overview</h2>
        <p id="overview-meta">Loading...</p>
      </div>
      <div id="header-time">--:--</div>
    </div>

    <div class="kpi-grid" id="kpi-grid">
      <div class="kpi-card" style="--accent:var(--green)">
        <div class="kpi-label">Portfolio Value</div>
        <div class="kpi-value" id="kpi-pv">—</div>
        <div class="kpi-sub" id="kpi-pv-sub">Loading...</div>
      </div>
      <div class="kpi-card" style="--accent:var(--blue)">
        <div class="kpi-label">Today's P&amp;L</div>
        <div class="kpi-value" id="kpi-pnl">—</div>
        <div class="kpi-sub" id="kpi-pnl-sub">Total return: —</div>
      </div>
      <div class="kpi-card" style="--accent:var(--purple)">
        <div class="kpi-label">Market Regime</div>
        <div class="kpi-value" id="kpi-regime">—</div>
        <div class="kpi-sub" id="kpi-regime-sub">MiroFish score: —</div>
      </div>
      <div class="kpi-card" style="--accent:var(--orange)">
        <div class="kpi-label">Signal Accuracy</div>
        <div class="kpi-value" id="kpi-acc">—</div>
        <div class="kpi-sub" id="kpi-acc-sub">Last 30 days</div>
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="card-title">Today's Top Signals <span id="signals-ts"></span></div>
        <div class="tbl-wrap">
          <table>
            <thead>
              <tr>
                <th>Symbol</th><th>Action</th><th>Score</th>
                <th>Target</th><th>Stop Loss</th><th>Confidence</th>
              </tr>
            </thead>
            <tbody id="top-signals-body">
              <tr><td colspan="6" class="loading-overlay"><span class="spinner"></span> Loading...</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Recent Notifications <span id="notif-ts"></span></div>
        <div class="notif-feed" id="notif-feed">
          <div class="loading-overlay"><span class="spinner"></span> Loading...</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ──────── Tab 2: Watchlist & Signals ──────── -->
  <div id="tab-watchlist" class="tab-panel">
    <div class="section-title">Watchlist &amp; Signals</div>
    <div class="section-sub">Full scored watchlist — color coded by action</div>

    <div class="card">
      <div class="card-title">Full Watchlist <span id="wl-ts"></span></div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Symbol</th><th>MiroFish Score</th>
              <th>Technical Score</th><th>Combined Score</th>
              <th>Action</th><th>Entry Zone</th><th>Target</th>
              <th>Stop</th><th>Regime</th>
            </tr>
          </thead>
          <tbody id="watchlist-body">
            <tr><td colspan="10" class="loading-overlay"><span class="spinner"></span> Loading...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Signal History — MiroFish Score (Last 30 Days)</div>
      <div class="chart-wrap">
        <canvas id="signal-history-chart"></canvas>
      </div>
    </div>
  </div>

  <!-- ──────── Tab 3: Portfolio ──────── -->
  <div id="tab-portfolio" class="tab-panel">
    <div class="section-title">Portfolio</div>
    <div class="section-sub">Open positions, equity curve, and closed trades</div>

    <div class="three-col" style="margin-bottom:20px">
      <div class="kpi-card" style="--accent:var(--green)">
        <div class="kpi-label">Cash Balance</div>
        <div class="kpi-value" id="port-cash" style="font-size:20px">—</div>
      </div>
      <div class="kpi-card" style="--accent:var(--blue)">
        <div class="kpi-label">Total Invested</div>
        <div class="kpi-value" id="port-invested" style="font-size:20px">—</div>
      </div>
      <div class="kpi-card" style="--accent:var(--orange)">
        <div class="kpi-label">Unrealised P&amp;L</div>
        <div class="kpi-value" id="port-unrealised" style="font-size:20px">—</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Portfolio Equity Curve</div>
      <div class="chart-wrap-lg">
        <canvas id="equity-chart"></canvas>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Open Positions</div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Qty</th><th>Entry Price</th>
              <th>Current Price</th><th>P&amp;L NPR</th><th>P&amp;L %</th>
              <th>Days Held</th><th>Strategy</th>
            </tr>
          </thead>
          <tbody id="positions-body">
            <tr><td colspan="8" class="loading-overlay"><span class="spinner"></span> Loading...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Trade History <span>Last 20 closed trades</span></div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Action</th><th>Entry Price</th>
              <th>Exit Price</th><th>Qty</th><th>P&amp;L %</th>
              <th>Entry Date</th><th>Exit Date</th><th>Regime</th>
            </tr>
          </thead>
          <tbody id="trade-history-body">
            <tr><td colspan="9" class="loading-overlay"><span class="spinner"></span> Loading...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ──────── Tab 4: Market Analysis ──────── -->
  <div id="tab-analysis" class="tab-panel">
    <div class="section-title">Market Analysis</div>
    <div class="section-sub">Run analysis cycle, regime history, score distribution</div>

    <div class="card" style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
      <div style="flex:1">
        <div style="font-weight:600;margin-bottom:4px">Run Full Analysis Cycle</div>
        <div style="font-size:12px;color:var(--text-muted)">
          Triggers the complete MiroFish daily cycle: scrape → news → simulation → indicators → regime → signals → watchlist
        </div>
      </div>
      <button class="btn btn-primary" id="btn-run-cycle" onclick="runCycle()">
        ▶ Run Analysis Now
      </button>
    </div>
    <div id="cycle-status" style="display:none;margin-bottom:20px" class="card">
      <div style="color:var(--green)">✓ Analysis cycle triggered in background. Check notifications for results.</div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="card-title">Market Regime History</div>
        <div class="chart-wrap">
          <canvas id="regime-chart"></canvas>
        </div>
      </div>
      <div class="card">
        <div class="card-title">MiroFish Score Distribution</div>
        <div class="chart-wrap">
          <canvas id="score-dist-chart"></canvas>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Sector Breakdown</div>
      <div id="sector-table-wrap">
        <div class="loading-overlay"><span class="spinner"></span> Loading...</div>
      </div>
    </div>
  </div>

  <!-- ──────── Tab 5: Reports ──────── -->
  <div id="tab-reports" class="tab-panel">
    <div class="section-title">Weekly Reports</div>
    <div class="section-sub">Auto-generated weekly paper-trading reviews</div>

    <div class="card" style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:24px">
      <div style="flex:1">
        <div style="font-weight:600;margin-bottom:4px">Generate Weekly Report</div>
        <div style="font-size:12px;color:var(--text-muted)">
          Run the AI-powered weekly review generator (requires LLM API key)
        </div>
      </div>
      <button class="btn btn-secondary" id="btn-gen-report" onclick="genReport()">
        📄 Generate Report Now
      </button>
    </div>

    <div class="two-col">
      <div>
        <div class="card">
          <div class="card-title">Available Reports</div>
          <div class="report-list" id="reports-list">
            <div class="loading-overlay"><span class="spinner"></span> Loading...</div>
          </div>
        </div>
      </div>
      <div>
        <div class="card">
          <div class="card-title">Signal Accuracy Breakdown</div>
          <div class="tbl-wrap">
            <table>
              <thead>
                <tr><th>Symbol/Period</th><th>1d</th><th>3d</th><th>5d</th><th>10d</th></tr>
              </thead>
              <tbody id="accuracy-body">
                <tr><td colspan="5" class="loading-overlay"><span class="spinner"></span> Loading...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <div id="report-viewer" class="report-content" style="display:none"></div>
  </div>

</div> <!-- #main -->

<script>
// ══════════════════════════════════════════════════════════════════
//  MiroFish Dashboard JS
// ══════════════════════════════════════════════════════════════════

// ── Chart.js global defaults ──────────────────────────────────────
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
Chart.defaults.font.size = 12;

// ── Tab navigation ────────────────────────────────────────────────
const tabPanels = document.querySelectorAll('.tab-panel');
const navItems = document.querySelectorAll('.nav-item');
const loaded = {};  // track which tabs have been initialised

function switchTab(tabId) {
  tabPanels.forEach(p => p.classList.remove('active'));
  navItems.forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
  if (!loaded[tabId]) {
    loaded[tabId] = true;
    loadTab(tabId);
  }
}

navItems.forEach(n => n.addEventListener('click', () => switchTab(n.dataset.tab)));

function loadTab(tabId) {
  switch(tabId) {
    case 'overview':  loadOverview(); break;
    case 'watchlist': loadWatchlist(); break;
    case 'portfolio': loadPortfolio(); break;
    case 'analysis':  loadAnalysis(); break;
    case 'reports':   loadReports(); break;
  }
}

// ── NST clock ────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const nst = new Date(now.getTime() + (5*60 + 45)*60000);
  const h = nst.getUTCHours().toString().padStart(2,'0');
  const m = nst.getUTCMinutes().toString().padStart(2,'0');
  const s = nst.getUTCSeconds().toString().padStart(2,'0');
  document.getElementById('nst-clock').textContent = `${h}:${m}:${s} NST`;
  document.getElementById('header-time').textContent = `${h}:${m}`;
  // Market status: Sun-Thu 11:00-15:00 NST  (0=Sun ... 4=Thu)
  const dow = nst.getUTCDay(); // 0=Sun...6=Sat
  const totalMin = nst.getUTCHours()*60 + nst.getUTCMinutes();
  const trading = dow >= 0 && dow <= 4 && totalMin >= 660 && totalMin <= 900;
  const badge = document.getElementById('market-status-badge');
  badge.textContent = trading ? 'OPEN' : 'CLOSED';
  badge.className = trading ? 'badge-open' : 'badge-closed';
}
setInterval(updateClock, 1000);
updateClock();

// ── Auto-refresh countdown ────────────────────────────────────────
let refreshSec = 30;
setInterval(() => {
  refreshSec--;
  if (refreshSec <= 0) {
    refreshSec = 30;
    const activeTab = document.querySelector('.tab-panel.active')?.id?.replace('tab-','');
    if (activeTab) loadTab(activeTab);
  }
  document.getElementById('refresh-countdown').textContent =
    `Auto-refresh in ${refreshSec}s`;
}, 1000);

// ── Helpers ───────────────────────────────────────────────────────
function fmt_npr(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return 'NPR ' + Number(v).toLocaleString('en-IN', {maximumFractionDigits:0});
}

function fmt_pct(v, decimals=2) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  const n = Number(v);
  const sign = n >= 0 ? '+' : '';
  return `${sign}${n.toFixed(decimals)}%`;
}

function fmt_score(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(2);
}

function pnl_class(v) {
  if (!v && v !== 0) return '';
  return Number(v) >= 0 ? 'kpi-positive' : 'kpi-negative';
}

function action_badge(action) {
  if (!action) return '';
  const a = action.toUpperCase();
  const cls = {BUY:'badge-buy', SELL:'badge-sell', WATCH:'badge-watch', HOLD:'badge-hold'}[a] || 'badge-unk';
  return `<span class="badge ${cls}">${a}</span>`;
}

function regime_badge(r) {
  if (!r) return '—';
  const u = r.toUpperCase();
  if (u.includes('BULL'))    return `<span class="badge badge-bull">BULL</span>`;
  if (u.includes('BEAR'))    return `<span class="badge badge-bear">BEAR</span>`;
  if (u.includes('SIDE'))    return `<span class="badge badge-side">SIDEWAYS</span>`;
  return `<span class="badge badge-unk">${u}</span>`;
}

function row_class(action) {
  if (!action) return '';
  const a = action.toUpperCase();
  return {BUY:'row-buy', SELL:'row-sell', WATCH:'row-watch', HOLD:'row-hold'}[a] || '';
}

function score_bar_html(score, max=100) {
  const pct = Math.min(100, Math.max(0, (Number(score)/max)*100));
  const cls = pct>=70 ? 'high' : pct>=40 ? 'med' : 'low';
  return `<div class="score-bar-wrap">
    <span style="min-width:36px;font-variant-numeric:tabular-nums">${Number(score).toFixed(0)}</span>
    <div class="score-bar"><div class="score-bar-fill ${cls}" style="width:${pct}%"></div></div>
  </div>`;
}

function ts_label(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleString('en-US', {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
  } catch { return iso; }
}

function empty_row(cols, msg='No data') {
  return `<tr><td colspan="${cols}" class="empty-state">${msg}</td></tr>`;
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ── Charts registry (for destroy-on-reload) ───────────────────────
const charts = {};
function mkChart(id, config) {
  if (charts[id]) { charts[id].destroy(); }
  const ctx = document.getElementById(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx, config);
  return charts[id];
}

// ══════════════════════════════════════════════════════════════════
//  Tab 1 — Overview
// ══════════════════════════════════════════════════════════════════
async function loadOverview() {
  try {
    const d = await api('/api/overview');

    // Header
    const dow = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
    const nstNow = new Date(Date.now() + (5*60+45)*60000);
    document.getElementById('overview-meta').textContent =
      `${dow[nstNow.getUTCDay()]} · ${nstNow.toISOString().slice(0,10)} · Session: ${d.session_id || '—'} · Open Positions: ${d.open_positions || 0}`;

    // KPI — Portfolio Value
    document.getElementById('kpi-pv').textContent = d.portfolio_value_formatted || '—';
    const retEl = document.getElementById('kpi-pv-sub');
    retEl.textContent = `Total return: ${fmt_pct(d.total_return_pct)}`;
    retEl.className = 'kpi-sub ' + pnl_class(d.total_return_pct);

    // KPI — Today P&L
    const pnlEl = document.getElementById('kpi-pnl');
    pnlEl.textContent = fmt_pct(d.today_pnl_pct);
    pnlEl.className = 'kpi-value ' + pnl_class(d.today_pnl_pct);
    document.getElementById('kpi-pnl-sub').textContent = `Total return: ${fmt_pct(d.total_return_pct)}`;

    // KPI — Regime
    const regEl = document.getElementById('kpi-regime');
    const r = (d.regime || 'UNKNOWN').toUpperCase();
    regEl.innerHTML = regime_badge(r);
    document.getElementById('kpi-regime-sub').textContent =
      `MiroFish score: ${fmt_score(d.mirofish_score)}`;

    // KPI — Accuracy
    const accEl = document.getElementById('kpi-acc');
    accEl.textContent = d.signal_accuracy ? d.signal_accuracy + '%' : '—';
    accEl.className = 'kpi-value ' + (d.signal_accuracy >= 60 ? 'kpi-positive' : d.signal_accuracy < 45 ? 'kpi-negative' : '');

    // Top signals
    document.getElementById('signals-ts').textContent = nstNow.toISOString().slice(0,10);
    const tbody = document.getElementById('top-signals-body');
    if (!d.top_signals || d.top_signals.length === 0) {
      tbody.innerHTML = empty_row(6, 'No signals available');
    } else {
      tbody.innerHTML = d.top_signals.map(s => `
        <tr class="${row_class(s.action)}">
          <td><strong>${s.symbol}</strong></td>
          <td>${action_badge(s.action)}</td>
          <td>${score_bar_html(s.score)}</td>
          <td>${fmt_npr(s.target)}</td>
          <td>${fmt_npr(s.stop_loss)}</td>
          <td>${s.confidence ? Number(s.confidence).toFixed(0)+'%' : '—'}</td>
        </tr>`).join('');
    }

    // Notifications
    const feed = document.getElementById('notif-feed');
    if (!d.notifications || d.notifications.length === 0) {
      feed.innerHTML = '<div class="empty-state"><p>No notifications yet</p></div>';
    } else {
      feed.innerHTML = [...d.notifications].reverse().map(n => `
        <div class="notif-item">
          <div class="notif-ts">${ts_label(n.ts)}</div>
          <div class="notif-msg">${escHtml(n.msg || '')}</div>
        </div>`).join('');
    }

  } catch(e) {
    console.error('Overview load failed:', e);
  }
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

// ══════════════════════════════════════════════════════════════════
//  Tab 2 — Watchlist & Signals
// ══════════════════════════════════════════════════════════════════
async function loadWatchlist() {
  try {
    const d = await api('/api/watchlist');
    const wl = d.watchlist || [];
    const now = new Date(Date.now()+(5*60+45)*60000);
    document.getElementById('wl-ts').textContent = now.toISOString().slice(0,10);

    const tbody = document.getElementById('watchlist-body');
    if (wl.length === 0) {
      tbody.innerHTML = empty_row(10, 'No watchlist data — run analysis cycle first');
    } else {
      tbody.innerHTML = wl.map((w,i) => {
        const action = w.action || 'WATCH';
        const mf = w.mirofish_score ?? w.score ?? 0;
        const tech = w.technical_score ?? w.tech_score ?? 0;
        const combined = w.combined_score ?? w.score ?? mf;
        const entry = w.entry_price ?? w.entry_zone ?? w.entry ?? 0;
        const target = w.target_price ?? w.target ?? 0;
        const stop = w.stop_loss ?? w.stop ?? 0;
        const regime = w.regime ?? '—';
        return `<tr class="${row_class(action)}">
          <td>${i+1}</td>
          <td><strong>${w.symbol||'—'}</strong></td>
          <td>${score_bar_html(mf)}</td>
          <td>${fmt_score(tech)}</td>
          <td>${score_bar_html(combined)}</td>
          <td>${action_badge(action)}</td>
          <td>${fmt_npr(entry)}</td>
          <td>${fmt_npr(target)}</td>
          <td>${fmt_npr(stop)}</td>
          <td>${regime_badge(regime)}</td>
        </tr>`;
      }).join('');
    }

    // Signal history chart
    const hist = d.signal_history || [];
    if (hist.length > 0) {
      const labels = hist.map(h => h.date);
      const scores = hist.map(h => Number(h.mirofish_score));
      mkChart('signal-history-chart', {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'MiroFish Score',
            data: scores,
            borderColor: '#00ff88',
            backgroundColor: 'rgba(0,255,136,0.07)',
            tension: 0.35,
            fill: true,
            pointRadius: 3,
            pointHoverRadius: 5,
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: 'rgba(48,54,61,0.5)' } },
            y: {
              grid: { color: 'rgba(48,54,61,0.5)' },
              ticks: { callback: v => v.toFixed(2) }
            }
          }
        }
      });
    }

  } catch(e) { console.error('Watchlist load failed:', e); }
}

// ══════════════════════════════════════════════════════════════════
//  Tab 3 — Portfolio
// ══════════════════════════════════════════════════════════════════
async function loadPortfolio() {
  try {
    const d = await api('/api/portfolio');

    // KPI cards
    document.getElementById('port-cash').textContent = d.cash_balance_formatted || '—';
    document.getElementById('port-invested').textContent = d.total_invested_formatted || '—';
    const unr = document.getElementById('port-unrealised');
    unr.textContent = d.unrealised_pnl_formatted || '—';
    unr.className = 'kpi-value ' + pnl_class(d.unrealised_pnl) + (d.unrealised_pnl < 0 ? ' kpi-negative' : '');

    // Equity curve chart
    const eq = d.equity_curve || [];
    if (eq.length > 0) {
      mkChart('equity-chart', {
        type: 'line',
        data: {
          labels: eq.map(e => e.date),
          datasets: [
            {
              label: 'Portfolio Value (NPR)',
              data: eq.map(e => e.portfolio_value),
              borderColor: '#00ff88',
              backgroundColor: 'rgba(0,255,136,0.06)',
              tension: 0.35, fill: true,
              pointRadius: 2,
            },
            {
              label: 'Cash (NPR)',
              data: eq.map(e => e.cash),
              borderColor: '#58a6ff',
              borderDash: [5,3],
              tension: 0.35, fill: false,
              pointRadius: 0,
            }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#8b949e' } } },
          scales: {
            x: { grid: { color: 'rgba(48,54,61,0.4)' } },
            y: {
              grid: { color: 'rgba(48,54,61,0.4)' },
              ticks: { callback: v => 'NPR ' + (v/1000).toFixed(0)+'K' }
            }
          }
        }
      });
    } else {
      document.getElementById('equity-chart').parentElement.innerHTML =
        '<div class="empty-state">No equity history yet</div>';
    }

    // Open positions table
    const pos = d.positions || [];
    const ptbody = document.getElementById('positions-body');
    if (pos.length === 0) {
      ptbody.innerHTML = empty_row(8, 'No open positions');
    } else {
      ptbody.innerHTML = pos.map(p => `
        <tr>
          <td><strong>${p.symbol}</strong></td>
          <td>${p.qty}</td>
          <td>${fmt_npr(p.entry_price)}</td>
          <td>${fmt_npr(p.current_price)}</td>
          <td class="${pnl_class(p.pnl_npr)}">${fmt_npr(p.pnl_npr)}</td>
          <td class="${pnl_class(p.pnl_pct)}">${fmt_pct(p.pnl_pct)}</td>
          <td>${p.days_held}d</td>
          <td>${p.strategy || '—'}</td>
        </tr>`).join('');
    }

    // Trade history table
    const trades = d.trade_history || [];
    const ttbody = document.getElementById('trade-history-body');
    if (trades.length === 0) {
      ttbody.innerHTML = empty_row(9, 'No closed trades yet');
    } else {
      ttbody.innerHTML = trades.slice().reverse().map(t => `
        <tr>
          <td><strong>${t.symbol}</strong></td>
          <td>${action_badge(t.action)}</td>
          <td>${fmt_npr(t.entry_price)}</td>
          <td>${fmt_npr(t.exit_price)}</td>
          <td>${t.qty}</td>
          <td class="${pnl_class(t.pnl_pct)}">${fmt_pct(t.pnl_pct)}</td>
          <td>${t.entry_date||'—'}</td>
          <td>${t.exit_date||'—'}</td>
          <td>${regime_badge(t.regime)}</td>
        </tr>`).join('');
    }

  } catch(e) { console.error('Portfolio load failed:', e); }
}

// ══════════════════════════════════════════════════════════════════
//  Tab 4 — Market Analysis
// ══════════════════════════════════════════════════════════════════
async function loadAnalysis() {
  try {
    const [wl_data, sig_data] = await Promise.all([
      api('/api/watchlist'),
      api('/api/signals'),
    ]);

    const hist = wl_data.signal_history || [];
    const wl = wl_data.watchlist || [];

    // Regime history chart
    if (hist.length > 0) {
      const regimeCounts = {};
      hist.forEach(h => { regimeCounts[h.regime] = (regimeCounts[h.regime]||0)+1; });
      const regLabels = Object.keys(regimeCounts);
      const regColors = regLabels.map(r => {
        if (r.includes('BULL')) return 'rgba(0,255,136,0.7)';
        if (r.includes('BEAR')) return 'rgba(255,77,79,0.7)';
        return 'rgba(255,215,0,0.7)';
      });
      mkChart('regime-chart', {
        type: 'doughnut',
        data: {
          labels: regLabels,
          datasets: [{ data: Object.values(regimeCounts), backgroundColor: regColors, borderWidth: 1, borderColor: '#30363d' }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#8b949e', padding: 16 } } }
        }
      });
    } else {
      document.getElementById('regime-chart').parentElement.innerHTML =
        '<div class="empty-state">No regime history yet</div>';
    }

    // Score distribution chart
    if (wl.length > 0) {
      const symbols = wl.slice(0,20).map(w => w.symbol || '?');
      const scores = wl.slice(0,20).map(w => Number(w.combined_score ?? w.score ?? w.mirofish_score ?? 0));
      const barColors = scores.map(s => s >= 70 ? 'rgba(0,255,136,0.7)' : s >= 40 ? 'rgba(255,215,0,0.7)' : 'rgba(255,77,79,0.7)');
      mkChart('score-dist-chart', {
        type: 'bar',
        data: {
          labels: symbols,
          datasets: [{
            label: 'Score',
            data: scores,
            backgroundColor: barColors,
            borderRadius: 4,
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { color: 'rgba(48,54,61,0.4)' }, ticks: { maxRotation: 45 } },
            y: { grid: { color: 'rgba(48,54,61,0.4)' }, min: 0, max: 100 }
          }
        }
      });
    }

    // Sector breakdown (derive from watchlist)
    const sectors = {};
    wl.forEach(w => {
      const sec = w.sector || 'Unknown';
      if (!sectors[sec]) sectors[sec] = {count:0, totalScore:0, buys:0};
      sectors[sec].count++;
      sectors[sec].totalScore += Number(w.combined_score??w.score??0);
      if ((w.action||'').toUpperCase()==='BUY') sectors[sec].buys++;
    });
    const secEntries = Object.entries(sectors).sort((a,b)=>b[1].count-a[1].count);
    const secWrap = document.getElementById('sector-table-wrap');
    if (secEntries.length === 0) {
      secWrap.innerHTML = '<div class="empty-state">No sector data — watchlist needed</div>';
    } else {
      secWrap.innerHTML = `<div class="tbl-wrap"><table>
        <thead><tr><th>Sector</th><th>Stocks Watched</th><th>Avg Score</th><th>BUY Signals</th></tr></thead>
        <tbody>${secEntries.map(([sec,v]) => `
          <tr>
            <td>${sec}</td>
            <td>${v.count}</td>
            <td>${(v.totalScore/v.count).toFixed(1)}</td>
            <td><span class="badge badge-buy">${v.buys}</span></td>
          </tr>`).join('')}
        </tbody></table></div>`;
    }

  } catch(e) { console.error('Analysis load failed:', e); }
}

async function runCycle() {
  const btn = document.getElementById('btn-run-cycle');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running...';
  try {
    await fetch('/api/run-cycle', {method:'POST'});
    document.getElementById('cycle-status').style.display = 'block';
    setTimeout(() => { document.getElementById('cycle-status').style.display = 'none'; }, 8000);
  } catch(e) { alert('Failed to trigger cycle: ' + e.message); }
  btn.disabled = false;
  btn.innerHTML = '▶ Run Analysis Now';
}

// ══════════════════════════════════════════════════════════════════
//  Tab 5 — Reports
// ══════════════════════════════════════════════════════════════════
async function loadReports() {
  try {
    const [rep, sig] = await Promise.all([
      api('/api/reports'),
      api('/api/signals'),
    ]);

    // Reports list
    const list = document.getElementById('reports-list');
    const reports = rep.reports || [];
    if (reports.length === 0) {
      list.innerHTML = '<div class="empty-state"><p>No reports yet — generate your first one</p></div>';
    } else {
      list.innerHTML = reports.map(r => `
        <div class="report-item" onclick="loadReport('${r.week_num}', this)">
          <div>
            <strong>Week ${r.week_num}</strong>
            <div style="font-size:11px;color:var(--text-muted)">${r.filename}</div>
          </div>
          <span style="font-size:12px;color:var(--text-muted)">Click to read →</span>
        </div>`).join('');
    }

    // Accuracy breakdown
    const acc = sig.accuracy_report || {};
    const tbody = document.getElementById('accuracy-body');
    const bySymbol = acc.by_symbol || acc.per_symbol || {};
    const globalRow = {
      '1d': acc['1d'] ?? acc.accuracy_1d,
      '3d': acc['3d'] ?? acc.accuracy_3d,
      '5d': acc['5d'] ?? acc.accuracy_5d,
      '10d': acc['10d'] ?? acc.accuracy_10d,
    };

    function acc_td(v) {
      if (v === null || v === undefined) return '<td>—</td>';
      const pct = Number(v);
      const cls = pct >= 65 ? 'acc-good' : pct >= 50 ? 'acc-ok' : 'acc-bad';
      return `<td class="${cls}">${pct.toFixed(1)}%</td>`;
    }

    let rows = '';
    if (Object.values(globalRow).some(v => v !== undefined && v !== null)) {
      rows += `<tr>
        <td><strong>Overall</strong></td>
        ${acc_td(globalRow['1d'])}${acc_td(globalRow['3d'])}${acc_td(globalRow['5d'])}${acc_td(globalRow['10d'])}
      </tr>`;
    }
    Object.entries(bySymbol).slice(0,20).forEach(([sym,v]) => {
      rows += `<tr>
        <td>${sym}</td>
        ${acc_td(v['1d']??v.d1)}${acc_td(v['3d']??v.d3)}${acc_td(v['5d']??v.d5)}${acc_td(v['10d']??v.d10)}
      </tr>`;
    });
    tbody.innerHTML = rows || empty_row(5, 'No accuracy data yet');

  } catch(e) { console.error('Reports load failed:', e); }
}

async function loadReport(weekNum, el) {
  document.querySelectorAll('.report-item').forEach(i => i.style.borderColor='');
  if (el) el.style.borderColor = 'var(--green)';
  const viewer = document.getElementById('report-viewer');
  viewer.style.display = 'block';
  viewer.innerHTML = '<div class="loading-overlay"><span class="spinner"></span> Loading report...</div>';
  try {
    const d = await api(`/api/report/${weekNum}`);
    viewer.innerHTML = marked.parse(d.content || '*(empty report)*');
    viewer.scrollIntoView({behavior:'smooth', block:'nearest'});
  } catch(e) {
    viewer.innerHTML = '<div class="empty-state">Report not found</div>';
  }
}

async function genReport() {
  const btn = document.getElementById('btn-gen-report');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  try {
    await fetch('/api/weekly-report', {method:'POST'});
    alert('Weekly report generation started in background. Refresh in a few minutes.');
  } catch(e) { alert('Failed: ' + e.message); }
  btn.disabled = false;
  btn.textContent = '📄 Generate Report Now';
}

// ══════════════════════════════════════════════════════════════════
//  Initial load
// ══════════════════════════════════════════════════════════════════
loadOverview();
loaded['overview'] = true;
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(content=HTML)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="NEPSE MiroFish Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    print(f"\n  NEPSE MiroFish Dashboard")
    print(f"  http://{args.host}:{args.port}\n")

    uvicorn.run(
        "dashboard.app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
