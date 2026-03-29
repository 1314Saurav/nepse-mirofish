"""
dashboard/app.py
NEPSE MiroFish — Bloomberg Terminal-Style Web Dashboard (FastAPI)

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

app = FastAPI(title="NEPSE MiroFish Dashboard", version="2.0.0")

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
            snap_dir = sd
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

    capital = float(state.get("capital", 1_000_000))
    starting = float(state.get("starting_capital", 1_000_000))
    positions = state.get("positions", {})
    pos_value = sum(
        float(p.get("entry_price", 0)) * int(p.get("qty", 0))
        for p in positions.values()
    )
    portfolio_value = capital + pos_value
    total_return_pct = (portfolio_value / starting - 1) * 100 if starting else 0.0

    today_pnl = 0.0
    if len(snaps) >= 2:
        today_pnl = snaps[-1].get("return_pct", 0.0) - snaps[-2].get("return_pct", 0.0)
    elif snaps:
        today_pnl = snaps[-1].get("return_pct", 0.0)

    regime = "UNKNOWN"
    mf_score = 0.0
    if snaps:
        regime = snaps[-1].get("regime", "SIDEWAYS")
        mf_score = snaps[-1].get("mirofish_score", 0.0)

    total_correct = accuracy.get("total_correct", 0)
    total_evaluated = accuracy.get("total_evaluated", 1)
    signal_accuracy = round(total_correct / max(total_evaluated, 1) * 100, 1)

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

    # Accuracy by horizon
    acc_by_horizon = {}
    for key in ["1d", "3d", "5d", "10d"]:
        val = accuracy.get(f"accuracy_{key}", accuracy.get(key, None))
        if val is not None:
            acc_by_horizon[key] = round(float(val) * 100 if float(val) <= 1 else float(val), 1)
        else:
            acc_by_horizon[key] = None

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
        "accuracy_by_horizon": acc_by_horizon,
        "top_signals": top_signals,
        "notifications": notifs,
        "session_id": state.get("paper_trade_id", "unknown"),
        "open_positions": len(positions),
        "cash": round(capital, 2),
        "starting_capital": round(starting, 2),
        "invested": round(pos_value, 2),
    }


@app.get("/api/watchlist")
async def api_watchlist():
    watchlist = _load_latest_watchlist()
    snaps = _load_snapshots()

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

    positions = []
    total_invested = 0.0
    for sym, pos in positions_raw.items():
        ep = float(pos.get("entry_price", 0))
        qty = int(pos.get("qty", 0))
        invested = ep * qty
        total_invested += invested
        current_price = ep
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

    equity_curve = [
        {
            "date": s.get("date", ""),
            "portfolio_value": round(float(s.get("portfolio_value", starting)), 2),
            "cash": round(float(s.get("cash", capital)), 2),
        }
        for s in snaps
    ]

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
# Bloomberg Terminal HTML dashboard
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MIROFISH▸ NEPSE TERMINAL</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ═══════════════════════════════════════════════════════════════════════
   BLOOMBERG TERMINAL — BASE RESET & VARIABLES
═══════════════════════════════════════════════════════════════════════ */
:root {
  --black:   #000000;
  --amber:   #FF9500;
  --green:   #00FF41;
  --red:     #FF3131;
  --white:   #FFFFFF;
  --cyan:    #00FFFF;
  --gray:    #333333;
  --gray2:   #222222;
  --gray3:   #111111;
  --amber-dim: #CC7700;
  --green-dim: #00BB33;
  --red-dim:   #CC2222;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  overflow: hidden;
  background: #000;
}

body {
  font-family: 'IBM Plex Mono', 'Courier New', monospace;
  font-size: 11px;
  color: var(--amber);
  background: var(--black);
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

/* ═══════════════════════════════════════════════════════════════════════
   SCROLLBARS
═══════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #444; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }

/* ═══════════════════════════════════════════════════════════════════════
   TOP STATUS BAR
═══════════════════════════════════════════════════════════════════════ */
#top-bar {
  position: sticky;
  top: 0;
  z-index: 200;
  background: #000;
  border-bottom: 1px solid var(--gray);
  padding: 3px 8px;
  display: flex;
  align-items: center;
  gap: 0;
  flex-shrink: 0;
  height: 22px;
  overflow: hidden;
  white-space: nowrap;
}

#top-bar .logo {
  color: var(--cyan);
  font-weight: 700;
  font-size: 12px;
  margin-right: 10px;
  letter-spacing: 1px;
}

#top-bar .separator {
  color: var(--gray);
  margin: 0 6px;
}

#top-bar .label { color: var(--white); font-size: 10px; }
#top-bar .val-up { color: var(--green); font-size: 10px; }
#top-bar .val-down { color: var(--red); font-size: 10px; }
#top-bar .val-neutral { color: var(--amber); font-size: 10px; }
#top-bar .clock { color: var(--cyan); font-size: 10px; font-weight: 700; }
#top-bar .market-open  { color: var(--green); font-size: 10px; }
#top-bar .market-closed { color: var(--red); font-size: 10px; }
#top-bar .spacer { flex: 1; }
#top-bar .refresh-info { color: var(--gray2); font-size: 10px; color: #555; }

/* ═══════════════════════════════════════════════════════════════════════
   FUNCTION KEY BAR
═══════════════════════════════════════════════════════════════════════ */
#fkey-bar {
  background: var(--gray3);
  border-bottom: 1px solid var(--gray);
  padding: 2px 8px;
  display: flex;
  align-items: center;
  gap: 2px;
  flex-shrink: 0;
  height: 20px;
  overflow: hidden;
}

.fkey {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  padding: 0 6px;
  height: 16px;
  border: 1px solid #444;
  background: #111;
  color: var(--white);
  font-family: inherit;
  font-size: 10px;
  font-weight: 700;
  transition: background 0.1s, color 0.1s;
  user-select: none;
  letter-spacing: 0.3px;
}

.fkey:hover { background: var(--amber); color: #000; border-color: var(--amber); }
.fkey .fnum { color: var(--amber); font-size: 9px; }
.fkey:hover .fnum { color: #000; }
.fkey-run { border-color: var(--green); color: var(--green); }
.fkey-run:hover { background: var(--green); color: #000; border-color: var(--green); }
.fkey-separator { width: 1px; height: 14px; background: #333; margin: 0 4px; }

/* ═══════════════════════════════════════════════════════════════════════
   MAIN GRID
═══════════════════════════════════════════════════════════════════════ */
#main-grid {
  flex: 1;
  display: grid;
  grid-template-columns: 18% 1fr 1fr;
  grid-template-rows: 45vh 1fr;
  overflow: hidden;
  min-height: 0;
}

/* Panel borders */
#main-grid > * {
  border-right: 1px solid var(--gray);
  border-bottom: 1px solid var(--gray);
  overflow-y: auto;
  overflow-x: hidden;
  padding: 4px 6px;
}

#main-grid > *:last-child { border-right: none; }

/* ═══════════════════════════════════════════════════════════════════════
   PANEL HEADERS
═══════════════════════════════════════════════════════════════════════ */
.panel-header {
  color: var(--white);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  border-bottom: 1px solid var(--gray);
  padding-bottom: 3px;
  margin-bottom: 5px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}

.panel-header .ph-title { color: var(--cyan); }
.panel-header .ph-sub { color: #555; font-size: 9px; font-weight: 400; }

.sub-header {
  color: var(--white);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.8px;
  border-bottom: 1px solid #222;
  padding: 3px 0 2px;
  margin: 6px 0 3px;
  color: #888;
}

/* ═══════════════════════════════════════════════════════════════════════
   TABLE STYLES
═══════════════════════════════════════════════════════════════════════ */
.term-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 10px;
}

.term-table th {
  color: var(--white);
  font-weight: 700;
  font-size: 9px;
  letter-spacing: 0.5px;
  padding: 1px 3px;
  text-align: right;
  border-bottom: 1px solid #333;
}

.term-table th:first-child { text-align: left; }

.term-table td {
  color: var(--amber);
  padding: 1px 3px;
  text-align: right;
  white-space: nowrap;
  font-size: 10px;
}

.term-table td:first-child { text-align: left; }
.term-table tr:hover td { background: #111; }

.td-sym { color: var(--white); font-weight: 700; font-size: 10px; }
.td-up  { color: var(--green); }
.td-down { color: var(--red); }
.td-buy  { color: var(--green); font-weight: 700; }
.td-sell { color: var(--red); font-weight: 700; }
.td-hold { color: var(--amber); }
.td-watch { color: #888; }
.td-cyan { color: var(--cyan); }
.td-white { color: var(--white); }
.td-muted { color: #555; font-size: 9px; }

/* ═══════════════════════════════════════════════════════════════════════
   PANEL 1 — MARKET MOVERS
═══════════════════════════════════════════════════════════════════════ */
#panel-movers {
  grid-column: 1;
  grid-row: 1;
}

/* Regime bar */
.regime-bar-wrap {
  margin: 2px 0;
}
.regime-bar-track {
  display: flex;
  align-items: center;
  gap: 4px;
  margin: 2px 0;
}
.regime-bar-fill {
  font-size: 10px;
  letter-spacing: 0;
  line-height: 1;
}
.regime-label {
  font-size: 10px;
  font-weight: 700;
  color: var(--white);
}
.regime-bull  { color: var(--green); }
.regime-bear  { color: var(--red); }
.regime-sideways { color: var(--amber); }
.regime-unknown  { color: #555; }

/* Accuracy bars */
.acc-row {
  display: flex;
  align-items: center;
  gap: 4px;
  margin: 2px 0;
  font-size: 10px;
}
.acc-label { color: #888; width: 22px; flex-shrink: 0; font-size: 9px; }
.acc-bar { color: var(--green); font-size: 10px; letter-spacing: -1px; }
.acc-val { color: var(--amber); font-size: 9px; margin-left: 2px; }

/* Score meter */
.score-row {
  font-size: 10px;
  color: #888;
  margin: 1px 0;
}
.score-row span { color: var(--amber); }

/* ═══════════════════════════════════════════════════════════════════════
   PANEL 2 — PORTFOLIO SUMMARY
═══════════════════════════════════════════════════════════════════════ */
#panel-portfolio {
  grid-column: 2;
  grid-row: 1;
}

.port-kpi-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px 8px;
  margin-bottom: 6px;
}

.port-kpi {
  padding: 3px 0;
  border-bottom: 1px solid #1a1a1a;
}

.port-kpi .kpi-label {
  color: #666;
  font-size: 9px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.port-kpi .kpi-val {
  color: var(--amber);
  font-size: 11px;
  font-weight: 700;
  margin-top: 1px;
}

.port-kpi .kpi-val.val-up { color: var(--green); }
.port-kpi .kpi-val.val-down { color: var(--red); }
.port-kpi .kpi-sub {
  color: #555;
  font-size: 9px;
  margin-top: 0;
}

.port-wide {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 2px 0;
  border-bottom: 1px solid #1a1a1a;
  font-size: 10px;
}
.port-wide .pw-label { color: #666; font-size: 9px; letter-spacing: 0.5px; }
.port-wide .pw-val { color: var(--amber); font-weight: 700; }
.port-wide .pw-val.val-up { color: var(--green); }
.port-wide .pw-val.val-down { color: var(--red); }
.port-wide .pw-pct { font-size: 9px; margin-left: 4px; }

/* ═══════════════════════════════════════════════════════════════════════
   PANEL 3 — TODAY'S SIGNALS
═══════════════════════════════════════════════════════════════════════ */
#panel-signals {
  grid-column: 3;
  grid-row: 1;
}

.signal-row-buy  { background: rgba(0,255,65,0.04); }
.signal-row-sell { background: rgba(255,49,49,0.04); }
.signal-row-hold { background: transparent; }

.conf-bar {
  display: inline-block;
  font-size: 9px;
  color: var(--amber);
  letter-spacing: -1px;
}

/* ═══════════════════════════════════════════════════════════════════════
   PANEL 4 — WATCHLIST
═══════════════════════════════════════════════════════════════════════ */
#panel-watchlist {
  grid-column: 1;
  grid-row: 2;
}

/* ═══════════════════════════════════════════════════════════════════════
   PANEL 5 — EQUITY CURVE
═══════════════════════════════════════════════════════════════════════ */
#panel-equity {
  grid-column: 2;
  grid-row: 2;
}

#equity-chart-wrap {
  position: relative;
  height: calc(100% - 30px);
  min-height: 120px;
}

/* ═══════════════════════════════════════════════════════════════════════
   PANEL 6 — AGENT SCORES
═══════════════════════════════════════════════════════════════════════ */
#panel-scores {
  grid-column: 3;
  grid-row: 2;
  border-right: none;
}

.score-bar-row {
  display: flex;
  align-items: center;
  gap: 4px;
  margin: 2px 0;
  font-size: 10px;
}
.sbr-sym {
  color: var(--white);
  font-weight: 700;
  width: 52px;
  flex-shrink: 0;
  font-size: 10px;
}
.sbr-bar {
  font-size: 10px;
  letter-spacing: -1px;
  min-width: 70px;
}
.sbr-val {
  font-size: 9px;
  margin-left: 2px;
  width: 32px;
  text-align: right;
}
.sbr-act {
  font-size: 9px;
  font-weight: 700;
  width: 32px;
  text-align: right;
}

/* ═══════════════════════════════════════════════════════════════════════
   TICKER BAR
═══════════════════════════════════════════════════════════════════════ */
#ticker-bar {
  flex-shrink: 0;
  height: 22px;
  background: #080808;
  border-top: 1px solid var(--gray);
  overflow: hidden;
  position: relative;
  display: flex;
  align-items: center;
}

#ticker-label {
  flex-shrink: 0;
  background: var(--amber);
  color: #000;
  font-size: 9px;
  font-weight: 700;
  padding: 0 6px;
  height: 100%;
  display: flex;
  align-items: center;
  letter-spacing: 1px;
  border-right: 1px solid #000;
}

#ticker-scroll-wrap {
  overflow: hidden;
  flex: 1;
  height: 100%;
  position: relative;
}

#ticker-inner {
  display: inline-flex;
  align-items: center;
  height: 100%;
  white-space: nowrap;
  animation: ticker-scroll 60s linear infinite;
  color: var(--amber);
  font-size: 10px;
}

@keyframes ticker-scroll {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

.tick-item {
  margin: 0 20px;
  color: var(--amber);
  font-size: 10px;
}
.tick-diamond { color: var(--cyan); margin: 0 8px; }

/* ═══════════════════════════════════════════════════════════════════════
   OVERLAY / MODAL
═══════════════════════════════════════════════════════════════════════ */
#help-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.92);
  z-index: 999;
  align-items: center;
  justify-content: center;
}
#help-overlay.active { display: flex; }

#help-box {
  background: #080808;
  border: 1px solid var(--amber);
  padding: 20px 28px;
  min-width: 380px;
  max-width: 500px;
}

#help-box h2 {
  color: var(--cyan);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 2px;
  margin-bottom: 14px;
  border-bottom: 1px solid #333;
  padding-bottom: 8px;
}

.help-row {
  display: flex;
  gap: 16px;
  margin: 4px 0;
  font-size: 10px;
}
.help-key {
  color: var(--white);
  font-weight: 700;
  width: 60px;
  flex-shrink: 0;
}
.help-desc { color: var(--amber); }

.help-close {
  margin-top: 16px;
  color: #555;
  font-size: 9px;
  cursor: pointer;
  text-align: center;
}
.help-close:hover { color: var(--amber); }

/* ═══════════════════════════════════════════════════════════════════════
   RUN INDICATOR
═══════════════════════════════════════════════════════════════════════ */
#run-indicator {
  display: none;
  position: fixed;
  top: 24px;
  right: 12px;
  background: #000;
  border: 1px solid var(--green);
  color: var(--green);
  font-size: 10px;
  padding: 4px 10px;
  z-index: 500;
  animation: blink-border 0.5s step-start infinite;
}
#run-indicator.active { display: block; }

@keyframes blink-border {
  50% { border-color: transparent; }
}

/* ═══════════════════════════════════════════════════════════════════════
   EMPTY STATE
═══════════════════════════════════════════════════════════════════════ */
.empty-state {
  color: #333;
  font-size: 10px;
  padding: 8px 0;
  text-align: center;
  letter-spacing: 1px;
}

/* ═══════════════════════════════════════════════════════════════════════
   MISC UTILITIES
═══════════════════════════════════════════════════════════════════════ */
.text-green  { color: var(--green); }
.text-red    { color: var(--red); }
.text-amber  { color: var(--amber); }
.text-cyan   { color: var(--cyan); }
.text-white  { color: var(--white); }
.text-muted  { color: #555; }
.text-right  { text-align: right; }
.fw-bold     { font-weight: 700; }
.fs-9        { font-size: 9px; }
.fs-10       { font-size: 10px; }
.fs-11       { font-size: 11px; }

.divider {
  border: none;
  border-top: 1px solid #1e1e1e;
  margin: 4px 0;
}
</style>
</head>
<body>

<!-- ══════════════════════════════════════════════════════════════════
     TOP STATUS BAR
══════════════════════════════════════════════════════════════════ -->
<div id="top-bar">
  <span class="logo">MIROFISH▸</span>
  <span class="separator">|</span>
  <span class="label">NEPSE:</span>&nbsp;
  <span class="val-neutral" id="tb-index">——</span>&nbsp;
  <span class="separator">|</span>
  <span class="label">NST:</span>&nbsp;
  <span class="clock" id="tb-clock">——:——:——</span>&nbsp;
  <span class="separator">|</span>
  <span id="tb-market-status" class="market-closed">● ——</span>
  <span class="separator">|</span>
  <span class="label">SESSION:</span>&nbsp;
  <span class="val-neutral" id="tb-session">——</span>
  <span class="separator">|</span>
  <span class="label">REGIME:</span>&nbsp;
  <span id="tb-regime" class="val-neutral">——</span>
  <span class="separator">|</span>
  <span class="label">SCORE:</span>&nbsp;
  <span id="tb-score" class="val-neutral">——</span>
  <div class="spacer"></div>
  <span class="refresh-info">AUTO-REFRESH: <span id="tb-countdown">30</span>s</span>
</div>

<!-- ══════════════════════════════════════════════════════════════════
     FUNCTION KEY BAR
══════════════════════════════════════════════════════════════════ -->
<div id="fkey-bar">
  <button class="fkey" onclick="scrollToPanel('panel-movers')" title="F1"><span class="fnum">F1</span> MKTW</button>
  <button class="fkey" onclick="scrollToPanel('panel-portfolio')" title="F2"><span class="fnum">F2</span> PORT</button>
  <button class="fkey" onclick="scrollToPanel('panel-signals')" title="F3"><span class="fnum">F3</span> SGNL</button>
  <button class="fkey" onclick="scrollToPanel('panel-movers')" title="F4"><span class="fnum">F4</span> REGM</button>
  <button class="fkey" onclick="scrollToPanel('panel-watchlist')" title="F5"><span class="fnum">F5</span> WTCH</button>
  <button class="fkey" onclick="scrollToPanel('panel-equity')" title="F6"><span class="fnum">F6</span> EQTY</button>
  <div class="fkey-separator"></div>
  <button class="fkey fkey-run" onclick="runCycle()" title="F7"><span class="fnum" style="color:inherit">F7</span> RUN NOW</button>
  <div class="fkey-separator"></div>
  <button class="fkey" onclick="showHelp()" title="F8"><span class="fnum">F8</span> HELP</button>
  <div class="fkey-separator"></div>
  <span style="color:#444;font-size:9px;margin-left:4px">LAST REFRESH: <span id="fk-last-refresh" style="color:#666">——</span></span>
</div>

<!-- ══════════════════════════════════════════════════════════════════
     MAIN PANEL GRID
══════════════════════════════════════════════════════════════════ -->
<div id="main-grid">

  <!-- ─── PANEL 1: MARKET MOVERS ─────────────────────────────────── -->
  <div id="panel-movers">
    <div class="panel-header">
      <span class="ph-title">TOP MOVERS</span>
      <span class="ph-sub" id="movers-date">——</span>
    </div>

    <table class="term-table" id="movers-table">
      <thead>
        <tr>
          <th>SYMBOL</th>
          <th>LAST</th>
          <th>CHG</th>
          <th>%</th>
        </tr>
      </thead>
      <tbody id="movers-body">
        <tr><td colspan="4" class="empty-state">NO DATA</td></tr>
      </tbody>
    </table>

    <div class="sub-header">─── REGIME ─────────────────</div>
    <div class="regime-bar-wrap" id="regime-section">
      <div class="regime-bar-track">
        <span class="regime-bar-fill" id="regime-bar-chars">░░░░░░░░░░</span>
        <span class="regime-label" id="regime-text">——</span>
      </div>
      <div class="score-row">Score: <span id="regime-score-val">——</span></div>
    </div>

    <div class="sub-header">─── ACCURACY ───────────────</div>
    <div id="accuracy-section">
      <div class="acc-row">
        <span class="acc-label">1d:</span>
        <span class="acc-bar" id="acc-bar-1d">░░░░░░░░░░</span>
        <span class="acc-val" id="acc-val-1d">—</span>
      </div>
      <div class="acc-row">
        <span class="acc-label">3d:</span>
        <span class="acc-bar" id="acc-bar-3d">░░░░░░░░░░</span>
        <span class="acc-val" id="acc-val-3d">—</span>
      </div>
      <div class="acc-row">
        <span class="acc-label">5d:</span>
        <span class="acc-bar" id="acc-bar-5d">░░░░░░░░░░</span>
        <span class="acc-val" id="acc-val-5d">—</span>
      </div>
      <div class="acc-row">
        <span class="acc-label">10d:</span>
        <span class="acc-bar" id="acc-bar-10d">░░░░░░░░░░</span>
        <span class="acc-val" id="acc-val-10d">—</span>
      </div>
    </div>
  </div><!-- /panel-movers -->

  <!-- ─── PANEL 2: PORTFOLIO SUMMARY ─────────────────────────────── -->
  <div id="panel-portfolio">
    <div class="panel-header">
      <span class="ph-title">PORTFOLIO</span>
      <span class="ph-sub" id="port-session-label">SESSION: ——</span>
    </div>

    <div class="port-wide">
      <span class="pw-label">TOTAL VALUE</span>
      <span>
        <span class="pw-val" id="port-total-val">NPR ——</span>
        <span class="pw-pct text-muted" id="port-total-pct"></span>
      </span>
    </div>
    <div class="port-wide">
      <span class="pw-label">CASH BALANCE</span>
      <span class="pw-val" id="port-cash">NPR ——</span>
    </div>
    <div class="port-wide">
      <span class="pw-label">INVESTED</span>
      <span class="pw-val" id="port-invested">NPR ——</span>
    </div>
    <div class="port-wide">
      <span class="pw-label">UNREALISED P&L</span>
      <span>
        <span class="pw-val" id="port-unrealised">NPR ——</span>
        <span class="pw-pct text-muted" id="port-unrealised-pct"></span>
      </span>
    </div>
    <div class="port-wide">
      <span class="pw-label">OPEN POSITIONS</span>
      <span>
        <span class="pw-val" id="port-open-pos">——</span>
        <span class="pw-pct text-muted"> / MAX 5</span>
      </span>
    </div>
    <div class="port-wide">
      <span class="pw-label">TODAY P&L</span>
      <span>
        <span class="pw-val" id="port-today-pnl">——</span>
        <span class="pw-pct text-muted" id="port-today-pnl-pct"></span>
      </span>
    </div>

    <div class="sub-header">─── OPEN POSITIONS ──────────────────────</div>
    <table class="term-table" id="positions-table">
      <thead>
        <tr>
          <th>SYM</th>
          <th>QTY</th>
          <th>ENTRY</th>
          <th>CURR</th>
          <th>P&L NPR</th>
          <th>%</th>
        </tr>
      </thead>
      <tbody id="positions-body">
        <tr><td colspan="6" class="empty-state">NO OPEN POSITIONS</td></tr>
      </tbody>
    </table>

    <div class="sub-header" style="margin-top:6px">─── RECENT TRADES ───────────────────────</div>
    <table class="term-table" id="trades-table">
      <thead>
        <tr>
          <th>DATE</th>
          <th>SYM</th>
          <th>ACT</th>
          <th>QTY</th>
          <th>PRICE</th>
          <th>P&L</th>
        </tr>
      </thead>
      <tbody id="trades-body">
        <tr><td colspan="6" class="empty-state">NO TRADE HISTORY</td></tr>
      </tbody>
    </table>
  </div><!-- /panel-portfolio -->

  <!-- ─── PANEL 3: TODAY'S SIGNALS ────────────────────────────────── -->
  <div id="panel-signals">
    <div class="panel-header">
      <span class="ph-title">SIGNALS</span>
      <span class="ph-sub" id="signals-date">——</span>
    </div>

    <table class="term-table" id="signals-table">
      <thead>
        <tr>
          <th>#</th>
          <th>SYM</th>
          <th>ACT</th>
          <th>SCORE</th>
          <th>TARGET</th>
          <th>STOP</th>
          <th>CONF</th>
        </tr>
      </thead>
      <tbody id="signals-body">
        <tr><td colspan="7" class="empty-state">RUN ANALYSIS TO GENERATE SIGNALS</td></tr>
      </tbody>
    </table>
  </div><!-- /panel-signals -->

  <!-- ─── PANEL 4: WATCHLIST ──────────────────────────────────────── -->
  <div id="panel-watchlist">
    <div class="panel-header">
      <span class="ph-title">WATCHLIST</span>
      <span class="ph-sub" id="wl-count">0 SYMBOLS</span>
    </div>

    <table class="term-table" id="watchlist-table">
      <thead>
        <tr>
          <th>SYM</th>
          <th>MF</th>
          <th>TECH</th>
          <th>COMB</th>
          <th>ACT</th>
          <th>REGIME</th>
        </tr>
      </thead>
      <tbody id="watchlist-body">
        <tr><td colspan="6" class="empty-state">NO WATCHLIST DATA</td></tr>
      </tbody>
    </table>
  </div><!-- /panel-watchlist -->

  <!-- ─── PANEL 5: EQUITY CURVE ───────────────────────────────────── -->
  <div id="panel-equity">
    <div class="panel-header">
      <span class="ph-title">EQUITY CURVE</span>
      <span class="ph-sub">PORTFOLIO vs BENCHMARK</span>
    </div>
    <div id="equity-chart-wrap">
      <canvas id="equity-chart"></canvas>
    </div>
  </div><!-- /panel-equity -->

  <!-- ─── PANEL 6: AGENT SCORES ───────────────────────────────────── -->
  <div id="panel-scores">
    <div class="panel-header">
      <span class="ph-title">MIROFISH AGENT SCORES</span>
      <span class="ph-sub">COMBINED</span>
    </div>
    <div id="scores-body">
      <div class="empty-state">NO SCORE DATA</div>
    </div>
  </div><!-- /panel-scores -->

</div><!-- /main-grid -->

<!-- ══════════════════════════════════════════════════════════════════
     BOTTOM TICKER
══════════════════════════════════════════════════════════════════ -->
<div id="ticker-bar">
  <div id="ticker-label">NEWS</div>
  <div id="ticker-scroll-wrap">
    <div id="ticker-inner" id="ticker-content">
      <span class="tick-item">◆ SYSTEM INITIALISING — FETCHING MARKET DATA</span>
      <span class="tick-diamond">◆</span>
      <span class="tick-item">◆ NEPSE MIROFISH TERMINAL v2.0 — BLOOMBERG STYLE INTERFACE</span>
      <span class="tick-diamond">◆</span>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════
     RUN INDICATOR
══════════════════════════════════════════════════════════════════ -->
<div id="run-indicator">● CYCLE RUNNING...</div>

<!-- ══════════════════════════════════════════════════════════════════
     HELP OVERLAY
══════════════════════════════════════════════════════════════════ -->
<div id="help-overlay">
  <div id="help-box">
    <h2>KEYBOARD SHORTCUTS</h2>
    <div class="help-row"><span class="help-key">F1</span><span class="help-desc">MKTW — Market Movers & Regime</span></div>
    <div class="help-row"><span class="help-key">F2</span><span class="help-desc">PORT — Portfolio Summary</span></div>
    <div class="help-row"><span class="help-key">F3</span><span class="help-desc">SGNL — Today's Signals</span></div>
    <div class="help-row"><span class="help-key">F4</span><span class="help-desc">REGM — Regime Detail</span></div>
    <div class="help-row"><span class="help-key">F5</span><span class="help-desc">WTCH — Watchlist</span></div>
    <div class="help-row"><span class="help-key">F6</span><span class="help-desc">EQTY — Equity Curve</span></div>
    <div class="help-row"><span class="help-key">F7</span><span class="help-desc">RUN NOW — Trigger analysis cycle</span></div>
    <div class="help-row"><span class="help-key">F8</span><span class="help-desc">HELP — This overlay</span></div>
    <div class="help-row"><span class="help-key">ESC</span><span class="help-desc">Close overlay</span></div>
    <div class="help-close" onclick="hideHelp()">[ PRESS ESC OR CLICK TO CLOSE ]</div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════
     JAVASCRIPT
══════════════════════════════════════════════════════════════════ -->
<script>
'use strict';

// ── State ────────────────────────────────────────────────────────────
let equityChart = null;
let countdownVal = 30;
let countdownTimer = null;
let refreshTimer = null;

// ── Utilities ────────────────────────────────────────────────────────

function fmtNPR(val) {
  if (val === null || val === undefined || isNaN(val)) return 'NPR ——';
  const n = parseFloat(val);
  return 'NPR ' + n.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}

function fmtNum(val, dec=2) {
  if (val === null || val === undefined || isNaN(val)) return '——';
  return parseFloat(val).toLocaleString('en-IN', {minimumFractionDigits: dec, maximumFractionDigits: dec});
}

function fmtPct(val) {
  if (val === null || val === undefined || isNaN(val)) return '';
  const n = parseFloat(val);
  const sign = n >= 0 ? '+' : '';
  return sign + n.toFixed(2) + '%';
}

function scoreBar(score, len=10) {
  const s = Math.max(0, Math.min(1, parseFloat(score) || 0));
  const filled = Math.round(s * len);
  return '█'.repeat(filled) + '░'.repeat(len - filled);
}

function scoreColor(score) {
  const s = parseFloat(score) || 0;
  if (s > 0.6) return '#00FF41';
  if (s >= 0.4) return '#FF9500';
  return '#FF3131';
}

function actionClass(action) {
  if (!action) return 'td-watch';
  const a = action.toUpperCase();
  if (a === 'BUY')  return 'td-buy';
  if (a === 'SELL') return 'td-sell';
  if (a === 'HOLD') return 'td-hold';
  return 'td-watch';
}

function pnlClass(val) {
  const n = parseFloat(val);
  if (n > 0)  return 'td-up';
  if (n < 0)  return 'td-down';
  return '';
}

function regimeClass(regime) {
  if (!regime) return 'regime-unknown';
  const r = regime.toUpperCase();
  if (r === 'BULL' || r === 'BULLISH') return 'regime-bull';
  if (r === 'BEAR' || r === 'BEARISH') return 'regime-bear';
  if (r === 'SIDEWAYS' || r === 'NEUTRAL') return 'regime-sideways';
  return 'regime-unknown';
}

function set(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

function setClass(id, cls) {
  const el = document.getElementById(id);
  if (el) el.className = cls;
}

// ── NST Clock ────────────────────────────────────────────────────────

function updateClock() {
  const now = new Date();
  // Convert to NST (UTC+5:45)
  const nst = new Date(now.getTime() + (5*60 + 45) * 60000 - now.getTimezoneOffset() * 60000);
  const h = String(nst.getUTCHours()).padStart(2,'0');
  const m = String(nst.getUTCMinutes()).padStart(2,'0');
  const s = String(nst.getUTCSeconds()).padStart(2,'0');
  setText('tb-clock', `${h}:${m}:${s}`);
}

setInterval(updateClock, 1000);
updateClock();

// ── Countdown ────────────────────────────────────────────────────────

function startCountdown() {
  countdownVal = 30;
  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = setInterval(() => {
    countdownVal--;
    setText('tb-countdown', countdownVal);
    if (countdownVal <= 0) {
      countdownVal = 30;
      fetchAll();
    }
  }, 1000);
}

// ── Panel scroll ─────────────────────────────────────────────────────

function scrollToPanel(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Help overlay ─────────────────────────────────────────────────────

function showHelp() {
  document.getElementById('help-overlay').classList.add('active');
}
function hideHelp() {
  document.getElementById('help-overlay').classList.remove('active');
}

document.addEventListener('keydown', (e) => {
  switch(e.key) {
    case 'F1': e.preventDefault(); scrollToPanel('panel-movers'); break;
    case 'F2': e.preventDefault(); scrollToPanel('panel-portfolio'); break;
    case 'F3': e.preventDefault(); scrollToPanel('panel-signals'); break;
    case 'F4': e.preventDefault(); scrollToPanel('panel-movers'); break;
    case 'F5': e.preventDefault(); scrollToPanel('panel-watchlist'); break;
    case 'F6': e.preventDefault(); scrollToPanel('panel-equity'); break;
    case 'F7': e.preventDefault(); runCycle(); break;
    case 'F8': e.preventDefault(); showHelp(); break;
    case 'Escape': hideHelp(); break;
  }
});

document.getElementById('help-overlay').addEventListener('click', (e) => {
  if (e.target === document.getElementById('help-overlay')) hideHelp();
});

// ── Run cycle ────────────────────────────────────────────────────────

async function runCycle() {
  const ind = document.getElementById('run-indicator');
  ind.classList.add('active');
  try {
    const r = await fetch('/api/run-cycle', { method: 'POST' });
    const d = await r.json();
    console.log('Run cycle:', d);
    setTimeout(() => {
      ind.classList.remove('active');
      fetchAll();
    }, 3000);
  } catch(err) {
    console.error(err);
    ind.classList.remove('active');
  }
}

// ── Data fetchers ────────────────────────────────────────────────────

async function fetchOverview() {
  try {
    const r = await fetch('/api/overview');
    const d = await r.json();
    renderOverview(d);
  } catch(e) { console.warn('overview fetch failed', e); }
}

async function fetchPortfolio() {
  try {
    const r = await fetch('/api/portfolio');
    const d = await r.json();
    renderPortfolio(d);
  } catch(e) { console.warn('portfolio fetch failed', e); }
}

async function fetchWatchlist() {
  try {
    const r = await fetch('/api/watchlist');
    const d = await r.json();
    renderWatchlist(d);
  } catch(e) { console.warn('watchlist fetch failed', e); }
}

async function fetchNotifications() {
  try {
    const r = await fetch('/api/notifications');
    const d = await r.json();
    renderTicker(d.notifications || []);
  } catch(e) { console.warn('notifications fetch failed', e); }
}

async function fetchAll() {
  const now = new Date();
  const ts = now.toTimeString().slice(0,8);
  setText('fk-last-refresh', ts);
  await Promise.allSettled([
    fetchOverview(),
    fetchPortfolio(),
    fetchWatchlist(),
    fetchNotifications(),
  ]);
}

// ── Renderers ────────────────────────────────────────────────────────

function renderOverview(d) {
  // Top bar
  const marketOpen = d.market_open;
  const statusEl = document.getElementById('tb-market-status');
  if (statusEl) {
    statusEl.textContent = marketOpen ? '● MARKET OPEN' : '● MARKET CLOSED';
    statusEl.className = marketOpen ? 'market-open' : 'market-closed';
  }

  setText('tb-session', d.session_id || 'demo');

  const regime = (d.regime || 'UNKNOWN').toUpperCase();
  const regEl = document.getElementById('tb-regime');
  if (regEl) {
    regEl.textContent = regime;
    regEl.className = regimeClass(regime);
  }

  const score = d.mirofish_score || 0;
  const scoreEl = document.getElementById('tb-score');
  if (scoreEl) {
    scoreEl.textContent = score.toFixed(3);
    scoreEl.style.color = scoreColor(score);
  }

  // Portfolio values in top bar — show NEPSE index if available
  const pvEl = document.getElementById('tb-index');
  if (pvEl) {
    const pv = d.portfolio_value || 0;
    const ret = d.total_return_pct || 0;
    const sign = ret >= 0 ? '▲+' : '▼';
    pvEl.textContent = `${fmtNum(pv, 0)} ${sign}${Math.abs(ret).toFixed(2)}%`;
    pvEl.className = ret >= 0 ? 'val-up' : 'val-down';
  }

  // Date
  const today = new Date();
  const dateStr = today.toISOString().slice(0,10);
  setText('movers-date', dateStr);
  setText('signals-date', dateStr);
  setText('port-session-label', `SESSION: ${d.session_id || 'demo'}`);

  // Movers from top_signals (use as proxy)
  const movers = (d.top_signals || []).slice(0, 8);
  const moversBody = document.getElementById('movers-body');
  if (moversBody) {
    if (movers.length === 0) {
      moversBody.innerHTML = '<tr><td colspan="4" class="empty-state">NO DATA</td></tr>';
    } else {
      moversBody.innerHTML = movers.map(s => {
        const act = (s.action || '').toUpperCase();
        const scr = parseFloat(s.score || 0);
        const tgt = s.target ? fmtNum(s.target, 0) : '——';
        const chgClass = act === 'BUY' ? 'td-up' : act === 'SELL' ? 'td-down' : '';
        const chgSign = act === 'BUY' ? '+' : act === 'SELL' ? '-' : '';
        return `<tr>
          <td class="td-sym">${s.symbol}</td>
          <td>${tgt}</td>
          <td class="${chgClass}">${chgSign}${(scr*10).toFixed(1)}</td>
          <td class="${chgClass}">${chgSign}${(scr*100).toFixed(2)}%</td>
        </tr>`;
      }).join('');
    }
  }

  // Regime
  const barLen = 10;
  const filled = Math.round(score * barLen);
  const barChars = '█'.repeat(filled) + '░'.repeat(barLen - filled);
  const regBarEl = document.getElementById('regime-bar-chars');
  if (regBarEl) {
    regBarEl.textContent = barChars;
    regBarEl.className = `regime-bar-fill ${regimeClass(regime)}`;
  }
  setText('regime-text', regime);
  document.getElementById('regime-text').className = `regime-label ${regimeClass(regime)}`;
  setText('regime-score-val', score.toFixed(3));
  document.getElementById('regime-score-val').style.color = scoreColor(score);

  // Accuracy
  const acc = d.accuracy_by_horizon || {};
  ['1d','3d','5d','10d'].forEach(h => {
    const v = acc[h];
    const barId = `acc-bar-${h}`;
    const valId = `acc-val-${h}`;
    if (v !== null && v !== undefined) {
      const pct = parseFloat(v);
      const filled = Math.round((pct/100) * 8);
      const bar = '█'.repeat(filled) + '░'.repeat(8 - filled);
      const barEl = document.getElementById(barId);
      if (barEl) {
        barEl.textContent = bar;
        barEl.style.color = pct >= 60 ? '#00FF41' : pct >= 50 ? '#FF9500' : '#FF3131';
      }
      setText(valId, pct.toFixed(0) + '%');
    } else {
      setText(barId, '░░░░░░░░');
      setText(valId, '—');
    }
  });

  // Signals table
  const signals = d.top_signals || [];
  const sigBody = document.getElementById('signals-body');
  if (sigBody) {
    if (signals.length === 0) {
      sigBody.innerHTML = '<tr><td colspan="7" class="empty-state">RUN ANALYSIS TO GENERATE SIGNALS</td></tr>';
    } else {
      sigBody.innerHTML = signals.map((s, i) => {
        const act = (s.action || '').toUpperCase();
        const sc = parseFloat(s.score || 0);
        const conf = s.confidence ? Math.round(parseFloat(s.confidence) * (parseFloat(s.confidence) <= 1 ? 100 : 1)) : Math.round(sc * 100);
        const tgt = s.target ? fmtNum(s.target, 0) : '——';
        const stp = s.stop_loss ? fmtNum(s.stop_loss, 0) : '——';
        const rowCls = act === 'BUY' ? 'signal-row-buy' : act === 'SELL' ? 'signal-row-sell' : 'signal-row-hold';
        const actCls = actionClass(act);
        const confFilled = Math.round(conf / 100 * 5);
        const confBar = '█'.repeat(confFilled) + '░'.repeat(5 - confFilled);
        return `<tr class="${rowCls}">
          <td class="td-muted">${i+1}</td>
          <td class="td-sym">${s.symbol}</td>
          <td class="${actCls}">${act}</td>
          <td style="color:${scoreColor(sc)}">${sc.toFixed(3)}</td>
          <td class="td-amber">${tgt}</td>
          <td class="td-muted">${stp}</td>
          <td><span class="conf-bar" style="color:${scoreColor(sc/1)}">${confBar}</span><span class="td-muted"> ${conf}%</span></td>
        </tr>`;
      }).join('');
    }
  }

  // Portfolio top-level (partial from overview)
  const portTotalEl = document.getElementById('port-total-val');
  const portTotalPctEl = document.getElementById('port-total-pct');
  if (portTotalEl) {
    portTotalEl.textContent = fmtNPR(d.portfolio_value);
    const ret = d.total_return_pct || 0;
    portTotalEl.className = `pw-val ${ret >= 0 ? 'val-up' : 'val-down'}`;
  }
  if (portTotalPctEl) {
    const ret = d.total_return_pct || 0;
    portTotalPctEl.textContent = fmtPct(ret);
    portTotalPctEl.className = `pw-pct ${ret >= 0 ? 'text-green' : 'text-red'}`;
  }

  const cashEl = document.getElementById('port-cash');
  if (cashEl) { cashEl.textContent = fmtNPR(d.cash); }

  const invEl = document.getElementById('port-invested');
  if (invEl) { invEl.textContent = fmtNPR(d.invested); }

  const openPosEl = document.getElementById('port-open-pos');
  if (openPosEl) { openPosEl.textContent = d.open_positions || 0; }

  const todayEl = document.getElementById('port-today-pnl');
  const todayPctEl = document.getElementById('port-today-pnl-pct');
  if (todayEl) {
    const tp = d.today_pnl_pct || 0;
    todayEl.textContent = fmtPct(tp);
    todayEl.className = `pw-val ${tp >= 0 ? 'val-up' : 'val-down'}`;
  }
}

function renderPortfolio(d) {
  // Unrealised P&L
  const unrlEl = document.getElementById('port-unrealised');
  const unrlPctEl = document.getElementById('port-unrealised-pct');
  if (unrlEl) {
    const upnl = d.unrealised_pnl || 0;
    unrlEl.textContent = fmtNPR(upnl);
    unrlEl.className = `pw-val ${upnl >= 0 ? 'val-up' : 'val-down'}`;
  }

  // Positions table
  const posBody = document.getElementById('positions-body');
  const positions = d.positions || [];
  if (posBody) {
    if (positions.length === 0) {
      posBody.innerHTML = '<tr><td colspan="6" class="empty-state">NO OPEN POSITIONS</td></tr>';
    } else {
      posBody.innerHTML = positions.map(p => {
        const pnlCls = pnlClass(p.pnl_pct);
        return `<tr>
          <td class="td-sym">${p.symbol}</td>
          <td>${p.qty}</td>
          <td class="td-amber">${fmtNum(p.entry_price, 0)}</td>
          <td class="td-cyan">${fmtNum(p.current_price, 0)}</td>
          <td class="${pnlCls}">${fmtNum(p.pnl_npr, 0)}</td>
          <td class="${pnlCls}">${fmtPct(p.pnl_pct)}</td>
        </tr>`;
      }).join('');
    }
  }

  // Trade history
  const tradesBody = document.getElementById('trades-body');
  const trades = (d.trade_history || []).reverse().slice(0, 10);
  if (tradesBody) {
    if (trades.length === 0) {
      tradesBody.innerHTML = '<tr><td colspan="6" class="empty-state">NO TRADE HISTORY</td></tr>';
    } else {
      tradesBody.innerHTML = trades.map(t => {
        const act = (t.action || t.type || '').toUpperCase();
        const actCls = actionClass(act);
        const pnl = t.pnl || t.realized_pnl || 0;
        const pnlCls = pnlClass(pnl);
        return `<tr>
          <td class="td-muted">${(t.date || t.timestamp || '').slice(0,10)}</td>
          <td class="td-sym">${t.symbol || '——'}</td>
          <td class="${actCls}">${act}</td>
          <td>${t.qty || '——'}</td>
          <td class="td-amber">${t.price ? fmtNum(t.price, 0) : '——'}</td>
          <td class="${pnlCls}">${pnl ? fmtNum(pnl, 0) : '——'}</td>
        </tr>`;
      }).join('');
    }
  }

  // Equity curve
  const curve = d.equity_curve || [];
  renderEquityChart(curve);
}

function renderWatchlist(d) {
  const wl = d.watchlist || [];
  setText('wl-count', `${wl.length} SYMBOLS`);

  // Watchlist table
  const wlBody = document.getElementById('watchlist-body');
  if (wlBody) {
    if (wl.length === 0) {
      wlBody.innerHTML = '<tr><td colspan="6" class="empty-state">NO WATCHLIST DATA</td></tr>';
    } else {
      wlBody.innerHTML = wl.map(w => {
        const sym = w.symbol || '——';
        const mf = parseFloat(w.mirofish_score || w.mf_score || w.score || 0);
        const tech = parseFloat(w.technical_score || w.tech_score || w.tech || 0);
        const comb = parseFloat(w.combined_score || w.score || mf);
        const act = (w.action || 'WATCH').toUpperCase();
        const regime = (w.regime || w.market_regime || '——').toUpperCase();
        const actCls = actionClass(act);
        const regCls = regimeClass(regime);
        return `<tr>
          <td class="td-sym">${sym}</td>
          <td style="color:${scoreColor(mf)}">${mf.toFixed(2)}</td>
          <td style="color:${scoreColor(tech)}">${tech.toFixed(2)}</td>
          <td style="color:${scoreColor(comb)}">${comb.toFixed(2)}</td>
          <td class="${actCls}">${act}</td>
          <td class="${regCls} fs-9">${regime}</td>
        </tr>`;
      }).join('');
    }
  }

  // Agent scores panel
  const scoresBody = document.getElementById('scores-body');
  if (scoresBody) {
    const top = wl.slice(0, 15);
    if (top.length === 0) {
      scoresBody.innerHTML = '<div class="empty-state">NO SCORE DATA</div>';
    } else {
      scoresBody.innerHTML = top.map(w => {
        const sym = w.symbol || '——';
        const score = parseFloat(w.combined_score || w.score || w.mirofish_score || 0);
        const act = (w.action || 'WATCH').toUpperCase();
        const bar = scoreBar(score, 10);
        const clr = scoreColor(score);
        const actCls = actionClass(act);
        return `<div class="score-bar-row">
          <span class="sbr-sym">${sym}</span>
          <span class="sbr-bar" style="color:${clr}">${bar}</span>
          <span class="sbr-val" style="color:${clr}">${score.toFixed(2)}</span>
          <span class="sbr-act ${actCls}">${act}</span>
        </div>`;
      }).join('');
    }
  }
}

function renderTicker(notifications) {
  const inner = document.getElementById('ticker-inner');
  if (!inner) return;
  if (notifications.length === 0) {
    inner.innerHTML = `
      <span class="tick-item">◆ NEPSE MIROFISH TERMINAL — AWAITING DATA</span>
      <span class="tick-diamond">◆</span>
      <span class="tick-item">◆ USE F7 TO RUN ANALYSIS CYCLE</span>
      <span class="tick-diamond">◆</span>
      <span class="tick-item">◆ NEPSE MIROFISH TERMINAL — AWAITING DATA</span>
      <span class="tick-diamond">◆</span>
      <span class="tick-item">◆ USE F7 TO RUN ANALYSIS CYCLE</span>
    `;
    return;
  }
  const items = notifications.slice(-30).reverse();
  const html = items.map(n => {
    const ts = n.ts ? String(n.ts).slice(11,16) : '';
    const msg = n.msg || n.message || JSON.stringify(n);
    return `<span class="tick-item">◆ ${ts ? ts + ' ' : ''}${msg}</span>`;
  }).join('<span class="tick-diamond"> ◆ </span>');
  // Duplicate for seamless loop
  inner.innerHTML = html + '<span class="tick-diamond"> ◆◆◆ </span>' + html;
}

// ── Equity Chart ─────────────────────────────────────────────────────

function renderEquityChart(curve) {
  const canvas = document.getElementById('equity-chart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');

  let labels, portfolioData;

  if (curve.length === 0) {
    // Demo flat line
    labels = ['Start', '1W', '2W', '3W', '4W', 'Now'];
    portfolioData = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000];
  } else {
    labels = curve.map(c => c.date ? String(c.date).slice(5) : '');
    portfolioData = curve.map(c => c.portfolio_value);
  }

  // Benchmark (flat starting capital or 1M)
  const startVal = portfolioData[0] || 1000000;
  const benchmarkData = portfolioData.map(() => startVal);

  if (equityChart) {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = portfolioData;
    equityChart.data.datasets[1].data = benchmarkData;
    equityChart.update('none');
    return;
  }

  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'PORTFOLIO',
          data: portfolioData,
          borderColor: '#00FF41',
          backgroundColor: 'rgba(0,255,65,0.05)',
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 3,
          fill: true,
          tension: 0.3,
        },
        {
          label: 'BENCHMARK',
          data: benchmarkData,
          borderColor: '#555555',
          backgroundColor: 'transparent',
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: {
          display: true,
          labels: {
            color: '#666',
            font: { family: "'IBM Plex Mono', monospace", size: 9 },
            boxWidth: 12,
            padding: 8,
          },
        },
        tooltip: {
          backgroundColor: '#111',
          borderColor: '#333',
          borderWidth: 1,
          titleColor: '#fff',
          bodyColor: '#FF9500',
          titleFont: { family: "'IBM Plex Mono', monospace", size: 9 },
          bodyFont: { family: "'IBM Plex Mono', monospace", size: 9 },
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: NPR ${ctx.parsed.y.toLocaleString('en-IN', {minimumFractionDigits: 0, maximumFractionDigits: 0})}`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#444',
            font: { family: "'IBM Plex Mono', monospace", size: 8 },
            maxTicksLimit: 8,
            maxRotation: 0,
          },
          grid: { color: '#111', drawBorder: false },
          border: { color: '#333' },
        },
        y: {
          ticks: {
            color: '#444',
            font: { family: "'IBM Plex Mono', monospace", size: 8 },
            callback: (v) => 'NPR ' + (v/1000).toFixed(0) + 'K',
          },
          grid: { color: '#111', drawBorder: false },
          border: { color: '#333' },
        },
      },
    },
  });
}

// ── Initialise ───────────────────────────────────────────────────────

async function init() {
  // Initial render with demo data
  renderEquityChart([]);
  renderTicker([]);
  // Fetch real data
  await fetchAll();
  startCountdown();
}

init();
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

    p = argparse.ArgumentParser(description="NEPSE MiroFish Bloomberg Terminal Dashboard")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    logger.info("Starting NEPSE MiroFish Terminal on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
