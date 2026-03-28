"""
paper_trading/live_dashboard.py
FastAPI web dashboard for paper trading session monitoring.
Run: python -m paper_trading.live_dashboard
URL: http://localhost:8080
Auto-refreshes every 30 seconds.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML template (dark professional theme, CSS grid layout)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>MiroFish Paper Trading Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #30363d;
      --text:     #e6edf3;
      --subtext:  #8b949e;
      --green:    #3fb950;
      --red:      #f85149;
      --yellow:   #d29922;
      --blue:     #58a6ff;
      --purple:   #bc8cff;
      --accent:   #1f6feb;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      min-height: 100vh;
    }

    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    header .logo {
      font-size: 18px;
      font-weight: 700;
      color: var(--blue);
      letter-spacing: 0.5px;
    }

    header .meta {
      font-size: 12px;
      color: var(--subtext);
    }

    #refresh-badge {
      display: inline-block;
      background: var(--accent);
      color: #fff;
      border-radius: 12px;
      padding: 3px 10px;
      font-size: 11px;
      margin-left: 10px;
    }

    main {
      padding: 20px 24px;
      max-width: 1440px;
      margin: 0 auto;
    }

    /* ── KPI cards ── */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px 20px;
    }

    .card .label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--subtext);
      margin-bottom: 8px;
    }

    .card .value {
      font-size: 26px;
      font-weight: 700;
      line-height: 1;
    }

    .card .sub {
      font-size: 12px;
      color: var(--subtext);
      margin-top: 4px;
    }

    .pos  { color: var(--green); }
    .neg  { color: var(--red);   }
    .neu  { color: var(--text);  }

    /* ── Regime badge ── */
    .regime-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 6px;
      padding: 6px 14px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }
    .regime-BULL     { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid var(--green); }
    .regime-BEAR     { background: rgba(248,81,73,0.15); color: var(--red);   border: 1px solid var(--red);   }
    .regime-SIDEWAYS { background: rgba(210,153,34,0.15); color: var(--yellow); border: 1px solid var(--yellow); }
    .regime-UNKNOWN  { background: rgba(139,148,158,0.15); color: var(--subtext); border: 1px solid var(--border); }

    /* ── Grid layout ── */
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }

    .grid-3 {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }

    @media (max-width: 900px) {
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
    }

    /* ── Section titles ── */
    .section-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--subtext);
      margin-bottom: 12px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
    }

    /* ── Tables ── */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th {
      text-align: left;
      padding: 8px 10px;
      color: var(--subtext);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }

    td {
      padding: 9px 10px;
      border-bottom: 1px solid rgba(48,54,61,0.5);
      vertical-align: middle;
    }

    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255,255,255,0.03); }

    .no-data {
      text-align: center;
      padding: 28px;
      color: var(--subtext);
      font-size: 13px;
    }

    /* ── Chart container ── */
    .chart-container {
      position: relative;
      height: 240px;
    }

    /* ── Accuracy stats ── */
    .acc-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 6px;
    }

    .acc-cell {
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      text-align: center;
    }

    .acc-cell .win { font-size: 11px; color: var(--subtext); margin-bottom: 4px; }
    .acc-cell .pct { font-size: 20px; font-weight: 700; }

    /* ── Status dot ── */
    .dot {
      display: inline-block;
      width: 8px; height: 8px;
      border-radius: 50%;
      margin-right: 5px;
    }
    .dot-green  { background: var(--green); }
    .dot-red    { background: var(--red); }
    .dot-yellow { background: var(--yellow); }

    /* ── Spinner ── */
    #spinner {
      position: fixed;
      top: 14px; right: 24px;
      width: 16px; height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<div id="spinner"></div>

<header>
  <div class="logo">MiroFish Paper Trading</div>
  <div class="meta">
    <span id="session-label">Loading…</span>
    <span id="refresh-badge">Refreshes in <span id="countdown">30</span>s</span>
  </div>
</header>

<main>
  <!-- KPI row -->
  <div class="kpi-row" id="kpi-row">
    <div class="card">
      <div class="label">Portfolio Value</div>
      <div class="value neu" id="kpi-pv">—</div>
      <div class="sub" id="kpi-return">—</div>
    </div>
    <div class="card">
      <div class="label">Cash Balance</div>
      <div class="value neu" id="kpi-cash">—</div>
      <div class="sub" id="kpi-cash-pct">—</div>
    </div>
    <div class="card">
      <div class="label">Daily P&amp;L</div>
      <div class="value neu" id="kpi-dpnl">—</div>
      <div class="sub" id="kpi-dpnl-sub">—</div>
    </div>
    <div class="card">
      <div class="label">Open Positions</div>
      <div class="value neu" id="kpi-pos">—</div>
      <div class="sub" id="kpi-days">—</div>
    </div>
    <div class="card" style="display:flex;align-items:center;gap:10px;">
      <div>
        <div class="label">MiroFish Regime</div>
        <div id="regime-badge" class="regime-badge regime-UNKNOWN">— UNKNOWN</div>
      </div>
    </div>
  </div>

  <!-- Equity curve + Watchlist -->
  <div class="grid-3">
    <div class="card">
      <div class="section-title">Equity Curve (NPR)</div>
      <div class="chart-container">
        <canvas id="equityChart"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="section-title">Top Watchlist Buys</div>
      <table id="watchlist-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Symbol</th>
            <th>Score</th>
            <th>Tier</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody id="watchlist-body">
          <tr><td colspan="5" class="no-data">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Positions table -->
  <div class="card" style="margin-bottom:20px;">
    <div class="section-title">Open Positions</div>
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th>Qty</th>
          <th>Entry Price</th>
          <th>Current Price</th>
          <th>P&amp;L %</th>
          <th>Stop Loss</th>
          <th>Target</th>
          <th>Strategy</th>
          <th>Regime</th>
        </tr>
      </thead>
      <tbody id="positions-body">
        <tr><td colspan="9" class="no-data">Loading…</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Accuracy stats -->
  <div class="card">
    <div class="section-title">Signal Accuracy Stats</div>
    <div class="acc-grid" id="acc-grid">
      <div class="acc-cell"><div class="win">1d</div><div class="pct" id="acc-1d">—</div></div>
      <div class="acc-cell"><div class="win">3d</div><div class="pct" id="acc-3d">—</div></div>
      <div class="acc-cell"><div class="win">5d</div><div class="pct" id="acc-5d">—</div></div>
      <div class="acc-cell"><div class="win">10d (last 10)</div><div class="pct" id="acc-10d">—</div></div>
    </div>
    <div style="margin-top:12px;font-size:12px;color:var(--subtext);" id="acc-meta">—</div>
  </div>
</main>

<script>
  // ── Equity chart ──────────────────────────────────────────────────────────
  const ctx = document.getElementById('equityChart').getContext('2d');
  let equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Portfolio Value (NPR)',
        data: [],
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.08)',
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 5,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => 'NPR ' + ctx.parsed.y.toLocaleString('en-IN', {maximumFractionDigits: 0})
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#8b949e', maxTicksLimit: 8, font: { size: 11 } },
          grid:  { color: 'rgba(48,54,61,0.5)' }
        },
        y: {
          ticks: {
            color: '#8b949e',
            font: { size: 11 },
            callback: v => 'NPR ' + (v / 1000).toFixed(0) + 'k'
          },
          grid: { color: 'rgba(48,54,61,0.5)' }
        }
      }
    }
  });

  // ── Helpers ───────────────────────────────────────────────────────────────
  function fmt(n) {
    if (n == null) return '—';
    return 'NPR ' + Number(n).toLocaleString('en-IN', {maximumFractionDigits: 0});
  }
  function fmtPct(n, showSign=true) {
    if (n == null) return '—';
    const s = showSign && n > 0 ? '+' : '';
    return s + Number(n).toFixed(2) + '%';
  }
  function colorClass(n) {
    if (n == null) return 'neu';
    return n > 0 ? 'pos' : (n < 0 ? 'neg' : 'neu');
  }
  function accColor(v) {
    if (v == null) return 'var(--subtext)';
    if (v >= 60)   return 'var(--green)';
    if (v >= 45)   return 'var(--yellow)';
    return 'var(--red)';
  }

  // ── Fetch & render ────────────────────────────────────────────────────────
  async function loadAll() {
    document.getElementById('spinner').style.display = 'block';
    try {
      const [state, positions, accuracy, equity, watchlist] = await Promise.all([
        fetch('/api/state').then(r => r.json()),
        fetch('/api/positions').then(r => r.json()),
        fetch('/api/accuracy').then(r => r.json()),
        fetch('/api/equity').then(r => r.json()),
        fetch('/api/watchlist').then(r => r.json()),
      ]);
      renderState(state);
      renderPositions(positions);
      renderAccuracy(accuracy);
      renderEquity(equity);
      renderWatchlist(watchlist);
    } catch (e) {
      console.error('Dashboard load error:', e);
    }
    document.getElementById('spinner').style.display = 'none';
  }

  function renderState(s) {
    const pv = s.portfolio_value || 0;
    const sc = s.starting_capital || 1000000;
    const ret = ((pv / sc) - 1) * 100;
    const cash = s.cash || 0;
    const cashPct = pv > 0 ? (cash / pv * 100) : 0;
    const dpnl = s.daily_pnl;
    const pos = s.open_positions || 0;
    const days = s.trading_days_elapsed || 0;
    const regime = s.regime || 'UNKNOWN';
    const mfScore = s.mirofish_score;
    const sessionId = s.session_id || '—';

    document.getElementById('session-label').textContent =
      'Session: ' + sessionId + '  |  ' + (s.last_updated || '');

    const pvEl = document.getElementById('kpi-pv');
    pvEl.textContent = fmt(pv);
    pvEl.className = 'value ' + colorClass(ret);

    document.getElementById('kpi-return').textContent = 'Total return: ' + fmtPct(ret);

    document.getElementById('kpi-cash').textContent = fmt(cash);
    document.getElementById('kpi-cash-pct').textContent =
      cashPct.toFixed(1) + '% of portfolio';

    const dpEl = document.getElementById('kpi-dpnl');
    if (dpnl != null) {
      dpEl.textContent = fmtPct(dpnl);
      dpEl.className = 'value ' + colorClass(dpnl);
      document.getElementById('kpi-dpnl-sub').textContent = 'vs previous close';
    } else {
      dpEl.textContent = '—';
      document.getElementById('kpi-dpnl-sub').textContent = 'No prior snapshot';
    }

    document.getElementById('kpi-pos').textContent = pos;
    document.getElementById('kpi-days').textContent =
      'Day ' + days + '/20' + (days >= 20 ? ' ✓ Ready' : '');

    // Regime badge
    const regimeEl = document.getElementById('regime-badge');
    regimeEl.className = 'regime-badge regime-' + regime;
    const dot = regime === 'BULL' ? '●' : (regime === 'BEAR' ? '●' : '●');
    const mfStr = mfScore != null ? ' (' + (mfScore > 0 ? '+' : '') + Number(mfScore).toFixed(2) + ')' : '';
    regimeEl.textContent = dot + ' ' + regime + mfStr;
  }

  function renderPositions(positions) {
    const tbody = document.getElementById('positions-body');
    if (!positions || positions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="no-data">No open positions</td></tr>';
      return;
    }
    tbody.innerHTML = positions.map(p => {
      const pnl = p.pnl_pct;
      const cur = p.current_price;
      return `<tr>
        <td><strong>${p.symbol}</strong></td>
        <td>${p.qty}</td>
        <td>${fmt(p.entry_price)}</td>
        <td>${cur != null ? fmt(cur) : '<span style="color:var(--subtext)">N/A</span>'}</td>
        <td class="${colorClass(pnl)}">${pnl != null ? fmtPct(pnl) : '<span style="color:var(--subtext)">N/A</span>'}</td>
        <td style="color:var(--red)">${fmt(p.stop_loss)}</td>
        <td style="color:var(--green)">${fmt(p.target_price)}</td>
        <td style="color:var(--subtext)">${p.strategy || '—'}</td>
        <td><span class="regime-badge regime-${p.regime || 'UNKNOWN'}" style="font-size:11px;padding:2px 8px;">${p.regime || '?'}</span></td>
      </tr>`;
    }).join('');
  }

  function renderAccuracy(acc) {
    function pct(v) {
      if (v == null) return '<span style="color:var(--subtext);font-size:14px;">N/A</span>';
      return `<span style="color:${accColor(v)}">${Number(v).toFixed(1)}%</span>`;
    }
    document.getElementById('acc-1d').innerHTML  = pct(acc.bull_accuracy_1d);
    document.getElementById('acc-3d').innerHTML  = pct(acc.overall_accuracy_3d);
    document.getElementById('acc-5d').innerHTML  = pct(acc.overall_accuracy_5d);
    document.getElementById('acc-10d').innerHTML = pct(acc.last_10_signals_accuracy);

    const trend = acc.accuracy_trend || 'unknown';
    const trendIcon = trend === 'improving' ? '↑' : (trend === 'declining' ? '↓' : '→');
    const total = acc.total_signals_evaluated || 0;
    const wl5 = acc.watchlist_accuracy_5d;
    document.getElementById('acc-meta').innerHTML =
      `Evaluated: <strong>${total}</strong> signals &nbsp;|&nbsp; ` +
      `Trend: <strong>${trendIcon} ${trend}</strong>` +
      (wl5 != null ? ` &nbsp;|&nbsp; Watchlist 5d: <strong style="color:${accColor(wl5)}">${wl5.toFixed(1)}%</strong>` : '');
  }

  function renderEquity(points) {
    if (!points || points.length === 0) return;
    equityChart.data.labels = points.map(p => p.date);
    equityChart.data.datasets[0].data = points.map(p => p.value);
    equityChart.update();
  }

  function renderWatchlist(items) {
    const tbody = document.getElementById('watchlist-body');
    if (!items || items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="no-data">No watchlist data</td></tr>';
      return;
    }
    const tierColors = { A: 'var(--green)', B: 'var(--yellow)', C: 'var(--blue)', AVOID: 'var(--red)' };
    tbody.innerHTML = items.slice(0, 5).map((w, i) => {
      const tc = tierColors[w.tier] || 'var(--subtext)';
      return `<tr>
        <td style="color:var(--subtext)">${i + 1}</td>
        <td><strong>${w.symbol || '?'}</strong></td>
        <td>${w.score != null ? Number(w.score).toFixed(0) : '—'}</td>
        <td><span style="color:${tc};font-weight:700;">${w.tier || '?'}</span></td>
        <td style="color:var(--green)">${w.action || 'WATCH'}</td>
      </tr>`;
    }).join('');
  }

  // ── Countdown + auto-refresh ──────────────────────────────────────────────
  let countdown = 30;
  const countdownEl = document.getElementById('countdown');
  setInterval(() => {
    countdown--;
    if (countdown <= 0) {
      countdown = 30;
      loadAll();
    }
    countdownEl.textContent = countdown;
  }, 1000);

  // Initial load
  loadAll();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

@dataclass
class SessionInfo:
    session_id: str
    data_dir: Path


class SessionManager:
    """Resolves the active paper trading session from disk."""

    def __init__(self, data_dir: str = "data/paper_trading", session_id: Optional[str] = None):
        self._base = Path(data_dir)
        self._override = session_id

    def get_active(self) -> SessionInfo:
        """Return the active session, preferring CLI override then active_session.txt then latest dir."""
        if self._override:
            return SessionInfo(self._override, self._base / self._override)

        # Try active_session.txt
        active_file = self._base / "active_session.txt"
        if active_file.exists():
            try:
                sid = active_file.read_text(encoding="utf-8").strip()
                if sid:
                    return SessionInfo(sid, self._base / sid)
            except Exception:
                pass

        # Fall back to latest directory
        dirs = sorted(
            [d for d in self._base.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if dirs:
            sid = dirs[0].name
            return SessionInfo(sid, dirs[0])

        # Default placeholder
        sid = f"paper_{date.today()}"
        return SessionInfo(sid, self._base / sid)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.debug("JSON load failed for %s: %s", path, exc)
        return {}


def load_state(session: SessionInfo) -> dict:
    """Load state.json for the session."""
    return _load_json(session.data_dir / "state.json")  # type: ignore[return-value]


def load_accuracy(session: SessionInfo) -> dict:
    """Load accuracy_report.json for the session."""
    return _load_json(session.data_dir / "accuracy_report.json")  # type: ignore[return-value]


def load_equity_history(session: SessionInfo) -> list[dict]:
    """
    Build equity curve from snapshot_*.json files in the session directory.
    Each point: {date: str, value: float}.
    """
    pattern = str(session.data_dir / "snapshot_*.json")
    files = sorted(glob.glob(pattern))
    points: list[dict] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                snap = json.load(fh)
            d = snap.get("date") or snap.get("snapshot_date")
            v = snap.get("portfolio_value")
            if d and v is not None:
                points.append({"date": str(d), "value": float(v)})
        except Exception as exc:
            logger.debug("Snapshot load failed for %s: %s", path, exc)
    return points


def load_latest_cycle(session: SessionInfo) -> dict:
    """Load the most recent cycle_*.json from the session directory."""
    pattern = str(session.data_dir / "cycle_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if files:
        result = _load_json(Path(files[0]))
        return result if isinstance(result, dict) else {}
    return {}


def load_watchlist() -> list[dict]:
    """Load data/watchlist_latest.json or fall back to strategy watchlist helper."""
    wl_path = Path("data/watchlist_latest.json")
    if wl_path.exists():
        raw = _load_json(wl_path)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return raw.get("items", raw.get("watchlist", []))

    # Fallback: try strategy module
    try:
        from strategy.watchlist import load_latest_watchlist  # type: ignore[import]
        result = load_latest_watchlist()
        if result:
            return [
                item if isinstance(item, dict) else item.__dict__
                for item in result
            ]
    except Exception as exc:
        logger.debug("Strategy watchlist load failed: %s", exc)

    return []


def _compute_daily_pnl(session: SessionInfo) -> Optional[float]:
    """Compute daily P&L % from the last two equity snapshots."""
    pts = load_equity_history(session)
    if len(pts) < 2:
        return None
    prev = pts[-2]["value"]
    cur  = pts[-1]["value"]
    if prev <= 0:
        return None
    return round((cur / prev - 1) * 100, 3)


def _build_state_response(session: SessionInfo) -> dict:
    """Assemble the /api/state payload."""
    state = load_state(session)
    cycle = load_latest_cycle(session)

    capital      = state.get("capital", 0.0)
    start_cap    = state.get("starting_capital", 1_000_000.0)
    positions    = state.get("positions", {})
    session_start = state.get("session_start", str(date.today()))

    # Estimate portfolio value (entry price × qty, same as engine)
    pos_value = sum(
        p.get("entry_price", 0) * p.get("qty", 0)
        for p in positions.values()
        if isinstance(p, dict)
    )
    portfolio_value = capital + pos_value

    # Trading days elapsed
    try:
        from backtest.calendar import get_trading_days  # type: ignore[import]
        days = len(get_trading_days(session_start, str(date.today())))
    except Exception:
        try:
            days = max(0, (date.today() - date.fromisoformat(session_start)).days)
        except Exception:
            days = 0

    # Regime / mirofish from latest snapshot
    snap_pattern = str(session.data_dir / "snapshot_*.json")
    snaps = sorted(glob.glob(snap_pattern), reverse=True)
    regime = "UNKNOWN"
    mirofish_score: Optional[float] = None
    if snaps:
        try:
            with open(snaps[0], encoding="utf-8") as fh:
                latest_snap = json.load(fh)
            regime        = latest_snap.get("regime", "UNKNOWN")
            mirofish_score = latest_snap.get("mirofish_score")
        except Exception:
            pass

    # Fallback from cycle log
    if regime == "UNKNOWN":
        for step in cycle.get("steps", []):
            if step.get("name") == "detect_regime":
                regime = step.get("regime", "UNKNOWN")
                break

    return {
        "session_id":            session.session_id,
        "portfolio_value":       round(portfolio_value, 2),
        "cash":                  round(capital, 2),
        "starting_capital":      round(start_cap, 2),
        "open_positions":        len(positions),
        "trading_days_elapsed":  days,
        "regime":                regime,
        "mirofish_score":        mirofish_score,
        "daily_pnl":             _compute_daily_pnl(session),
        "last_updated":          datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _build_positions_response(session: SessionInfo) -> list[dict]:
    """Assemble the /api/positions payload with live-price enrichment where possible."""
    state = load_state(session)
    positions = state.get("positions", {})

    result: list[dict] = []
    symbols = list(positions.keys())

    # Try fetching current prices
    current_prices: dict[str, float] = {}
    if symbols:
        try:
            from db.loader import get_latest_prices  # type: ignore[import]
            data = get_latest_prices(symbols)
            current_prices = {s: d.get("close", 0.0) for s, d in data.items() if d}
        except Exception:
            pass

    for sym, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        ep = pos.get("entry_price", 0.0)
        qty = pos.get("qty", 0)
        sl  = pos.get("stop_loss", round(ep * 0.92, 2))
        tp  = pos.get("target_price", round(ep * 1.18, 2))
        cp  = current_prices.get(sym)
        pnl_pct: Optional[float] = None
        if cp and ep > 0:
            pnl_pct = round((cp / ep - 1) * 100, 2)

        result.append({
            "symbol":        sym,
            "qty":           qty,
            "entry_price":   ep,
            "current_price": cp,
            "pnl_pct":       pnl_pct,
            "stop_loss":     sl,
            "target_price":  tp,
            "strategy":      pos.get("strategy", ""),
            "regime":        pos.get("regime", "UNKNOWN"),
            "entry_date":    pos.get("entry_date", ""),
            "signal_score":  pos.get("signal_score", 0.0),
        })
    return result


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(session_manager: SessionManager):
    """Create and return the FastAPI application."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:
        raise ImportError("fastapi not installed — run: pip install fastapi uvicorn") from exc

    app = FastAPI(title="MiroFish Paper Trading Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get("/api/state")
    async def api_state():
        session = session_manager.get_active()
        try:
            data = _build_state_response(session)
        except Exception as exc:
            logger.warning("State build error: %s", exc)
            data = {"session_id": session.session_id, "error": str(exc)}
        return JSONResponse(content=data)

    @app.get("/api/positions")
    async def api_positions():
        session = session_manager.get_active()
        try:
            data = _build_positions_response(session)
        except Exception as exc:
            logger.warning("Positions build error: %s", exc)
            data = []
        return JSONResponse(content=data)

    @app.get("/api/accuracy")
    async def api_accuracy():
        session = session_manager.get_active()
        data = load_accuracy(session)
        if not isinstance(data, dict):
            data = {}
        return JSONResponse(content=data)

    @app.get("/api/equity")
    async def api_equity():
        session = session_manager.get_active()
        try:
            points = load_equity_history(session)
        except Exception as exc:
            logger.warning("Equity history error: %s", exc)
            points = []
        return JSONResponse(content=points)

    @app.get("/api/watchlist")
    async def api_watchlist():
        try:
            items = load_watchlist()
            # Normalise to plain dicts (handles dataclass/object entries)
            result: list[dict] = []
            for item in items[:10]:
                if isinstance(item, dict):
                    result.append(item)
                elif hasattr(item, "__dict__"):
                    result.append(item.__dict__)
            return JSONResponse(content=result)
        except Exception as exc:
            logger.warning("Watchlist load error: %s", exc)
            return JSONResponse(content=[])

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="MiroFish paper trading live dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP to bind")
    parser.add_argument("--session-id", help="Paper trading session ID (overrides active_session.txt)")
    parser.add_argument("--data-dir", default="data/paper_trading", help="Path to paper trading data root")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed — run: pip install uvicorn")
        raise SystemExit(1)

    session_manager = SessionManager(data_dir=args.data_dir, session_id=args.session_id)
    active = session_manager.get_active()
    logger.info("Dashboard starting — session: %s  data: %s", active.session_id, active.data_dir)
    logger.info("URL: http://%s:%d", args.host, args.port)

    app = create_app(session_manager)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
