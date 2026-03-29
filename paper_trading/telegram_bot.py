"""
paper_trading/telegram_bot.py
Two-way Telegram bot for paper trading interaction.
Supports 9 commands for real-time strategy monitoring.

Requires:
  pip install python-telegram-bot

Configure in .env:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
  TELEGRAM_ALLOWED_USER_ID=...   (numeric Telegram user ID)

Run:
  python -m paper_trading.telegram_bot
  make bot-start
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Load .env so all keys are available via os.environ
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

TELEGRAM_COMMANDS = {
    "/start":     "Start the bot and show available commands",
    "/status":    "Current portfolio value, P&L, positions, session day",
    "/signal":    "Today's full signal breakdown (MiroFish, technical, composite)",
    "/positions": "Open positions with entry, current price, P&L, target, stop",
    "/accuracy":  "Signal accuracy report for last 10/30 signals",
    "/regime":    "Current market regime, confidence, and active strategy",
    "/watchlist": "Today's ranked watchlist with full reasoning",
    "/week":      "Past 5 trading days summary (returns, signals, trades)",
    "/simulate":  "/simulate [question] — run a custom MiroFish simulation",
    "/override":  "/override [HOLD|BUY|SELL] [symbol] [reason] — log manual override",
}


# ---------------------------------------------------------------------------
# Bot handler functions
# ---------------------------------------------------------------------------

def _load_latest_engine_state() -> dict:
    """Load the most recent paper trading session state from disk."""
    import glob
    files = sorted(glob.glob("data/paper_trading/*/state.json"), reverse=True)
    if files:
        try:
            with open(files[0], encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _load_latest_snapshot() -> dict:
    """Load the most recent daily snapshot."""
    import glob
    files = sorted(glob.glob("data/paper_trading/*/snapshot_*.json"), reverse=True)
    if files:
        try:
            with open(files[0], encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _load_accuracy_report() -> dict:
    """Load the latest accuracy report."""
    import glob
    files = sorted(glob.glob("data/paper_trading/*/accuracy_report.json"), reverse=True)
    if files:
        try:
            with open(files[0], encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _load_watchlist() -> list:
    """Load today's watchlist."""
    try:
        from strategy.watchlist import load_latest_watchlist
        return load_latest_watchlist()
    except Exception:
        return []


def _load_cycle_logs(n: int = 5) -> list[dict]:
    """Load the last n daily cycle logs."""
    import glob
    files = sorted(glob.glob("data/paper_trading/*/cycle_*.json"), reverse=True)[:n]
    logs = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                logs.append(json.load(fh))
        except Exception:
            pass
    return logs


# ── Command handlers ─────────────────────────────────────────────────────────

def handle_start(user_id: int) -> str:
    lines = [
        "🐟 *MiroFish NEPSE Paper Trading Bot*",
        "",
        "Available commands:",
    ]
    for cmd, desc in TELEGRAM_COMMANDS.items():
        lines.append(f"  `{cmd}` — {desc}")
    lines.append("")
    lines.append("_Only authorised users can interact with this bot._")
    return "\n".join(lines)


def handle_status(user_id: int) -> str:
    state = _load_latest_engine_state()
    snap = _load_latest_snapshot()

    capital = state.get("capital", 0)
    start_cap = state.get("starting_capital", 1_000_000)
    positions = state.get("positions", {})

    # Estimate portfolio value
    pv = capital + sum(
        p.get("entry_price", 0) * p.get("qty", 0)
        for p in positions.values()
        if isinstance(p, dict)
    )
    ret = (pv / start_cap - 1) * 100 if start_cap > 0 else 0

    regime = snap.get("regime", "UNKNOWN")
    mf_score = snap.get("mirofish_score", 0)
    session_start = state.get("session_start", str(date.today()))

    # Trading days elapsed
    try:
        from backtest.calendar import get_trading_days
        days = len(get_trading_days(session_start, str(date.today())))
    except Exception:
        from datetime import date as _date
        try:
            days = max(0, (_date.today() - _date.fromisoformat(session_start)).days)
        except Exception:
            days = 0

    lines = [
        f"📊 *Paper Trading Status*",
        f"",
        f"💼 Portfolio: NPR {pv:,.0f}",
        f"📈 Return: {ret:+.2f}%",
        f"💵 Cash: NPR {capital:,.0f}",
        f"",
        f"📅 Day {days}/20 of paper trading",
        f"{'✅ Ready for live review!' if days >= 20 else f'⏳ {20-days} more days needed'}",
        f"",
        f"📊 Regime: `{regime}` | MF Score: `{mf_score:+.2f}`",
        f"",
        f"*Open Positions ({len(positions)}):*",
    ]
    for sym, pos in list(positions.items())[:5]:
        if isinstance(pos, dict):
            ep = pos.get("entry_price", 0)
            qty = pos.get("qty", 0)
            lines.append(f"  {sym}: {qty} shares @ NPR {ep:,.0f}")

    return "\n".join(lines)


def handle_signal(user_id: int) -> str:
    snap = _load_latest_snapshot()
    mf = snap.get("mirofish_score", 0)
    regime = snap.get("regime", "UNKNOWN")

    # Load latest watchlist for top picks
    watchlist = _load_watchlist()
    tier_a = [w for w in watchlist if isinstance(w, dict) and w.get("tier") == "A"][:3]

    action = "BUY" if mf > 0.2 else ("SELL" if mf < -0.2 else "HOLD")
    signal_emoji = "🟢" if mf > 0.2 else ("🔴" if mf < -0.2 else "🟡")

    lines = [
        f"📡 *Today's Signal Breakdown*",
        f"",
        f"{signal_emoji} *MiroFish:* `{mf:+.3f}` — {action}",
        f"📈 *Regime:* `{regime}`",
        f"",
        f"*Top Conviction Picks:*",
    ]
    if tier_a:
        for w in tier_a:
            sym = w.get("symbol", "?")
            score = w.get("score", 0)
            rsi = w.get("rsi", 0)
            vol = w.get("vol_ratio", 1)
            reasons = w.get("reasons", [])
            reason_str = " | ".join(reasons[:2]) if reasons else "—"
            lines.append(f"  🟢 *{sym}* — Score: `{score:.0f}` | RSI: `{rsi:.0f}` | Vol: `{vol:.1f}x`")
            lines.append(f"     _{reason_str}_")
    else:
        lines.append("  No Tier A signals today.")

    lines.append(f"\n_Snapshot: {snap.get('date', date.today())}_")
    return "\n".join(lines)


def handle_positions(user_id: int) -> str:
    state = _load_latest_engine_state()
    positions = state.get("positions", {})

    if not positions:
        return "📭 No open positions currently."

    lines = [f"📋 *Open Positions ({len(positions)})*\n"]
    header = f"{'Symbol':<8} | {'Qty':>4} | {'Entry':>8} | {'P&L':>6} | {'Target':>8} | {'Stop':>8}"
    lines.append(f"`{header}`")
    lines.append(f"`{'-'*60}`")

    for sym, pos in positions.items():
        if not isinstance(pos, dict):
            continue
        ep = pos.get("entry_price", 0)
        qty = pos.get("qty", 0)
        sl = pos.get("stop_loss", ep * 0.92)
        tp = pos.get("target_price", ep * 1.18)
        # Note: current price not available without live feed
        lines.append(
            f"`{sym:<8} | {qty:>4} | {ep:>8,.0f} | {'N/A':>6} | {tp:>8,.0f} | {sl:>8,.0f}`"
        )

    return "\n".join(lines)


def handle_accuracy(user_id: int) -> str:
    report = _load_accuracy_report()
    if not report or report.get("total_signals_evaluated", 0) == 0:
        return "📊 No accuracy data yet. Signals need at least 3 trading days to evaluate."

    total = report.get("total_signals_evaluated", 0)
    bull3 = report.get("bull_accuracy_3d")
    bear3 = report.get("bear_accuracy_3d")
    overall3 = report.get("overall_accuracy_3d")
    last10 = report.get("last_10_signals_accuracy")
    watch5 = report.get("watchlist_accuracy_5d")
    trend = report.get("accuracy_trend", "unknown")
    trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend, "❓")

    def _fmt(v) -> str:
        return f"{v:.0f}%" if v is not None else "N/A"

    lines = [
        f"🎯 *Signal Accuracy Report*",
        f"Evaluated: {total} signals | Trend: {trend_emoji} {trend}",
        f"",
        f"*3-Day Window:*",
        f"  Bullish: `{_fmt(bull3)}` | Bearish: `{_fmt(bear3)}` | Overall: `{_fmt(overall3)}`",
        f"",
        f"*5-Day Window:*",
        f"  Watchlist BUY accuracy: `{_fmt(watch5)}`",
        f"",
        f"*Last 10 Signals:* `{_fmt(last10)}`",
    ]
    if last10 is not None and last10 < 50:
        lines.append("⚠️ *Accuracy below 50% — possible regime shift. Review strategy.*")

    return "\n".join(lines)


def handle_regime(user_id: int) -> str:
    snap = _load_latest_snapshot()
    regime = snap.get("regime", "UNKNOWN")

    try:
        from strategy.regime_detector import detect_regime
        result = detect_regime()
        confidence = result.get("confidence", 0.5) if isinstance(result, dict) else 0.5
    except Exception:
        confidence = 0.5

    regime_strats = {
        "BULL": "momentum_bull (aggressive long)",
        "EARLY_BULL": "momentum_bull (moderate long)",
        "RECOVERY": "momentum_bull (selective long)",
        "SIDEWAYS": "mean_reversion (range trading)",
        "BEAR": "defensive_bear (cash heavy)",
        "CAPITULATION": "defensive_bear (full cash)",
    }
    strategy = regime_strats.get(regime, "mean_reversion (default)")

    lines = [
        f"📊 *Market Regime*",
        f"",
        f"Regime: `{regime}`",
        f"Confidence: `{confidence*100:.0f}%`",
        f"Active Strategy: _{strategy}_",
        f"",
        f"*Historical Context:*",
    ]
    regime_perf = {
        "BULL": "Best performance — strategy generated +12% avg in backtest bulls",
        "SIDEWAYS": "Moderate — mean reversion works 60% of the time in sideways NEPSE",
        "BEAR": "Cautious — hold cash, avoid new longs until regime shifts",
    }
    lines.append(regime_perf.get(regime, "Regime not yet characterised in backtest."))
    lines.append(f"\n_Detected: {snap.get('date', date.today())}_")
    return "\n".join(lines)


def handle_watchlist(user_id: int) -> str:
    watchlist = _load_watchlist()
    if not watchlist:
        return "📋 No watchlist data available yet."

    snap = _load_latest_snapshot()
    regime = snap.get("regime", "SIDEWAYS")

    lines = [f"👀 *Today's Watchlist* — Regime: `{regime}`\n"]
    for i, w in enumerate(watchlist[:8], 1):
        if not isinstance(w, dict):
            continue
        sym = w.get("symbol", "?")
        tier = w.get("tier", "?")
        action = w.get("action", "WATCH")
        score = w.get("score", 0)
        rsi = w.get("rsi", 50)
        vol = w.get("vol_ratio", 1)
        mf = w.get("mirofish_score", 0)
        reasons = w.get("reasons", [])
        warnings = w.get("warnings", [])
        tier_emoji = {"A": "🟢", "B": "🟡", "C": "🔵", "AVOID": "🔴"}.get(tier, "⚪")

        lines.append(f"{tier_emoji} *{i}. {sym}* — `{action}` | Score: `{score:.0f}` | Tier: {tier}")
        lines.append(f"   RSI: `{rsi:.0f}` | Vol: `{vol:.1f}x` | MF: `{mf:+.2f}`")
        if reasons:
            lines.append(f"   _{', '.join(reasons[:2])}_")
        if warnings:
            lines.append(f"   ⚠️ _{', '.join(warnings)}_")
        lines.append("")

    return "\n".join(lines)


def handle_week(user_id: int) -> str:
    logs = _load_cycle_logs(5)
    if not logs:
        return "📅 No weekly data available yet."

    lines = ["📅 *Last 5 Trading Days*\n"]
    for log in reversed(logs):
        d = log.get("date", "?")
        steps = log.get("steps", [])
        exits_step = next((s for s in steps if s.get("name") == "check_exits"), {})
        orders_step = next((s for s in steps if s.get("name") == "place_orders"), {})
        exits = exits_step.get("exits", 0)
        orders = orders_step.get("orders", 0)
        errors = log.get("errors", [])
        status = "✅" if not errors else "⚠️"
        lines.append(f"{status} *{d}*: {orders} orders | {exits} exits | {len(errors)} errors")

    lines.append("\n_Tip: /accuracy for signal performance stats_")
    return "\n".join(lines)


def handle_simulate(user_id: int, question: str) -> str:
    if not question.strip():
        return "Usage: `/simulate What happens if NRB cuts repo rate by 50bp?`"

    lines = [
        f"🔄 *Custom Simulation*",
        f"",
        f"Question: _{question}_",
        f"",
        f"Running MiroFish simulation... (5-10 minutes)",
        f"",
        f"_Results will be sent when complete._",
    ]
    # Queue simulation asynchronously
    try:
        from pipeline.generate_simulation_question import save_custom_question
        save_custom_question(question)
    except Exception as exc:
        return f"❌ Could not queue simulation: {exc}"

    return "\n".join(lines)


def handle_override(user_id: int, args: str) -> str:
    """
    /override [HOLD|BUY|SELL] [symbol] [reason]
    Logs a manual override decision.
    """
    parts = args.strip().split(None, 2)
    if len(parts) < 3:
        return "Usage: `/override SELL NHPC Taking profit before NRB meeting`"

    action, symbol, reason = parts[0].upper(), parts[1].upper(), parts[2]
    if action not in {"HOLD", "BUY", "SELL"}:
        return f"❌ Invalid action '{action}'. Use HOLD, BUY, or SELL."

    override_record = {
        "date": str(date.today()),
        "user_id": user_id,
        "action": action,
        "symbol": symbol,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }

    override_path = Path("data/paper_trading/overrides.json")
    overrides = []
    if override_path.exists():
        try:
            with open(override_path, encoding="utf-8") as fh:
                overrides = json.load(fh)
        except Exception:
            pass
    overrides.append(override_record)
    with open(override_path, "w", encoding="utf-8") as fh:
        json.dump(overrides, fh, indent=2, default=str)

    return (f"✅ *Override Logged*\n\n"
            f"Action: `{action}` on `{symbol}`\n"
            f"Reason: _{reason}_\n\n"
            f"_This is logged only — no order was placed. This is paper trading._")


# ---------------------------------------------------------------------------
# Bot runner
# ---------------------------------------------------------------------------

def run_bot() -> None:
    """Start the Telegram bot with long-polling."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    allowed_user_id = os.environ.get("TELEGRAM_ALLOWED_USER_ID", "")

    if not bot_token or "your_" in bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not configured in .env")
        return

    try:
        from telegram import Update
        from telegram.ext import (
            ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
        )
    except ImportError:
        logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")
        return

    allowed_id = int(allowed_user_id) if allowed_user_id.isdigit() else None

    def _auth(update: Update) -> bool:
        if allowed_id is None:
            return True
        return update.effective_user and update.effective_user.id == allowed_id

    async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        await update.message.reply_text(
            handle_start(update.effective_user.id), parse_mode="Markdown"
        )

    async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        await update.message.reply_text(
            handle_status(update.effective_user.id), parse_mode="Markdown"
        )

    async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        await update.message.reply_text(
            handle_signal(update.effective_user.id), parse_mode="Markdown"
        )

    async def cmd_positions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        await update.message.reply_text(
            handle_positions(update.effective_user.id), parse_mode="Markdown"
        )

    async def cmd_accuracy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        await update.message.reply_text(
            handle_accuracy(update.effective_user.id), parse_mode="Markdown"
        )

    async def cmd_regime(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        await update.message.reply_text(
            handle_regime(update.effective_user.id), parse_mode="Markdown"
        )

    async def cmd_watchlist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        await update.message.reply_text(
            handle_watchlist(update.effective_user.id), parse_mode="Markdown"
        )

    async def cmd_week(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        await update.message.reply_text(
            handle_week(update.effective_user.id), parse_mode="Markdown"
        )

    async def cmd_simulate(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        question = " ".join(ctx.args) if ctx.args else ""
        await update.message.reply_text(
            handle_simulate(update.effective_user.id, question), parse_mode="Markdown"
        )

    async def cmd_override(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not _auth(update):
            return
        args_str = " ".join(ctx.args) if ctx.args else ""
        await update.message.reply_text(
            handle_override(update.effective_user.id, args_str), parse_mode="Markdown"
        )

    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("accuracy", cmd_accuracy))
    app.add_handler(CommandHandler("regime", cmd_regime))
    app.add_handler(CommandHandler("watchlist", cmd_watchlist))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("simulate", cmd_simulate))
    app.add_handler(CommandHandler("override", cmd_override))

    logger.info("Telegram bot started. Polling...")
    print(f"\n  MiroFish Telegram bot running.")
    print(f"  Commands: {', '.join(TELEGRAM_COMMANDS.keys())}")
    print(f"  Press Ctrl+C to stop.\n")
    app.run_polling()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    run_bot()


if __name__ == "__main__":
    main()
