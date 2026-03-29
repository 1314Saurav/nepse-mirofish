"""
paper_trading/price_monitor.py
Intraday price monitor for open paper positions during NEPSE market hours.
Polls ShareSansar/Merolagani every 10 minutes (11:00-15:00 NST, Sun-Thu).

Run:
  python -m paper_trading.price_monitor
  make monitor-start
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Nepal Standard Time = UTC+5:45
NST = timezone(timedelta(hours=5, minutes=45))

MARKET_OPEN_HOUR = 11
MARKET_OPEN_MIN = 0
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 0

NEPSE_TRADING_WEEKDAYS = {0, 1, 2, 3, 6}   # Mon-Thu + Sun


def _nst_now() -> datetime:
    return datetime.now(tz=NST)


def _write_notification(msg: str) -> None:
    """Write alert to notifications file for dashboard to read."""
    import json, datetime
    path = Path(__file__).resolve().parents[1] / "data" / "notifications" / "alerts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.datetime.now().isoformat(), "msg": msg}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _is_market_hours() -> bool:
    """Return True if current NST time is within NEPSE market hours (11:00-15:00, Sun-Thu)."""
    now = _nst_now()
    if now.weekday() not in NEPSE_TRADING_WEEKDAYS:
        return False
    t = (now.hour, now.minute)
    open_t = (MARKET_OPEN_HOUR, MARKET_OPEN_MIN)
    close_t = (MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN)
    return open_t <= t < close_t


class LivePriceMonitor:
    """
    Polls live prices every 10 minutes during NEPSE market hours.
    Sends real-time alerts if positions approach stop loss or target,
    NEPSE index moves sharply, or news breaks on position sectors.
    """

    def __init__(self, paper_positions: dict, alert_thresholds: Optional[dict] = None):
        self.positions = paper_positions       # {symbol: PaperPosition}
        self.thresholds = alert_thresholds or {
            "stop_loss_warning_pct": -5.0,     # warn at -5% (1% before stop at -7%)
            "stop_loss_emergency_pct": -7.5,   # emergency if stop missed
            "target_approach_pct": 15.0,       # warn at 15% gain (near 18% target)
            "nepse_move_alert_pct": 2.0,       # alert if NEPSE moves > 2%
            "circuit_breaker_pct": 9.5,        # circuit breaker threshold
        }
        self.poll_interval = 600               # 10 minutes in seconds
        self._running = False
        self._last_prices: dict[str, float] = {}
        self._alert_cooldown: dict[str, int] = {}  # epoch of last alert per symbol

    # ── Main loop ───────────────────────────────────────────────────────────

    def start_monitoring(self) -> None:
        """
        Run during market hours only (11:00-15:00 NST, Sun-Thu).
        After 15:00, hands off to daily cycle for EOD processing.
        """
        logger.info("LivePriceMonitor started. Waiting for market hours...")
        self._running = True

        try:
            while self._running:
                if _is_market_hours():
                    self._run_poll_cycle()
                    time.sleep(self.poll_interval)
                else:
                    now = _nst_now()
                    logger.debug("Outside market hours (%s NST). Sleeping 5 min.", now.strftime("%H:%M"))
                    time.sleep(300)
        except KeyboardInterrupt:
            logger.info("LivePriceMonitor stopped by user.")
        finally:
            self._running = False

    def stop_monitoring(self) -> None:
        self._running = False

    def _run_poll_cycle(self) -> None:
        """Single polling cycle: fetch prices and check all alerts."""
        now = _nst_now()
        logger.info("Price poll cycle at %s NST", now.strftime("%H:%M"))

        current_prices = self._fetch_live_prices(list(self.positions.keys()))

        # Check each open position
        for symbol, pos in self.positions.items():
            current_price = current_prices.get(symbol)
            if current_price is None or current_price <= 0:
                continue

            self._last_prices[symbol] = current_price
            entry = getattr(pos, "entry_price", current_price)
            pnl_pct = (current_price / entry - 1) * 100 if entry > 0 else 0

            # Stop loss proximity check
            sl_alert = self.check_stop_loss_proximity(symbol, current_price)
            if sl_alert.get("alert"):
                self._send_position_alert(symbol, sl_alert, current_price, pnl_pct)

            # Circuit breaker check
            prev_close = self._get_prev_close(symbol)
            if prev_close and self.detect_circuit_breaker(symbol, current_price, prev_close):
                msg = (f"🚨 CIRCUIT BREAKER ALERT: {symbol}\n"
                       f"Price: NPR {current_price:,.2f} (prev close: NPR {prev_close:,.2f})\n"
                       f"Move: {((current_price/prev_close-1)*100):+.1f}%\n"
                       f"Do NOT place new orders in {symbol} today.")
                _write_notification(msg)

        # NEPSE index move check
        nepse_index = current_prices.get("__NEPSE__")
        prev_nepse = self._last_prices.get("__NEPSE__")
        if nepse_index and prev_nepse and prev_nepse > 0:
            move_pct = (nepse_index / prev_nepse - 1) * 100
            if abs(move_pct) > self.thresholds["nepse_move_alert_pct"]:
                direction = "UP" if move_pct > 0 else "DOWN"
                msg = (f"{'📈' if move_pct > 0 else '📉'} *NEPSE Index Alert*\n"
                       f"NEPSE moved {direction} {abs(move_pct):.1f}% intraday\n"
                       f"Current: {nepse_index:,.2f} | Previous: {prev_nepse:,.2f}\n"
                       f"{'Consider regime re-assessment.' if abs(move_pct) > 3 else ''}")
                _write_notification(msg)
        if nepse_index:
            self._last_prices["__NEPSE__"] = nepse_index

    # ── Alert checks ────────────────────────────────────────────────────────

    def check_stop_loss_proximity(self, symbol: str, current_price: float) -> dict:
        """
        Alert levels:
        - Yellow: position at -5% (1% from stop loss)
        - Red:    position at -7.5% (emergency — stop may have been missed)
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return {"alert": False}

        entry = getattr(pos, "entry_price", current_price)
        if entry <= 0:
            return {"alert": False}

        pnl_pct = (current_price / entry - 1) * 100
        stop_loss = getattr(pos, "stop_loss", entry * 0.92)
        target = getattr(pos, "target_price", entry * 1.18)

        # Cooldown: don't spam alerts for same symbol within 30 min
        now_epoch = int(time.time())
        last_alert = self._alert_cooldown.get(symbol, 0)
        if now_epoch - last_alert < 1800:
            return {"alert": False, "cooldown": True}

        if pnl_pct <= self.thresholds["stop_loss_emergency_pct"]:
            self._alert_cooldown[symbol] = now_epoch
            return {
                "alert": True,
                "level": "RED",
                "message": f"EMERGENCY: {symbol} at {pnl_pct:.1f}% — STOP MAY BE MISSED",
                "pnl_pct": pnl_pct,
                "stop_loss": stop_loss,
                "emoji": "🚨",
            }
        elif pnl_pct <= self.thresholds["stop_loss_warning_pct"]:
            self._alert_cooldown[symbol] = now_epoch
            return {
                "alert": True,
                "level": "YELLOW",
                "message": f"WARNING: {symbol} approaching stop loss at {pnl_pct:.1f}%",
                "pnl_pct": pnl_pct,
                "stop_loss": stop_loss,
                "emoji": "⚠️",
            }
        elif pnl_pct >= self.thresholds["target_approach_pct"]:
            self._alert_cooldown[symbol] = now_epoch
            return {
                "alert": True,
                "level": "GREEN",
                "message": f"TARGET NEAR: {symbol} at +{pnl_pct:.1f}% (target {((target/entry-1)*100):.0f}%)",
                "pnl_pct": pnl_pct,
                "target": target,
                "emoji": "🎯",
            }

        return {"alert": False, "pnl_pct": pnl_pct}

    def detect_circuit_breaker(self, symbol: str, current_price: float,
                                prev_close: float) -> bool:
        """
        Return True if price move > 9.5% (near NEPSE circuit breaker limit).
        """
        if prev_close <= 0:
            return False
        move_pct = abs((current_price / prev_close - 1) * 100)
        return move_pct >= self.thresholds["circuit_breaker_pct"]

    def intraday_news_alert(self, position_symbol: str, news_headline: str) -> None:
        """
        If breaking news mentions an open position's company or sector,
        write an immediate notification alert.
        """
        if position_symbol not in self.positions:
            return
        pos = self.positions[position_symbol]
        entry = getattr(pos, "entry_price", 0)
        msg = (f"📰 *BREAKING NEWS — {position_symbol}*\n\n"
               f"{news_headline}\n\n"
               f"You hold {getattr(pos, 'qty', 0)} shares @ NPR {entry:,.2f}\n"
               f"Review position immediately.")
        _write_notification(msg)

    # ── Data fetching ────────────────────────────────────────────────────────

    def _fetch_live_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current prices from ShareSansar scraper or DB."""
        prices = {}
        try:
            from db.loader import get_latest_prices
            data = get_latest_prices(symbols)
            for sym, row in data.items():
                prices[sym] = float(row.get("ltp") or row.get("close") or 0)
        except Exception as exc:
            logger.debug("Live price fetch failed: %s", exc)
        return prices

    def _get_prev_close(self, symbol: str) -> Optional[float]:
        """Get previous day's close price."""
        try:
            from db.loader import get_stock_price
            row = get_stock_price(symbol)
            if row:
                return float(row.get("close") or 0)
        except Exception:
            pass
        return None

    # ── Alert sending ────────────────────────────────────────────────────────

    def _send_position_alert(self, symbol: str, alert: dict,
                              current_price: float, pnl_pct: float) -> None:
        """Format and write a position alert notification."""
        emoji = alert.get("emoji", "⚠️")
        msg = (f"{emoji} *Position Alert: {symbol}*\n\n"
               f"{alert['message']}\n\n"
               f"Current: NPR {current_price:,.2f} | P&L: {pnl_pct:+.1f}%\n"
               f"Stop Loss: NPR {alert.get('stop_loss', 0):,.2f}")
        if "target" in alert:
            msg += f" | Target: NPR {alert['target']:,.2f}"
        _write_notification(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")
    parser = argparse.ArgumentParser(description="NEPSE live price monitor")
    parser.add_argument("--symbols", nargs="*", help="Symbols to monitor")
    args = parser.parse_args()

    # Load positions from latest paper trading session
    positions = {}
    try:
        import glob
        state_files = sorted(glob.glob("data/paper_trading/*/state.json"), reverse=True)
        if state_files:
            with open(state_files[0], encoding="utf-8") as fh:
                state = json.load(fh)
            positions = {s: type("Pos", (), v)() for s, v in state.get("positions", {}).items()}
            print(f"Loaded {len(positions)} positions from {state_files[0]}")
    except Exception as exc:
        logger.warning("Could not load positions: %s. Starting with empty positions.", exc)

    monitor = LivePriceMonitor(positions)
    monitor.start_monitoring()


if __name__ == "__main__":
    main()
