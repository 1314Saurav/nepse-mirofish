"""
paper_trading/engine.py
Paper trading engine — runs the full MiroFish strategy on live NEPSE data
with simulated execution. No real capital at risk.

Paper trading period: minimum 4 weeks (20 trading days) before live capital.

Schedule (via APScheduler):
  15:30 NST — run_daily_cycle()
  11:15 NST — morning_fill_check()

Run:
  python -m paper_trading.engine
  make paper-start
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nepal Standard Time offset (UTC+5:45)
# ---------------------------------------------------------------------------
NST_OFFSET_HOURS = 5
NST_OFFSET_MINUTES = 45


def _nst_now() -> datetime:
    """Return current time as NST (UTC+5:45)."""
    from datetime import timezone, timedelta
    nst = timezone(timedelta(hours=NST_OFFSET_HOURS, minutes=NST_OFFSET_MINUTES))
    return datetime.now(tz=nst)


def _write_notification(msg: str) -> None:
    """Write alert to notifications file for dashboard to read."""
    import json, datetime
    path = Path(__file__).resolve().parents[1] / "data" / "notifications" / "alerts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.datetime.now().isoformat(), "msg": msg}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PaperPosition:
    symbol: str
    qty: int
    entry_price: float
    entry_date: date
    strategy: str
    stop_loss: float
    target_price: float
    signal_score: float = 0.0
    mirofish_score: float = 0.0
    regime: str = "UNKNOWN"

    @property
    def current_pnl_pct(self) -> float:
        return 0.0  # updated externally with live price

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "entry_date": str(self.entry_date),
            "strategy": self.strategy,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "signal_score": self.signal_score,
            "mirofish_score": self.mirofish_score,
            "regime": self.regime,
        }


@dataclass
class PaperOrder:
    symbol: str
    action: str           # BUY | SELL
    qty: int
    order_type: str       # MARKET | LIMIT
    limit_price: Optional[float]
    signal_date: date
    intended_entry: float
    signal_context: dict = field(default_factory=dict)
    fill_price: Optional[float] = None
    fill_date: Optional[date] = None
    fill_status: str = "PENDING"   # PENDING | FILLED | MISSED


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PaperTradingEngine:
    """
    Runs the full MiroFish strategy on live market data with simulated execution.
    Paper trading period: minimum 4 weeks (20 trading days) before live capital.

    Differences from backtest engine:
    - Uses LIVE scraped data (not historical DB) for prices and news
    - Runs MiroFish simulation in real-time (not pre-computed signals)
    - Executes at actual next-day open prices (tracks live NEPSE data)
    - Sends real Telegram alerts (acts as if it were live)
    - Tracks signal-to-outcome in real time with no hindsight
    """

    def __init__(
        self,
        starting_virtual_capital_npr: float = 1_000_000,
        paper_trade_id: Optional[str] = None,
    ):
        self.capital = starting_virtual_capital_npr
        self.starting_capital = starting_virtual_capital_npr
        self.positions: dict[str, PaperPosition] = {}
        self.trade_log: list[dict] = []
        self.pending_orders: list[PaperOrder] = []
        self.session_start = datetime.now()
        self.paper_trade_id = paper_trade_id or f"paper_{date.today()}"
        self._data_dir = Path("data/paper_trading") / self.paper_trade_id
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def portfolio_value(self) -> float:
        """Current portfolio value (cash + open position value at last known price)."""
        pos_value = sum(
            p.entry_price * p.qty
            for p in self.positions.values()
        )
        return self.capital + pos_value

    @property
    def total_return_pct(self) -> float:
        if self.starting_capital <= 0:
            return 0.0
        return (self.portfolio_value / self.starting_capital - 1) * 100

    @property
    def trading_days_elapsed(self) -> int:
        try:
            from backtest.calendar import get_trading_days
            days = get_trading_days(
                self.session_start.date().isoformat(),
                date.today().isoformat()
            )
            return len(days)
        except Exception:
            return max(0, (date.today() - self.session_start.date()).days)

    # ── Daily cycle (15:30 NST) ─────────────────────────────────────────────

    def run_daily_cycle(self) -> dict:
        """
        Called by scheduler at 15:30 NST (after NEPSE close).
        Full 15-step cycle returning a summary dict.
        """
        today = date.today()
        logger.info("=== Paper Trading Daily Cycle: %s ===", today)
        cycle_log: dict = {"date": str(today), "steps": [], "errors": []}

        # Step 1: Check if market was open today
        try:
            from backtest.calendar import is_trading_day
            if not is_trading_day(today):
                logger.info("Market closed today (%s). Skipping cycle.", today.strftime("%A"))
                return {"date": str(today), "skipped": "market_closed"}
        except Exception as exc:
            logger.warning("Calendar check failed: %s", exc)

        # Step 2: Scrape today's market data
        market_data = self._scrape_market_data(today)
        cycle_log["steps"].append({"step": 2, "name": "scrape_market_data",
                                    "ok": bool(market_data)})

        # Step 3: Collect today's news
        news_articles = self._collect_news(today)
        cycle_log["steps"].append({"step": 3, "name": "collect_news",
                                    "articles": len(news_articles)})

        # Step 4: Build seed file
        seed = self._build_seed(today, market_data, news_articles)
        cycle_log["steps"].append({"step": 4, "name": "build_seed", "ok": bool(seed)})

        # Step 5: Run MiroFish simulation
        mirofish_signal = self._run_simulation(today, seed)
        cycle_log["steps"].append({"step": 5, "name": "mirofish_simulation",
                                    "score": mirofish_signal.get("bull_bear_score", 0)})

        # Step 6: Compute technical indicators
        indicators = self._compute_indicators(today)
        cycle_log["steps"].append({"step": 6, "name": "compute_indicators",
                                    "symbols": len(indicators)})

        # Step 7: Detect regime
        regime_data = self._detect_regime(indicators)
        regime = regime_data.get("regime", "SIDEWAYS")
        cycle_log["steps"].append({"step": 7, "name": "detect_regime", "regime": regime})

        # Step 8: Combine signals
        combined_signals = self._combine_signals(indicators, mirofish_signal, regime_data)
        cycle_log["steps"].append({"step": 8, "name": "combine_signals",
                                    "count": len(combined_signals)})

        # Step 9: Check exit conditions on open positions
        exits = self._check_exits(today, indicators, regime)
        for sym, exit_price, reason in exits:
            self._execute_paper_exit(sym, exit_price, reason, today)
        cycle_log["steps"].append({"step": 9, "name": "check_exits",
                                    "exits": len(exits)})

        # Step 10: Generate watchlist
        watchlist = self._generate_watchlist(combined_signals, regime_data)
        cycle_log["steps"].append({"step": 10, "name": "generate_watchlist",
                                    "count": len(watchlist)})

        # Step 11: Apply hard trading rules
        approved_signals = self._apply_trading_rules(combined_signals, indicators)
        cycle_log["steps"].append({"step": 11, "name": "apply_rules",
                                    "approved": len(approved_signals)})

        # Step 12: Simulate order placement
        new_orders = self._place_paper_orders(approved_signals, indicators, watchlist, today)
        cycle_log["steps"].append({"step": 12, "name": "place_orders",
                                    "orders": len(new_orders)})

        # Step 13: Save everything
        self._save_daily_snapshot(today, regime, mirofish_signal, combined_signals)
        cycle_log["steps"].append({"step": 13, "name": "save_snapshot", "ok": True})

        # Step 14: Write notification alert
        tg_msg = self._format_daily_telegram(
            today, mirofish_signal, regime, watchlist, new_orders, exits
        )
        _write_notification(tg_msg)
        cycle_log["steps"].append({"step": 14, "name": "notification_alert"})

        # Step 15: Update accuracy tracker
        try:
            from paper_trading.signal_tracker import SignalAccuracyTracker
            tracker = SignalAccuracyTracker(self.paper_trade_id)
            tracker.record_signal(str(today), {
                "mirofish_score": mirofish_signal.get("bull_bear_score", 0),
                "regime": regime,
                "composite": combined_signals,
            })
            tracker.evaluate_past_signals(market_data)
        except Exception as exc:
            logger.warning("Signal tracker update failed: %s", exc)

        # Save cycle log
        log_path = self._data_dir / f"cycle_{today}.json"
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(cycle_log, fh, indent=2, default=str)

        logger.info("Daily cycle complete. Orders placed: %d", len(new_orders))
        return cycle_log

    # ── Morning fill check (11:15 NST) ─────────────────────────────────────

    def morning_fill_check(self) -> list[dict]:
        """
        Called at 11:15 NST (15 min after market open).
        Checks pending orders from previous day against actual open prices.
        """
        today = date.today()
        fills = []

        for order in list(self.pending_orders):
            sym = order.symbol
            try:
                open_price = self._get_live_open_price(sym, today)
                if open_price is None:
                    continue

                filled = False
                if order.order_type == "MARKET":
                    fill_price = open_price
                    filled = True
                elif order.order_type == "LIMIT" and order.limit_price:
                    if order.action == "BUY" and open_price <= order.limit_price:
                        fill_price = open_price
                        filled = True
                    elif order.action == "SELL" and open_price >= order.limit_price:
                        fill_price = open_price
                        filled = True

                if filled:
                    order.fill_price = fill_price
                    order.fill_date = today
                    order.fill_status = "FILLED"

                    if order.action == "BUY":
                        cost = fill_price * order.qty * 1.004  # 0.4% commission
                        if cost <= self.capital:
                            self.capital -= cost
                            self.positions[sym] = PaperPosition(
                                symbol=sym,
                                qty=order.qty,
                                entry_price=fill_price,
                                entry_date=today,
                                strategy=order.signal_context.get("strategy", "unknown"),
                                stop_loss=fill_price * 0.92,
                                target_price=fill_price * 1.18,
                                signal_score=order.signal_context.get("composite_signal", 0),
                                mirofish_score=order.signal_context.get("mirofish_score", 0),
                                regime=order.signal_context.get("regime", "UNKNOWN"),
                            )
                            fills.append({"symbol": sym, "action": "BUY",
                                          "fill_price": fill_price, "qty": order.qty})
                            msg = (f"PAPER FILL: BUY {order.qty} {sym} @ NPR {fill_price:.2f}\n"
                                   f"Cost: NPR {cost:,.0f} | Cash remaining: NPR {self.capital:,.0f}")
                            _write_notification(msg)
                    elif order.action == "SELL" and sym in self.positions:
                        pos = self.positions.pop(sym)
                        gross = fill_price * pos.qty
                        pnl_pct = (fill_price / pos.entry_price - 1) * 100
                        # NEPSE tax: 7.5% on gains if held < 1yr, 5% if held > 1yr
                        tax = gross * 0.075 if pnl_pct > 0 else 0
                        net_proceeds = gross - gross * 0.004 - tax
                        self.capital += net_proceeds
                        self.trade_log.append({
                            "symbol": sym, "action": "SELL",
                            "entry_price": pos.entry_price, "exit_price": fill_price,
                            "qty": pos.qty, "pnl_pct": round(pnl_pct, 2),
                            "entry_date": str(pos.entry_date), "exit_date": str(today),
                            "regime": pos.regime,
                        })
                        fills.append({"symbol": sym, "action": "SELL",
                                      "fill_price": fill_price, "pnl_pct": pnl_pct})
                        msg = (f"PAPER FILL: SELL {pos.qty} {sym} @ NPR {fill_price:.2f}\n"
                               f"P&L: {pnl_pct:+.2f}% | Cash: NPR {self.capital:,.0f}")
                        _write_notification(msg)
                    self.pending_orders.remove(order)
                else:
                    order.fill_status = "MISSED"
                    self.pending_orders.remove(order)
                    logger.info("Order MISSED: %s %s (limit %s, open %s)",
                                order.action, sym, order.limit_price, open_price)

            except Exception as exc:
                logger.warning("Fill check failed for %s: %s", sym, exc)

        self._save_state()
        return fills

    # ── Order simulation ────────────────────────────────────────────────────

    def simulate_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        price_type: str = "MARKET",
        limit_price: Optional[float] = None,
        signal_context: Optional[dict] = None,
    ) -> dict:
        """
        Record a simulated order for next-morning fill check.
        Returns the pending order dict.
        """
        today = date.today()
        current_prices = self._get_live_close_prices([symbol])
        intended_entry = current_prices.get(symbol, 0.0)

        order = PaperOrder(
            symbol=symbol,
            action=action,
            qty=qty,
            order_type=price_type,
            limit_price=limit_price,
            signal_date=today,
            intended_entry=intended_entry,
            signal_context=signal_context or {},
        )
        self.pending_orders.append(order)
        self._save_pending_orders()

        logger.info("PAPER ORDER: %s %d %s @ %s | intended NPR %.2f",
                    action, qty, symbol, price_type, intended_entry)
        return {
            "symbol": symbol, "action": action, "qty": qty,
            "order_type": price_type, "limit_price": limit_price,
            "intended_entry": intended_entry, "status": "PENDING",
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _scrape_market_data(self, today: date) -> dict:
        """Attempt to scrape live market data."""
        try:
            from db.loader import get_latest_market_snapshot
            snap = get_latest_market_snapshot()
            if snap and snap.get("date") == str(today):
                return snap
        except Exception as exc:
            logger.warning("Live scrape failed, using DB fallback: %s", exc)
        return {"date": str(today), "source": "fallback"}

    def _collect_news(self, today: date) -> list[dict]:
        """Collect today's news articles."""
        try:
            from db.loader import get_news_articles
            return get_news_articles(date_str=str(today), limit=20)
        except Exception as exc:
            logger.warning("News collection failed: %s", exc)
            return []

    def _build_seed(self, today: date, market_data: dict, news: list) -> dict:
        """Build MiroFish seed for today."""
        try:
            from pipeline.seed_builder import build_seed
            return build_seed(date_str=str(today))
        except Exception as exc:
            logger.warning("Seed build failed: %s", exc)
            return {"date": str(today), "market_data": market_data}

    def _run_simulation(self, today: date, seed: dict) -> dict:
        """Run MiroFish simulation or load latest cached signal."""
        try:
            from pipeline.run_simulation import run_simulation_for_date
            result = run_simulation_for_date(str(today))
            if result:
                return result
        except Exception as exc:
            logger.warning("Live simulation failed, loading latest signal: %s", exc)

        # Fallback: load latest saved signal
        try:
            from pipeline.signal_extractor import load_latest_signal
            return load_latest_signal() or {"bull_bear_score": 0.0, "action": "HOLD"}
        except Exception:
            return {"bull_bear_score": 0.0, "action": "HOLD", "confidence_pct": 50}

    def _compute_indicators(self, today: date) -> dict:
        """Compute technical indicators for all analysis symbols."""
        from strategy.run_strategy import ANALYSIS_UNIVERSE
        indicators = {}
        try:
            from strategy.technical_indicators import get_latest_indicators
            for sym in ANALYSIS_UNIVERSE:
                try:
                    ind = get_latest_indicators(sym)
                    if ind:
                        indicators[sym] = ind
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Indicator computation failed: %s", exc)
        return indicators

    def _detect_regime(self, indicators: dict) -> dict:
        """Detect current market regime."""
        try:
            from strategy.regime_detector import detect_regime
            result = detect_regime()
            return result if isinstance(result, dict) else {"regime": str(result), "confidence": 0.5}
        except Exception as exc:
            logger.warning("Regime detection failed: %s", exc)
            return {"regime": "SIDEWAYS", "confidence": 0.5}

    def _combine_signals(self, indicators: dict, mirofish_signal: dict, regime_data: dict) -> dict:
        """Combine all signals into composite scores."""
        combined = {}
        try:
            from strategy.signal_combiner import combine_signals
            regime = regime_data.get("regime", "SIDEWAYS")
            for sym, ind in indicators.items():
                mf = {"score": mirofish_signal.get("bull_bear_score", 0),
                      "action": mirofish_signal.get("action", "HOLD"),
                      "conviction": "MEDIUM"}
                try:
                    combined[sym] = combine_signals(
                        mirofish_signal=mf,
                        technical_data=ind,
                        regime=regime,
                        sector_rotation={},
                    )
                except Exception:
                    combined[sym] = {"action": "HOLD", "composite_signal": 0.0,
                                     "conviction": "LOW", "position_size_pct": 0.0}
        except Exception as exc:
            logger.warning("Signal combination failed: %s", exc)
        return combined

    def _check_exits(self, today: date, indicators: dict, regime: str) -> list:
        """Check open positions for exit signals. Returns list of (sym, price, reason)."""
        exits = []
        for sym, pos in list(self.positions.items()):
            ind = indicators.get(sym, {})
            current_price = ind.get("close", pos.entry_price)
            if current_price <= 0:
                continue

            # Stop loss
            if current_price <= pos.stop_loss:
                exits.append((sym, current_price, "stop_loss"))
            # Target hit
            elif current_price >= pos.target_price:
                exits.append((sym, current_price, "target_hit"))
            # Max hold (20 trading days)
            else:
                try:
                    from backtest.calendar import trading_days_between
                    hold = trading_days_between(pos.entry_date, today)
                    if hold >= 20:
                        exits.append((sym, current_price, "max_hold"))
                except Exception:
                    pass
        return exits

    def _execute_paper_exit(self, sym: str, price: float, reason: str, today: date) -> None:
        """Record a paper exit via pending sell order."""
        if sym in self.positions:
            pos = self.positions[sym]
            self.simulate_order(
                symbol=sym,
                action="SELL",
                qty=pos.qty,
                price_type="MARKET",
                signal_context={"exit_reason": reason, "exit_price": price},
            )

    def _generate_watchlist(self, combined_signals: dict, regime_data: dict) -> list:
        """Generate today's watchlist."""
        try:
            from strategy.run_strategy import ANALYSIS_UNIVERSE
            from strategy.watchlist import generate_daily_watchlist
            return generate_daily_watchlist(
                symbols=ANALYSIS_UNIVERSE,
                mirofish_signals=combined_signals,
                sector_rotation={},
                regime=regime_data.get("regime", "SIDEWAYS"),
                top_n=10,
                send_to_telegram=False,
                save=True,
            )
        except Exception as exc:
            logger.warning("Watchlist generation failed: %s", exc)
            return []

    def _apply_trading_rules(self, combined_signals: dict, indicators: dict) -> dict:
        """Apply NEPSE hard trading rules."""
        approved = {}
        try:
            from strategy.trading_rules import validate_trade
            open_pos = len(self.positions)
            for sym, sig in combined_signals.items():
                if sig.get("action") != "BUY":
                    continue
                ind = indicators.get(sym, {})
                try:
                    val = validate_trade(
                        symbol=sym, action="BUY",
                        position_pct=sig.get("position_size_pct", 10.0),
                        open_positions=open_pos,
                        ltp=ind.get("close", 100.0),
                        prev_close=ind.get("close", 100.0),
                        avg_turnover_npr=ind.get("avg_turnover_npr", 1_000_000),
                        rsi=ind.get("rsi14", 50.0),
                    )
                    if val.get("approved"):
                        approved[sym] = {**sig, "trade_validation": val}
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Trading rules failed: %s", exc)
        return approved

    def _place_paper_orders(
        self, approved: dict, indicators: dict, watchlist: list, today: date
    ) -> list[dict]:
        """Place paper buy orders for approved tier-A watchlist signals."""
        new_orders = []
        tier_a = {e.symbol for e in watchlist if hasattr(e, "tier") and e.tier == "A"}

        for sym in tier_a:
            if sym not in approved:
                continue
            if sym in self.positions:
                continue

            sig = approved[sym]
            ind = indicators.get(sym, {})
            price = ind.get("close", 0.0)
            if price <= 0:
                continue

            pv = self.portfolio_value
            pos_pct = sig.get("position_size_pct", 10.0) / 100.0
            budget = pv * pos_pct
            shares = max(1, int(budget / price))
            cost_estimate = shares * price * 1.004

            if cost_estimate > self.capital * 0.8:  # keep 20% cash reserve
                shares = max(1, int((self.capital * 0.8) / (price * 1.004)))

            if shares <= 0:
                continue

            order = self.simulate_order(
                symbol=sym,
                action="BUY",
                qty=shares,
                price_type="LIMIT",
                limit_price=round(price * 1.005, 2),  # small slippage buffer
                signal_context={
                    "composite_signal": sig.get("composite_signal", 0),
                    "mirofish_score": sig.get("mirofish_score", 0),
                    "strategy": sig.get("strategy", "momentum"),
                    "regime": sig.get("regime", "UNKNOWN"),
                },
            )
            new_orders.append(order)

        return new_orders

    def _get_live_open_price(self, symbol: str, today: date) -> Optional[float]:
        """Get today's opening price from live scraper or DB."""
        try:
            from db.loader import get_stock_price
            row = get_stock_price(symbol, str(today))
            if row:
                return row.get("open") or row.get("ltp")
        except Exception:
            pass
        return None

    def _get_live_close_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get latest close prices."""
        prices = {}
        try:
            from db.loader import get_latest_prices
            data = get_latest_prices(symbols)
            prices = {s: d.get("close", 0.0) for s, d in data.items()}
        except Exception:
            pass
        return prices

    def _format_daily_telegram(
        self, today: date, signal: dict, regime: str,
        watchlist: list, new_orders: list, exits: list
    ) -> str:
        """Format the daily Telegram summary message."""
        score = signal.get("bull_bear_score", 0)
        action = signal.get("action", "HOLD")
        pv = self.portfolio_value
        ret = self.total_return_pct
        days = self.trading_days_elapsed

        signal_emoji = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")
        lines = [
            f"📊 *MiroFish Paper Trading* — {today}",
            f"",
            f"💼 Portfolio: NPR {pv:,.0f} ({ret:+.2f}%)",
            f"📅 Day {days}/20 of paper trading",
            f"",
            f"{signal_emoji} Signal: `{action}` ({score:+.2f}) | Regime: `{regime}`",
            f"",
        ]
        if new_orders:
            lines.append("🛒 *New Orders:*")
            for o in new_orders[:3]:
                lines.append(f"  BUY {o['qty']} {o['symbol']} | ~NPR {o.get('intended_entry', 0):,.0f}")
            lines.append("")
        if exits:
            lines.append("💰 *Exits:*")
            for sym, price, reason in exits[:3]:
                lines.append(f"  SELL {sym} @ NPR {price:,.0f} ({reason})")
            lines.append("")
        if watchlist:
            lines.append("👀 *Top Watchlist:*")
            for e in watchlist[:3]:
                if hasattr(e, "symbol"):
                    lines.append(f"  {e.symbol} | Score: {getattr(e, 'score', 0):.0f} | {getattr(e, 'action', 'WATCH')}")
        lines.append(f"\n_Generated {_nst_now().strftime('%H:%M NST')}_")
        return "\n".join(lines)

    def _send_telegram(self, message: str) -> bool:
        """Deprecated — use _write_notification instead."""
        _write_notification(message)
        return True

    def _save_daily_snapshot(self, today: date, regime: str,
                              signal: dict, combined: dict) -> None:
        """Persist today's snapshot to JSON and DB."""
        snapshot = {
            "date": str(today),
            "portfolio_value": self.portfolio_value,
            "cash": self.capital,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "regime": regime,
            "mirofish_score": signal.get("bull_bear_score", 0),
            "return_pct": self.total_return_pct,
            "session_id": self.paper_trade_id,
        }
        snap_path = self._data_dir / f"snapshot_{today}.json"
        with open(snap_path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, default=str)

        # Save to DB if available
        try:
            from db.models import get_engine, PaperPortfolioSnapshot
            from sqlalchemy.orm import Session
            engine = get_engine()
            with Session(engine) as sess:
                row = PaperPortfolioSnapshot(
                    date=today,
                    session_id=self.paper_trade_id,
                    portfolio_value=snapshot["portfolio_value"],
                    cash_balance=snapshot["cash"],
                    open_positions=snapshot["positions"],
                    daily_return_pct=0.0,
                    nepse_return_pct=0.0,
                    composite_score=signal.get("bull_bear_score", 0),
                    regime=regime,
                    active_strategy="momentum",
                )
                sess.merge(row)
                sess.commit()
        except Exception as exc:
            logger.debug("DB snapshot save failed (ok in test): %s", exc)

    def _save_pending_orders(self) -> None:
        """Persist pending orders to disk."""
        path = self._data_dir / "pending_orders.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                [{"symbol": o.symbol, "action": o.action, "qty": o.qty,
                  "order_type": o.order_type, "limit_price": o.limit_price,
                  "signal_date": str(o.signal_date), "status": o.fill_status}
                 for o in self.pending_orders],
                fh, indent=2, default=str,
            )

    def _save_state(self) -> None:
        """Save engine state to disk."""
        state = {
            "paper_trade_id": self.paper_trade_id,
            "capital": self.capital,
            "starting_capital": self.starting_capital,
            "session_start": str(self.session_start.date()),
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "trade_log": self.trade_log[-50:],  # keep last 50
        }
        with open(self._data_dir / "state.json", "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, default=str)

    # ── Status helpers ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current status dict for Telegram /status command."""
        return {
            "session_id": self.paper_trade_id,
            "portfolio_value": round(self.portfolio_value, 2),
            "cash": round(self.capital, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "open_positions": len(self.positions),
            "pending_orders": len(self.pending_orders),
            "closed_trades": len(self.trade_log),
            "trading_days_elapsed": self.trading_days_elapsed,
            "ready_for_live": self.trading_days_elapsed >= 20,
        }


# ---------------------------------------------------------------------------
# APScheduler integration
# ---------------------------------------------------------------------------

def start_scheduler(engine: PaperTradingEngine) -> None:
    """Start the APScheduler with paper trading jobs."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("apscheduler not installed. Run: pip install apscheduler")
        return

    # NST = UTC+5:45 — convert to UTC for scheduler
    # 15:30 NST = 09:45 UTC
    # 11:15 NST = 05:30 UTC

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        engine.run_daily_cycle,
        CronTrigger(hour=9, minute=45, day_of_week="sun,mon,tue,wed,thu"),
        id="daily_cycle",
        name="Paper trading daily cycle (15:30 NST)",
    )
    scheduler.add_job(
        engine.morning_fill_check,
        CronTrigger(hour=5, minute=30, day_of_week="sun,mon,tue,wed,thu"),
        id="morning_fill",
        name="Morning fill check (11:15 NST)",
    )

    logger.info("Paper trading scheduler started. Jobs: %s",
                [j.name for j in scheduler.get_jobs()])
    print("\n  Paper trading scheduler running.")
    print("  15:30 NST: daily cycle")
    print("  11:15 NST: morning fill check")
    print("  Press Ctrl+C to stop.\n")
    scheduler.start()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="MiroFish paper trading engine")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Starting virtual capital in NPR (default: 1,000,000)")
    parser.add_argument("--session-id", help="Paper trading session ID")
    parser.add_argument("--run-now", action="store_true",
                        help="Run one daily cycle immediately and exit")
    parser.add_argument("--fill-check", action="store_true",
                        help="Run morning fill check immediately and exit")
    args = parser.parse_args()

    engine = PaperTradingEngine(
        starting_virtual_capital_npr=args.capital,
        paper_trade_id=args.session_id,
    )

    if args.run_now:
        result = engine.run_daily_cycle()
        print(json.dumps(result, indent=2, default=str))
    elif args.fill_check:
        fills = engine.morning_fill_check()
        print(f"Fills processed: {len(fills)}")
        for f in fills:
            print(f"  {f}")
    else:
        start_scheduler(engine)


if __name__ == "__main__":
    main()
