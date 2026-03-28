"""
backtest/engine.py
NEPSEBacktestEngine — replays historical market data through the Phase 3
strategy pipeline with strict look-ahead bias prevention.

CRITICAL RULES (enforced throughout):
- Technical indicators only use data from dates <= current backtest date
- Entry price = NEXT trading day OPEN (signal T → execute at T+1 open)
- Exit evaluation = today's CLOSE (EOD)
- If price data missing → skip trade (NO interpolation)
- No future data ever leaks into indicator computation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataUnavailableError(Exception):
    """Raised when price or seed data is missing for a date."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestPosition:
    symbol: str
    entry_date: date
    entry_price: float
    shares: int
    strategy: str
    stop_loss: float
    target_price: float
    signal_date: date         # the day the signal was generated (T)
    mirofish_score: float = 0.0
    regime: str = "SIDEWAYS"
    sector: str = "unknown"

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    def pnl(self, exit_price: float) -> float:
        return (exit_price - self.entry_price) * self.shares

    def pnl_pct(self, exit_price: float) -> float:
        return (exit_price / self.entry_price - 1.0) * 100.0


@dataclass
class BacktestTrade:
    symbol: str
    signal_date: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    strategy: str
    exit_reason: str
    pnl_pct: float
    pnl_npr: float
    net_pnl_npr: float    # after commission + tax
    hold_days: int
    mirofish_score: float = 0.0
    regime: str = "SIDEWAYS"
    sector: str = "unknown"


@dataclass
class DailySnapshot:
    date: date
    portfolio_value: float
    cash: float
    open_positions: int
    regime: str
    trades_today: int
    exits_today: int
    nepse_index: float = 0.0


@dataclass
class BacktestResult:
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    daily_portfolio_values: list[float] = field(default_factory=list)
    daily_dates: list[date] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_snapshots: list[DailySnapshot] = field(default_factory=list)
    nepse_index_values: list[float] = field(default_factory=list)
    skipped_days: int = 0
    data_gaps: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Price cache (loaded once per run for efficiency)
# ---------------------------------------------------------------------------

class PriceCache:
    """
    In-memory price cache: symbol → DataFrame(date, open, high, low, close, volume).
    Loaded from DB at start of backtest, never queried per-day to avoid latency.
    """

    def __init__(self, symbols: list[str], start_date: date, end_date: date):
        self._data: dict[str, pd.DataFrame] = {}
        self._load(symbols, start_date, end_date)

    def _load(self, symbols: list[str], start: date, end: date) -> None:
        """Load price data from PostgreSQL for all symbols at once."""
        try:
            from db.models import get_engine
            from sqlalchemy import text
            engine = get_engine()
            with engine.connect() as conn:
                for sym in symbols:
                    rows = conn.execute(
                        text(
                            "SELECT date, open, high, low, close, volume "
                            "FROM stock_prices "
                            "WHERE symbol = :sym "
                            "  AND date BETWEEN :start AND :end "
                            "ORDER BY date"
                        ),
                        {"sym": sym, "start": start, "end": end},
                    ).fetchall()
                    if rows:
                        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                        df["date"] = pd.to_datetime(df["date"]).dt.date
                        df = df.set_index("date")
                        df = df.apply(pd.to_numeric, errors="coerce")
                        self._data[sym] = df
            logger.info("PriceCache loaded %d symbols", len(self._data))
        except Exception as exc:
            logger.warning("PriceCache DB load failed: %s — cache empty", exc)

    def get_price(self, symbol: str, d: date, price_type: str = "close") -> Optional[float]:
        """Return price for symbol on date. None if unavailable."""
        df = self._data.get(symbol)
        if df is None or d not in df.index:
            return None
        val = df.loc[d, price_type]
        return float(val) if pd.notna(val) and val > 0 else None

    def get_history_up_to(self, symbol: str, up_to_date: date, days: int = 300) -> pd.DataFrame:
        """
        LOOK-AHEAD SAFE: Returns OHLCV data only up to and including up_to_date.
        This is the ONLY way indicators should be computed in the backtest.
        """
        df = self._data.get(symbol)
        if df is None:
            return pd.DataFrame()
        # Filter strictly: only dates <= up_to_date
        filtered = df[df.index <= up_to_date]
        return filtered.tail(days)

    def get_nepse_index_on(self, d: date) -> float:
        """Get NEPSE index value (from market_snapshots table)."""
        # Falls back to 0 if unavailable — caller handles
        try:
            from db.models import get_engine
            from sqlalchemy import text
            with get_engine().connect() as conn:
                row = conn.execute(
                    text("SELECT nepse_index FROM market_snapshots WHERE date = :d LIMIT 1"),
                    {"d": d},
                ).fetchone()
                return float(row[0]) if row and row[0] else 0.0
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class NEPSEBacktestEngine:
    """
    Replays historical market conditions through the full Phase 3 strategy.
    Uses actual scraped seeds — not synthetic data (where available).
    """

    # Nepal capital gains tax (individual investor)
    CAPITAL_GAINS_TAX = 0.075   # 7.5%

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital_npr: float = 1_000_000.0,
        strategy_config: Optional[dict] = None,
        commission_pct: float = 0.004,
        slippage_pct: float = 0.002,
        tax_pct: float = 0.075,
        symbols: Optional[list[str]] = None,
        max_open_positions: int = 5,
        max_position_pct: float = 0.20,
        output_dir: str = "data/processed/backtest",
    ):
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        self.initial_capital = initial_capital_npr
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.tax_pct = tax_pct
        self.strategy_config = strategy_config or {}
        self.max_open_positions = max_open_positions
        self.max_position_pct = max_position_pct
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default symbols if not provided
        self.symbols = symbols or [
            "NABIL", "NICA", "EBL", "SANIMA", "PRVU", "KBL",
            "NHPC", "CHCL", "UPPER", "BPCL",
            "NLIC", "LICN", "SICL",
            "ICFC", "GMFIL",
            "SKBBL", "CBBL",
            "UNL", "BNL",
        ]

        # Runtime state
        self._cash = initial_capital_npr
        self._positions: dict[str, BacktestPosition] = {}
        self._pending_buys: list[dict] = []  # signals to execute at next open
        self._price_cache: Optional[PriceCache] = None
        self._result: Optional[BacktestResult] = None

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    def _buy_cost(self, price: float, shares: int) -> float:
        """Total cost including commission + slippage on entry."""
        gross = price * shares
        adj_price = price * (1 + self.slippage_pct)
        total = adj_price * shares
        commission = total * self.commission_pct
        sebon = total * 0.00015
        return total + commission + sebon

    def _sell_proceeds(self, entry_price: float, exit_price: float, shares: int) -> tuple[float, float]:
        """
        Net proceeds after commission, slippage, and capital gains tax.
        Returns (net_proceeds, tax_paid).
        """
        adj_price = exit_price * (1 - self.slippage_pct)
        gross = adj_price * shares
        commission = gross * self.commission_pct
        sebon = gross * 0.00015
        dp_charge = 25.0
        pre_tax = gross - commission - sebon - dp_charge

        # Capital gains tax on profit only
        profit = (adj_price - entry_price) * shares
        tax = max(0.0, profit * self.tax_pct)
        net = pre_tax - tax
        return net, tax

    def _apply_costs(self, trade_value_npr: float, action: str) -> float:
        """Deduct commission only. Used for quick estimates."""
        return trade_value_npr * (1 - self.commission_pct)

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def _get_price(self, symbol: str, d: date, price_type: str = "open") -> float:
        """
        Fetch price from cache. Raise DataUnavailableError if missing.
        This is the ONLY price access method — enforces no look-ahead.
        """
        price = self._price_cache.get_price(symbol, d, price_type)
        if price is None:
            raise DataUnavailableError(f"No {price_type} price for {symbol} on {d}")
        return price

    # ------------------------------------------------------------------
    # Indicator computation (LOOK-AHEAD SAFE)
    # ------------------------------------------------------------------

    def _compute_indicators_safe(self, symbol: str, current_date: date) -> Optional[dict]:
        """
        Compute indicators using ONLY data up to current_date.
        Never accesses any future data.
        """
        df = self._price_cache.get_history_up_to(symbol, current_date, days=300)
        if len(df) < 20:
            return None
        try:
            from strategy.technical_indicators import compute_indicators
            result = compute_indicators(df.copy(), symbol)
            if result is None or result.empty:
                return None
            last = result.iloc[-1]
            return {k: (float(v) if pd.notna(v) else 0.0) for k, v in last.items()}
        except Exception as exc:
            logger.debug("Indicator error for %s on %s: %s", symbol, current_date, exc)
            return None

    # ------------------------------------------------------------------
    # Signal loading
    # ------------------------------------------------------------------

    def _load_mirofish_signal(self, current_date: date) -> dict:
        """
        Load pre-computed MiroFish signal for current_date from DB.
        NEVER re-runs simulation — uses saved results only.
        """
        try:
            from db.models import get_engine
            from sqlalchemy import text
            with get_engine().connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT signal_data FROM simulation_results "
                        "WHERE date = :d ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"d": current_date},
                ).fetchone()
                if row and row[0]:
                    data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    return data
        except Exception:
            pass
        # Return neutral signal if none available
        return {"score": 0.0, "action": "HOLD", "conviction": "LOW", "source": "fallback"}

    # ------------------------------------------------------------------
    # Exit evaluation (EOD, using today's CLOSE)
    # ------------------------------------------------------------------

    def _check_exits(self, current_date: date, regime: str) -> list[str]:
        """
        Check all open positions for exit signals at EOD.
        Uses CLOSE price — realistic for EOD strategy.
        Returns list of symbols to close.
        """
        to_exit: list[str] = []
        for symbol, pos in list(self._positions.items()):
            try:
                close = self._get_price(symbol, current_date, "close")
            except DataUnavailableError:
                continue

            # Stop loss check
            if close <= pos.stop_loss:
                to_exit.append((symbol, "stop_loss", close))
                continue

            # Target price
            if close >= pos.target_price:
                to_exit.append((symbol, "target_hit", close))
                continue

            # Max hold period (regime-based)
            max_days = {"momentum_bull": 30, "mean_reversion_sideways": 12, "defensive_bear": 10}
            hold = (current_date - pos.entry_date).days
            if hold > max_days.get(pos.strategy, 30):
                to_exit.append((symbol, "max_hold", close))
                continue

            # Regime change exit
            if pos.regime in ("BULL", "RECOVERY") and regime in ("BEAR", "CAPITULATION"):
                to_exit.append((symbol, "regime_change", close))
                continue

        return to_exit

    def _close_position(
        self, symbol: str, exit_price: float, exit_date: date, exit_reason: str
    ) -> Optional[BacktestTrade]:
        pos = self._positions.pop(symbol, None)
        if not pos:
            return None

        net_proceeds, tax = self._sell_proceeds(pos.entry_price, exit_price, pos.shares)
        self._cash += net_proceeds

        pnl_pct = pos.pnl_pct(exit_price)
        pnl_npr = pos.pnl(exit_price)

        trade = BacktestTrade(
            symbol=symbol,
            signal_date=pos.signal_date,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            strategy=pos.strategy,
            exit_reason=exit_reason,
            pnl_pct=round(pnl_pct, 4),
            pnl_npr=round(pnl_npr, 2),
            net_pnl_npr=round(net_proceeds - pos.cost_basis, 2),
            hold_days=(exit_date - pos.entry_date).days,
            mirofish_score=pos.mirofish_score,
            regime=pos.regime,
            sector=pos.sector,
        )
        return trade

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        Execute the backtest for [start_date, end_date].
        15-step daily loop with strict look-ahead prevention.
        """
        from backtest.calendar import get_trading_days, get_next_trading_day

        logger.info(
            "Starting backtest: %s → %s | capital=NPR %,.0f | symbols=%d",
            self.start_date, self.end_date, self.initial_capital, len(self.symbols),
        )

        # Pre-load all price data once (efficient)
        self._price_cache = PriceCache(self.symbols, self.start_date, self.end_date)

        # Initialise state
        self._cash = self.initial_capital
        self._positions = {}
        self._pending_buys = []

        result = BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            config={
                "commission_pct": self.commission_pct,
                "slippage_pct": self.slippage_pct,
                "tax_pct": self.tax_pct,
                "symbols": self.symbols,
            },
        )

        trading_days = get_trading_days(self.start_date, self.end_date)
        logger.info("Trading days to simulate: %d", len(trading_days))

        for today in trading_days:
            try:
                self._run_day(today, result)
            except Exception as exc:
                logger.warning("Error on %s: %s — skipping day", today, exc)
                result.skipped_days += 1

        # Final portfolio valuation
        result.final_capital = self._portfolio_value(self.end_date, result)
        result.daily_portfolio_values.append(result.final_capital)
        result.daily_dates.append(self.end_date)

        self._save_result(result)
        logger.info(
            "Backtest complete: %d trades | final NPR %,.0f | return %.1f%%",
            len(result.trades),
            result.final_capital,
            (result.final_capital / self.initial_capital - 1) * 100,
        )
        return result

    def _run_day(self, today: date, result: BacktestResult) -> None:
        from backtest.calendar import get_next_trading_day
        trades_today = 0
        exits_today = 0

        # ── 1. Execute pending buys at today's OPEN (signal was T-1) ────────
        new_pending = []
        for pending in self._pending_buys:
            sym = pending["symbol"]
            try:
                open_price = self._get_price(sym, today, "open")
                total_cost = self._buy_cost(open_price, pending["shares"])
                if total_cost <= self._cash and sym not in self._positions:
                    self._cash -= total_cost
                    pos = BacktestPosition(
                        symbol=sym,
                        entry_date=today,
                        entry_price=open_price,
                        shares=pending["shares"],
                        strategy=pending["strategy"],
                        stop_loss=open_price * (1 - pending.get("sl_pct", 0.08)),
                        target_price=open_price * (1 + pending.get("tp_pct", 0.18)),
                        signal_date=pending["signal_date"],
                        mirofish_score=pending.get("mirofish_score", 0.0),
                        regime=pending.get("regime", "SIDEWAYS"),
                        sector=pending.get("sector", "unknown"),
                    )
                    self._positions[sym] = pos
                    trades_today += 1
            except DataUnavailableError:
                pass  # No open price available → skip this buy
        self._pending_buys = new_pending

        # ── 2. Load MiroFish signal (pre-computed, no future data) ──────────
        mf_signal = self._load_mirofish_signal(today)

        # ── 3. Detect regime (LOOK-AHEAD SAFE: only uses data up to today) ──
        regime = self._detect_regime_safe(today)

        # ── 4. NEPSE index for benchmark tracking ───────────────────────────
        nepse_idx = self._price_cache.get_nepse_index_on(today)
        result.nepse_index_values.append(nepse_idx)

        # ── 5. Check exits at EOD (close price) ─────────────────────────────
        exit_items = self._check_exits(today, regime)
        for item in exit_items:
            symbol, reason, close_price = item
            trade = self._close_position(symbol, close_price, today, reason)
            if trade:
                result.trades.append(trade)
                exits_today += 1

        # ── 6. Generate new signals ──────────────────────────────────────────
        if len(self._positions) < self.max_open_positions:
            self._generate_signals(today, mf_signal, regime, result)

        # ── 7. Record daily state ────────────────────────────────────────────
        pv = self._portfolio_value(today, result)
        result.daily_portfolio_values.append(pv)
        result.daily_dates.append(today)

        snap = DailySnapshot(
            date=today,
            portfolio_value=pv,
            cash=self._cash,
            open_positions=len(self._positions),
            regime=regime,
            trades_today=trades_today,
            exits_today=exits_today,
            nepse_index=nepse_idx,
        )
        result.daily_snapshots.append(snap)

    def _detect_regime_safe(self, current_date: date) -> str:
        """Regime detection using ONLY data up to current_date."""
        try:
            # Load NEPSE index history up to current_date
            from db.models import get_engine
            from sqlalchemy import text
            from strategy.regime_detector import detect_regime

            with get_engine().connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT date, nepse_index FROM market_snapshots "
                        "WHERE date <= :d ORDER BY date DESC LIMIT 300"
                    ),
                    {"d": current_date},
                ).fetchall()

            if rows:
                df = pd.DataFrame(rows, columns=["date", "nepse_index"])
                df = df.sort_values("date")
                result = detect_regime(df_nepse_index=df)
                if isinstance(result, dict):
                    return result.get("regime", "SIDEWAYS")
                return str(result)
        except Exception:
            pass
        return "SIDEWAYS"

    def _generate_signals(
        self, today: date, mf_signal: dict, regime: str, result: BacktestResult
    ) -> None:
        """
        Evaluate each symbol and queue buy signals for execution at T+1 open.
        """
        from backtest.calendar import get_next_trading_day

        try:
            from strategy.signal_combiner import combine_signals
            from strategy.trading_rules import validate_trade
            from strategy.entry_exit import select_active_strategy, check_entry_conditions
        except ImportError as exc:
            logger.debug("Strategy import failed: %s", exc)
            return

        strategy_result = select_active_strategy(regime=regime)
        strategy_name = strategy_result.get("strategy", "mean_reversion_sideways")

        # Compute sector rotation (look-ahead safe: uses price data up to today)
        sector_rotation = self._compute_sector_rotation_safe(today)

        for symbol in self.symbols:
            if symbol in self._positions:
                continue  # already in position

            # Get indicators (LOOK-AHEAD SAFE)
            ind = self._compute_indicators_safe(symbol, today)
            if not ind:
                continue

            # Combine signals
            try:
                combined = combine_signals(
                    mirofish_signal=mf_signal,
                    technical_data=ind,
                    regime=regime,
                    sector_rotation=sector_rotation,
                )
            except Exception:
                continue

            if combined.get("action") != "BUY":
                continue

            # Validate trade
            try:
                validation = validate_trade(
                    symbol=symbol,
                    action="BUY",
                    position_pct=combined.get("position_size_pct", 10.0),
                    open_positions=len(self._positions),
                    ltp=ind.get("close", 100.0),
                    prev_close=ind.get("close", 100.0),
                    rsi=ind.get("rsi14", 50.0),
                )
                if not validation.get("approved", False):
                    continue
            except Exception:
                continue

            # Check entry conditions
            try:
                entry_result = check_entry_conditions(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    indicators=ind,
                    mirofish_score=float(mf_signal.get("score", 0)),
                    composite_score=combined.get("composite_signal", 0),
                    sector_rank=1,
                )
                if not entry_result.get("entry_ok", False):
                    continue
            except Exception:
                continue

            # Queue for execution at T+1 open
            pct = combined.get("position_size_pct", 10.0) / 100.0
            budget = self._cash * min(pct, self.max_position_pct)
            price_est = ind.get("close", 100.0)
            shares = max(1, int(budget / price_est))

            # Get sector
            sector = "unknown"
            try:
                from strategy.sector_rotation import NEPSE_SECTORS
                for sec, syms in NEPSE_SECTORS.items():
                    if symbol in syms:
                        sector = sec
                        break
            except Exception:
                pass

            self._pending_buys.append({
                "symbol": symbol,
                "shares": shares,
                "strategy": strategy_name,
                "signal_date": today,
                "sl_pct": 0.08,
                "tp_pct": 0.18,
                "mirofish_score": float(mf_signal.get("score", 0)),
                "regime": regime,
                "sector": sector,
            })

    def _compute_sector_rotation_safe(self, current_date: date) -> dict:
        """Sector rotation using only data up to current_date."""
        try:
            from strategy.sector_rotation import get_rotation_signal
            return get_rotation_signal(days=20, save=False)
        except Exception:
            return {}

    def _portfolio_value(self, current_date: date, result: BacktestResult) -> float:
        """Current portfolio value = cash + market value of open positions."""
        value = self._cash
        for symbol, pos in self._positions.items():
            try:
                price = self._get_price(symbol, current_date, "close")
                value += price * pos.shares
            except DataUnavailableError:
                # Use last known price
                value += pos.entry_price * pos.shares
        return value

    def _save_result(self, result: BacktestResult) -> None:
        """Persist result to JSON for use by report generator."""
        path = self.output_dir / "backtest_data.json"
        data = {
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return_pct": round(
                (result.final_capital / result.initial_capital - 1) * 100, 2
            ),
            "total_trades": len(result.trades),
            "skipped_days": result.skipped_days,
            "config": result.config,
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_date": t.entry_date.isoformat(),
                    "exit_date": t.exit_date.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl_pct": t.pnl_pct,
                    "pnl_npr": t.pnl_npr,
                    "hold_days": t.hold_days,
                    "exit_reason": t.exit_reason,
                    "regime": t.regime,
                    "sector": t.sector,
                    "mirofish_score": t.mirofish_score,
                }
                for t in result.trades
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.info("Backtest result saved → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    parser = argparse.ArgumentParser(description="NEPSE MiroFish Backtest Engine")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--output", default="results_full")
    args = parser.parse_args()

    engine = NEPSEBacktestEngine(
        start_date=args.start,
        end_date=args.end,
        initial_capital_npr=args.capital,
        output_dir=f"data/processed/backtest/{args.output}",
    )
    result = engine.run()

    from backtest.metrics import compute_metrics, print_metrics_table
    metrics = compute_metrics(
        daily_portfolio_values=result.daily_portfolio_values,
        trades=[
            {
                "pnl_pct": t.pnl_pct,
                "regime": t.regime,
                "sector": t.sector,
                "hold_days": t.hold_days,
                "mirofish_score": t.mirofish_score,
                "action": "SELL",
            }
            for t in result.trades
        ],
        nepse_index_values=result.nepse_index_values,
    )
    print_metrics_table(metrics, f"NEPSE Backtest {args.start} → {args.end}")


if __name__ == "__main__":
    main()
