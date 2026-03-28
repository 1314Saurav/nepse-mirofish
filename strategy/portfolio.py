"""
strategy/portfolio.py
Paper-trading NEPSE portfolio tracker.
Persists positions and trade log to PostgreSQL (portfolio_positions + trade_log).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_engine():
    try:
        from db.models import get_engine
        return get_engine()
    except Exception as exc:
        logger.warning("DB engine unavailable: %s", exc)
        return None


def _ensure_tables(engine) -> None:
    """Create portfolio tables if they don't exist."""
    from sqlalchemy import text
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            id              SERIAL PRIMARY KEY,
            symbol          VARCHAR(20) NOT NULL,
            entry_date      DATE NOT NULL,
            entry_price     NUMERIC(12,4) NOT NULL,
            shares          INTEGER NOT NULL,
            strategy        VARCHAR(50),
            stop_loss       NUMERIC(12,4),
            target_price    NUMERIC(12,4),
            status          VARCHAR(20) DEFAULT 'OPEN',
            meta            JSONB,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS trade_log (
            id              SERIAL PRIMARY KEY,
            symbol          VARCHAR(20) NOT NULL,
            action          VARCHAR(10) NOT NULL,
            trade_date      DATE NOT NULL,
            price           NUMERIC(12,4) NOT NULL,
            shares          INTEGER NOT NULL,
            value_npr       NUMERIC(14,2),
            pnl_npr         NUMERIC(14,2),
            pnl_pct         NUMERIC(8,4),
            strategy        VARCHAR(50),
            exit_reason     VARCHAR(100),
            meta            JSONB,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
        """,
    ]
    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    entry_date: date
    entry_price: float
    shares: int
    strategy: str = "momentum_bull"
    stop_loss: float = 0.0
    target_price: float = 0.0
    status: str = "OPEN"        # OPEN / CLOSED
    db_id: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    def unrealised_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.shares

    def unrealised_pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (current_price / self.entry_price - 1.0) * 100.0


@dataclass
class TradeRecord:
    symbol: str
    action: str              # BUY / SELL
    trade_date: date
    price: float
    shares: int
    value_npr: float
    pnl_npr: float = 0.0
    pnl_pct: float = 0.0
    strategy: str = ""
    exit_reason: str = ""
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Portfolio class
# ---------------------------------------------------------------------------

class NEPSEPortfolio:
    """
    Paper-trading portfolio for NEPSE.

    Usage
    -----
    p = NEPSEPortfolio(initial_cash=500_000)
    p.open_position("NABIL", price=1200, shares=10, strategy="momentum_bull",
                    stop_loss=1104, target_price=1416)
    p.close_position("NABIL", price=1350, exit_reason="target_hit")
    print(p.get_summary())
    """

    BROKER_COMMISSION = 0.004    # 0.4% each side
    SEBON_FEE = 0.00015          # 0.015%
    DP_CHARGE = 25.0             # NPR per sell

    def __init__(self, initial_cash: float = 500_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}   # symbol → Position
        self._trade_log: list[TradeRecord] = []
        self._engine = _get_engine()
        if self._engine:
            try:
                _ensure_tables(self._engine)
                self._load_from_db()
            except Exception as exc:
                logger.warning("DB init failed, running in-memory only: %s", exc)

    # ------------------------------------------------------------------
    # Transaction costs
    # ------------------------------------------------------------------

    def _buy_cost(self, price: float, shares: int) -> float:
        """Total cost including broker + SEBON fees."""
        gross = price * shares
        commission = gross * self.BROKER_COMMISSION
        sebon = gross * self.SEBON_FEE
        return gross + commission + sebon

    def _sell_proceeds(self, price: float, shares: int) -> float:
        """Net proceeds after broker + SEBON + DP."""
        gross = price * shares
        commission = gross * self.BROKER_COMMISSION
        sebon = gross * self.SEBON_FEE
        return gross - commission - sebon - self.DP_CHARGE

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        price: float,
        shares: int,
        strategy: str = "momentum_bull",
        stop_loss: float = 0.0,
        target_price: float = 0.0,
        meta: Optional[dict] = None,
    ) -> bool:
        """
        Open a new paper-trade BUY.  Returns False if insufficient cash
        or position already open.
        """
        if symbol in self.positions and self.positions[symbol].status == "OPEN":
            logger.warning("Position already open for %s", symbol)
            return False

        total_cost = self._buy_cost(price, shares)
        if total_cost > self.cash:
            logger.warning(
                "Insufficient cash for %s: need %.2f, have %.2f",
                symbol, total_cost, self.cash,
            )
            return False

        self.cash -= total_cost

        pos = Position(
            symbol=symbol,
            entry_date=date.today(),
            entry_price=price,
            shares=shares,
            strategy=strategy,
            stop_loss=stop_loss if stop_loss > 0 else price * 0.92,
            target_price=target_price if target_price > 0 else price * 1.18,
            status="OPEN",
            meta=meta or {},
        )
        self.positions[symbol] = pos
        self._persist_position(pos)

        trade = TradeRecord(
            symbol=symbol,
            action="BUY",
            trade_date=date.today(),
            price=price,
            shares=shares,
            value_npr=total_cost,
            strategy=strategy,
            meta=meta or {},
        )
        self._trade_log.append(trade)
        self._persist_trade(trade)

        logger.info(
            "OPEN %s × %d @ %.2f | SL=%.2f TP=%.2f | cash_left=%.0f",
            symbol, shares, price, pos.stop_loss, pos.target_price, self.cash,
        )
        return True

    def close_position(
        self,
        symbol: str,
        price: float,
        exit_reason: str = "manual",
        meta: Optional[dict] = None,
    ) -> Optional[TradeRecord]:
        """
        Close an existing OPEN position.  Returns the TradeRecord or None.
        """
        pos = self.positions.get(symbol)
        if not pos or pos.status != "OPEN":
            logger.warning("No open position for %s", symbol)
            return None

        proceeds = self._sell_proceeds(price, pos.shares)
        self.cash += proceeds

        pnl_npr = proceeds - pos.cost_basis
        pnl_pct = (pnl_npr / pos.cost_basis) * 100.0 if pos.cost_basis > 0 else 0.0

        pos.status = "CLOSED"
        self._update_position_status(pos)

        trade = TradeRecord(
            symbol=symbol,
            action="SELL",
            trade_date=date.today(),
            price=price,
            shares=pos.shares,
            value_npr=proceeds,
            pnl_npr=round(pnl_npr, 2),
            pnl_pct=round(pnl_pct, 4),
            strategy=pos.strategy,
            exit_reason=exit_reason,
            meta=meta or {},
        )
        self._trade_log.append(trade)
        self._persist_trade(trade)

        logger.info(
            "CLOSE %s × %d @ %.2f | PnL=%.2f NPR (%.2f%%) | reason=%s",
            symbol, pos.shares, price, pnl_npr, pnl_pct, exit_reason,
        )
        return trade

    # ------------------------------------------------------------------
    # Portfolio queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.status == "OPEN"]

    def portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Total equity = cash + market value of open positions."""
        equity = self.cash
        for pos in self.get_open_positions():
            price = current_prices.get(pos.symbol, pos.entry_price)
            equity += price * pos.shares
        return equity

    def get_summary(self, current_prices: Optional[dict[str, float]] = None) -> dict:
        """Return a summary dict suitable for printing or JSON serialisation."""
        current_prices = current_prices or {}
        open_pos = self.get_open_positions()
        total_value = self.portfolio_value(current_prices)
        total_pnl = total_value - self.initial_cash
        total_pnl_pct = (total_pnl / self.initial_cash) * 100.0

        closed_trades = [t for t in self._trade_log if t.action == "SELL"]
        realised_pnl = sum(t.pnl_npr for t in closed_trades)
        win_trades = [t for t in closed_trades if t.pnl_npr > 0]
        win_rate = len(win_trades) / len(closed_trades) if closed_trades else 0.0

        positions_detail = []
        for pos in open_pos:
            cp = current_prices.get(pos.symbol, pos.entry_price)
            positions_detail.append({
                "symbol": pos.symbol,
                "entry_date": pos.entry_date.isoformat(),
                "entry_price": pos.entry_price,
                "current_price": cp,
                "shares": pos.shares,
                "cost_basis": round(pos.cost_basis, 2),
                "market_value": round(cp * pos.shares, 2),
                "unrealised_pnl": round(pos.unrealised_pnl(cp), 2),
                "unrealised_pnl_pct": round(pos.unrealised_pnl_pct(cp), 2),
                "stop_loss": pos.stop_loss,
                "target_price": pos.target_price,
                "strategy": pos.strategy,
            })

        return {
            "date": date.today().isoformat(),
            "initial_cash": self.initial_cash,
            "cash_remaining": round(self.cash, 2),
            "total_portfolio_value": round(total_value, 2),
            "total_pnl_npr": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "realised_pnl_npr": round(realised_pnl, 2),
            "open_positions_count": len(open_pos),
            "total_trades": len(closed_trades),
            "win_rate": round(win_rate * 100, 1),
            "open_positions": positions_detail,
        }

    def check_exit_conditions(
        self,
        current_prices: dict[str, float],
        current_rsi: Optional[dict[str, float]] = None,
        mirofish_scores: Optional[dict[str, float]] = None,
        regime: str = "SIDEWAYS",
    ) -> list[dict]:
        """
        Check every open position against exit rules.
        Returns list of {symbol, reason, current_price} dicts for positions
        that should be closed.
        """
        from strategy.entry_exit import check_exit_conditions as _check_exit
        exits = []
        for pos in self.get_open_positions():
            cp = current_prices.get(pos.symbol)
            if cp is None:
                continue

            # Highest price since entry (use current if not tracked)
            highest = max(cp, pos.entry_price)

            days_held = (date.today() - pos.entry_date).days
            rsi_val = (current_rsi or {}).get(pos.symbol, 50.0)
            mf_val = (mirofish_scores or {}).get(pos.symbol, 0.0)

            result = _check_exit(
                strategy_name=pos.strategy,
                entry_price=pos.entry_price,
                current_price=cp,
                highest_price_since=highest,
                days_held=days_held,
                current_rsi=rsi_val,
                mirofish_score=mf_val,
                current_regime=regime,
            )

            if result.get("exit"):
                exits.append({
                    "symbol": pos.symbol,
                    "reason": result.get("reason", "exit_signal"),
                    "current_price": cp,
                })

        return exits

    def get_trade_log(self) -> list[dict]:
        """Return all trade records as list of dicts."""
        return [asdict(t) for t in self._trade_log]

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _persist_position(self, pos: Position) -> None:
        if not self._engine:
            return
        from sqlalchemy import text
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text("""
                        INSERT INTO portfolio_positions
                            (symbol, entry_date, entry_price, shares, strategy,
                             stop_loss, target_price, status, meta)
                        VALUES (:sym, :ed, :ep, :sh, :st, :sl, :tp, :status, :meta)
                        RETURNING id
                    """),
                    {
                        "sym": pos.symbol,
                        "ed": pos.entry_date,
                        "ep": pos.entry_price,
                        "sh": pos.shares,
                        "st": pos.strategy,
                        "sl": pos.stop_loss,
                        "tp": pos.target_price,
                        "status": pos.status,
                        "meta": json.dumps(pos.meta),
                    },
                )
                pos.db_id = result.scalar()
        except Exception as exc:
            logger.warning("Failed to persist position %s: %s", pos.symbol, exc)

    def _update_position_status(self, pos: Position) -> None:
        if not self._engine or not pos.db_id:
            return
        from sqlalchemy import text
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("UPDATE portfolio_positions SET status=:s WHERE id=:id"),
                    {"s": pos.status, "id": pos.db_id},
                )
        except Exception as exc:
            logger.warning("Failed to update position status: %s", exc)

    def _persist_trade(self, trade: TradeRecord) -> None:
        if not self._engine:
            return
        from sqlalchemy import text
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO trade_log
                            (symbol, action, trade_date, price, shares, value_npr,
                             pnl_npr, pnl_pct, strategy, exit_reason, meta)
                        VALUES (:sym, :act, :td, :pr, :sh, :val, :pnl, :pct,
                                :st, :er, :meta)
                    """),
                    {
                        "sym": trade.symbol,
                        "act": trade.action,
                        "td": trade.trade_date,
                        "pr": trade.price,
                        "sh": trade.shares,
                        "val": trade.value_npr,
                        "pnl": trade.pnl_npr,
                        "pct": trade.pnl_pct,
                        "st": trade.strategy,
                        "er": trade.exit_reason,
                        "meta": json.dumps(trade.meta),
                    },
                )
        except Exception as exc:
            logger.warning("Failed to persist trade %s: %s", trade.symbol, exc)

    def _load_from_db(self) -> None:
        """Load open positions and recent trade log from DB on startup."""
        if not self._engine:
            return
        from sqlalchemy import text
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT id, symbol, entry_date, entry_price, shares, "
                        "strategy, stop_loss, target_price, status, meta "
                        "FROM portfolio_positions WHERE status='OPEN'"
                    )
                ).fetchall()

                for row in rows:
                    pos = Position(
                        symbol=row[1],
                        entry_date=row[2],
                        entry_price=float(row[3]),
                        shares=int(row[4]),
                        strategy=row[5] or "momentum_bull",
                        stop_loss=float(row[6] or 0),
                        target_price=float(row[7] or 0),
                        status=row[8],
                        db_id=row[0],
                        meta=json.loads(row[9]) if row[9] else {},
                    )
                    self.positions[pos.symbol] = pos

                # Estimate cash from trade log
                buy_rows = conn.execute(
                    text(
                        "SELECT SUM(value_npr) FROM trade_log WHERE action='BUY'"
                    )
                ).scalar() or 0
                sell_rows = conn.execute(
                    text(
                        "SELECT SUM(value_npr) FROM trade_log WHERE action='SELL'"
                    )
                ).scalar() or 0
                self.cash = self.initial_cash - float(buy_rows) + float(sell_rows)

                logger.info(
                    "Loaded %d open positions from DB | cash=%.0f",
                    len(rows), self.cash,
                )
        except Exception as exc:
            logger.warning("Failed to load portfolio from DB: %s", exc)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def load_portfolio(initial_cash: float = 500_000.0) -> NEPSEPortfolio:
    """Return an NEPSEPortfolio instance (loads DB state if available)."""
    return NEPSEPortfolio(initial_cash=initial_cash)
