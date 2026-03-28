"""
db/models.py

SQLAlchemy 2.0 ORM models for the nepse-mirofish database.

Tables
------
  market_snapshots    – daily NEPSE index snapshot
  stock_prices        – daily OHLCV + fundamentals per symbol
  news_articles       – scraped news with category + sentiment
  nrb_rates           – NRB policy rates and forex
  ipo_events          – IPO open/upcoming/allotted records
  simulation_seeds    – daily MiroFish seed JSON + brief
  mirofish_signals    – simulation output per day
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger, Column, Date, DateTime, Float, Index,
    Integer, JSON, Numeric, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ── 1. market_snapshots ────────────────────────────────────────────────────────

class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    date            = Column(Date, nullable=False, unique=True)
    nepse_index     = Column(Float)
    sensitive_index = Column(Float)
    float_index     = Column(Float)
    banking_index   = Column(Float)
    turnover_npr    = Column(Float)
    traded_shares   = Column(BigInteger)
    total_transactions = Column(Integer)
    scrips_traded   = Column(Integer)
    scraped_at      = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_market_snapshots_date", "date"),
    )


# ── 2. stock_prices ────────────────────────────────────────────────────────────

class StockPrice(Base):
    __tablename__ = "stock_prices"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    date       = Column(Date, nullable=False)
    symbol     = Column(String(20), nullable=False)
    ltp        = Column(Float)
    open       = Column(Float)
    high       = Column(Float)
    low        = Column(Float)
    close      = Column(Float)
    volume     = Column(BigInteger)
    pe_ratio   = Column(Float)
    eps        = Column(Float)
    market_cap = Column(Float)
    scraped_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_stock_prices_symbol_date"),
        Index("ix_stock_prices_date", "date"),
        Index("ix_stock_prices_symbol_date", "symbol", "date"),
    )


# ── 3. news_articles ───────────────────────────────────────────────────────────

class NewsArticle(Base):
    __tablename__ = "news_articles"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    published_at    = Column(DateTime(timezone=True), nullable=False)
    title           = Column(Text, nullable=False)
    url             = Column(Text, nullable=False, unique=True)
    source          = Column(String(100))
    category        = Column(String(50))
    body_excerpt    = Column(Text)
    sentiment_score = Column(Float)
    inserted_at     = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_news_articles_published_at", "published_at"),
        Index("ix_news_articles_category_published", "category", "published_at"),
    )


# ── 4. nrb_rates ──────────────────────────────────────────────────────────────

class NrbRate(Base):
    __tablename__ = "nrb_rates"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    date                 = Column(Date, nullable=False, unique=True)
    bank_rate            = Column(Float)
    repo_rate            = Column(Float)
    reverse_repo         = Column(Float)
    crr                  = Column(Float)
    slr                  = Column(Float)
    usd_npr_buy          = Column(Float)
    usd_npr_sell         = Column(Float)
    eur_npr_buy          = Column(Float)
    eur_npr_sell         = Column(Float)
    inr_npr_buy          = Column(Float)
    inr_npr_sell         = Column(Float)
    cny_npr_buy          = Column(Float)
    cny_npr_sell         = Column(Float)
    credit_deposit_ratio = Column(Float)
    scraped_at           = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_nrb_rates_date", "date"),
    )


# ── 5. ipo_events ──────────────────────────────────────────────────────────────

class IpoEvent(Base):
    __tablename__ = "ipo_events"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    company          = Column(String(200), nullable=False)
    issue_price      = Column(Float)
    open_date        = Column(Date)
    close_date       = Column(Date)
    status           = Column(String(20))    # open | upcoming | allotted
    allotment_ratio  = Column(String(50))    # e.g. "1:5" or percentage string
    scraped_at       = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("company", "open_date", name="uq_ipo_events_company_open"),
        Index("ix_ipo_events_status", "status"),
        Index("ix_ipo_events_open_date", "open_date"),
    )


# ── 6. simulation_seeds ────────────────────────────────────────────────────────

class SimulationSeed(Base):
    __tablename__ = "simulation_seeds"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    date                = Column(Date, nullable=False, unique=True)
    seed_json           = Column(JSONB, nullable=False)
    brief_text          = Column(Text)
    simulation_question = Column(Text)
    created_at          = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_simulation_seeds_date", "date"),
    )


# ── 7. mirofish_signals ────────────────────────────────────────────────────────

class MirofishSignal(Base):
    __tablename__ = "mirofish_signals"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    date                 = Column(Date, nullable=False, unique=True)
    bull_bear_score      = Column(Float)           # -1.0 (full bear) to +1.0 (full bull)
    confidence_pct       = Column(Float)
    sector_breakdown     = Column(JSONB)            # {sector: score, ...}
    raw_simulation_output = Column(JSONB)
    created_at           = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_mirofish_signals_date", "date"),
    )


# ── 8. portfolio_positions ─────────────────────────────────────────────────────

class PortfolioPosition(Base):
    """Paper-trading open/closed positions."""
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    entry_date = Column(Date, nullable=False)
    entry_price = Column(Numeric(12, 4), nullable=False)
    shares = Column(Integer, nullable=False)
    strategy = Column(String(50))
    stop_loss = Column(Numeric(12, 4))
    target_price = Column(Numeric(12, 4))
    status = Column(String(20), default="OPEN", index=True)
    meta = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<PortfolioPosition {self.symbol} {self.status} @ {self.entry_price}>"


# ── 9. trade_log ───────────────────────────────────────────────────────────────

class TradeLog(Base):
    """Record of every paper trade (BUY and SELL)."""
    __tablename__ = "trade_log"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(10), nullable=False)       # BUY / SELL
    trade_date = Column(Date, nullable=False, index=True)
    price = Column(Numeric(12, 4), nullable=False)
    shares = Column(Integer, nullable=False)
    value_npr = Column(Numeric(14, 2))
    pnl_npr = Column(Numeric(14, 2))
    pnl_pct = Column(Numeric(8, 4))
    strategy = Column(String(50))
    exit_reason = Column(String(100))
    meta = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<TradeLog {self.action} {self.symbol} @ {self.price}>"


# ── Engine helper (used in tests / CLI) ───────────────────────────────────────

def get_engine(db_url: Optional[str] = None):
    """Create a SQLAlchemy engine from NEPSE_DB_URL env or explicit url."""
    import os
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    load_dotenv()
    url = db_url or os.getenv("NEPSE_DB_URL", "")
    if not url:
        raise RuntimeError("NEPSE_DB_URL is not set.")
    return create_engine(url, pool_pre_ping=True)


def create_all_tables(db_url: Optional[str] = None) -> None:
    """Create all tables (used for quick local setup without Alembic)."""
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    print("All tables created.")


# ── Paper trading models ─────────────────────────────────────────────────────

from sqlalchemy import Boolean, ForeignKey
from sqlalchemy import UniqueConstraint as _UC

class PaperTradingSession(Base):
    """Metadata for each paper trading session."""
    __tablename__ = "paper_trading_sessions"

    id               = Column(String(50), primary_key=True)
    start_date       = Column(Date, nullable=False)
    end_date         = Column(Date, nullable=True)
    starting_capital = Column(Float, nullable=False)
    current_value    = Column(Float, nullable=False)
    status           = Column(String(10), default="ACTIVE")   # ACTIVE | COMPLETED | ABORTED
    config_snapshot  = Column(JSON, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_paper_sessions_status", "status"),
    )


class PaperOrder(Base):
    """Individual paper trading orders."""
    __tablename__ = "paper_orders"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    session_id      = Column(String(50), nullable=False)
    signal_date     = Column(Date)
    symbol          = Column(String(10))
    action          = Column(String(4))    # BUY | SELL
    qty             = Column(Integer)
    order_type      = Column(String(6))    # MARKET | LIMIT
    limit_price     = Column(Float, nullable=True)
    intended_entry  = Column(Float)
    fill_price      = Column(Float, nullable=True)
    fill_date       = Column(Date, nullable=True)
    fill_status     = Column(String(8), default="PENDING")   # PENDING | FILLED | MISSED
    signal_score    = Column(Float, nullable=True)
    strategy_mode   = Column(String(20), nullable=True)
    regime          = Column(String(10), nullable=True)
    mirofish_score  = Column(Float, nullable=True)
    technical_score = Column(Float, nullable=True)
    notes           = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_paper_orders_session_date", "session_id", "signal_date"),
        Index("ix_paper_orders_symbol", "symbol"),
    )


class PaperPortfolioSnapshot(Base):
    """Daily portfolio snapshot for paper trading sessions."""
    __tablename__ = "paper_portfolio_snapshots"

    date             = Column(Date, primary_key=True)
    session_id       = Column(String(50), primary_key=True)
    portfolio_value  = Column(Float)
    cash_balance     = Column(Float)
    open_positions   = Column(JSON, nullable=True)
    daily_return_pct = Column(Float, nullable=True)
    nepse_return_pct = Column(Float, nullable=True)
    composite_score  = Column(Float, nullable=True)
    regime           = Column(String(10), nullable=True)
    active_strategy  = Column(String(20), nullable=True)

    __table_args__ = (
        Index("ix_paper_snapshots_session", "session_id"),
    )
