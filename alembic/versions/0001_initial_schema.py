"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-03-25

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── market_snapshots ──────────────────────────────────────────────────────
    op.create_table(
        "market_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("nepse_index", sa.Float()),
        sa.Column("sensitive_index", sa.Float()),
        sa.Column("float_index", sa.Float()),
        sa.Column("banking_index", sa.Float()),
        sa.Column("turnover_npr", sa.Float()),
        sa.Column("traded_shares", sa.BigInteger()),
        sa.Column("total_transactions", sa.Integer()),
        sa.Column("scrips_traded", sa.Integer()),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date"),
    )
    op.create_index("ix_market_snapshots_date", "market_snapshots", ["date"])

    # ── stock_prices ──────────────────────────────────────────────────────────
    op.create_table(
        "stock_prices",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("ltp", sa.Float()),
        sa.Column("open", sa.Float()),
        sa.Column("high", sa.Float()),
        sa.Column("low", sa.Float()),
        sa.Column("close", sa.Float()),
        sa.Column("volume", sa.BigInteger()),
        sa.Column("pe_ratio", sa.Float()),
        sa.Column("eps", sa.Float()),
        sa.Column("market_cap", sa.Float()),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "date", name="uq_stock_prices_symbol_date"),
    )
    op.create_index("ix_stock_prices_date", "stock_prices", ["date"])
    op.create_index("ix_stock_prices_symbol_date", "stock_prices", ["symbol", "date"])

    # ── news_articles ─────────────────────────────────────────────────────────
    op.create_table(
        "news_articles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source", sa.String(100)),
        sa.Column("category", sa.String(50)),
        sa.Column("body_excerpt", sa.Text()),
        sa.Column("sentiment_score", sa.Float()),
        sa.Column("inserted_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url"),
    )
    op.create_index("ix_news_articles_published_at", "news_articles", ["published_at"])
    op.create_index(
        "ix_news_articles_category_published",
        "news_articles", ["category", "published_at"],
    )

    # ── nrb_rates ─────────────────────────────────────────────────────────────
    op.create_table(
        "nrb_rates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("bank_rate", sa.Float()),
        sa.Column("repo_rate", sa.Float()),
        sa.Column("reverse_repo", sa.Float()),
        sa.Column("crr", sa.Float()),
        sa.Column("slr", sa.Float()),
        sa.Column("usd_npr_buy", sa.Float()),
        sa.Column("usd_npr_sell", sa.Float()),
        sa.Column("eur_npr_buy", sa.Float()),
        sa.Column("eur_npr_sell", sa.Float()),
        sa.Column("inr_npr_buy", sa.Float()),
        sa.Column("inr_npr_sell", sa.Float()),
        sa.Column("cny_npr_buy", sa.Float()),
        sa.Column("cny_npr_sell", sa.Float()),
        sa.Column("credit_deposit_ratio", sa.Float()),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date"),
    )
    op.create_index("ix_nrb_rates_date", "nrb_rates", ["date"])

    # ── ipo_events ────────────────────────────────────────────────────────────
    op.create_table(
        "ipo_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company", sa.String(200), nullable=False),
        sa.Column("issue_price", sa.Float()),
        sa.Column("open_date", sa.Date()),
        sa.Column("close_date", sa.Date()),
        sa.Column("status", sa.String(20)),
        sa.Column("allotment_ratio", sa.String(50)),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company", "open_date", name="uq_ipo_events_company_open"),
    )
    op.create_index("ix_ipo_events_status", "ipo_events", ["status"])
    op.create_index("ix_ipo_events_open_date", "ipo_events", ["open_date"])

    # ── simulation_seeds ──────────────────────────────────────────────────────
    op.create_table(
        "simulation_seeds",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("seed_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("brief_text", sa.Text()),
        sa.Column("simulation_question", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date"),
    )
    op.create_index("ix_simulation_seeds_date", "simulation_seeds", ["date"])

    # ── mirofish_signals ──────────────────────────────────────────────────────
    op.create_table(
        "mirofish_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("bull_bear_score", sa.Float()),
        sa.Column("confidence_pct", sa.Float()),
        sa.Column("sector_breakdown", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("raw_simulation_output", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date"),
    )
    op.create_index("ix_mirofish_signals_date", "mirofish_signals", ["date"])


def downgrade() -> None:
    op.drop_table("mirofish_signals")
    op.drop_table("simulation_seeds")
    op.drop_table("ipo_events")
    op.drop_table("nrb_rates")
    op.drop_table("news_articles")
    op.drop_table("stock_prices")
    op.drop_table("market_snapshots")
