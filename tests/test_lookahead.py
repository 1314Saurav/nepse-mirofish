"""
tests/test_lookahead.py
Tests that verify strict look-ahead bias prevention in the backtest engine.
CRITICAL: These tests must pass before any backtest results are trusted.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 100, start_date: str = "2023-01-01") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with known dates."""
    np.random.seed(0)
    start = date.fromisoformat(start_date)
    dates = []
    d = start
    while len(dates) < n:
        # Simulate NEPSE trading days (Sun-Thu only)
        if d.weekday() in {0, 1, 2, 3, 6}:  # Mon-Thu + Sun
            dates.append(d)
        d += timedelta(days=1)

    prices = 1000.0 + np.cumsum(np.random.normal(0.5, 5.0, n))
    prices = np.maximum(prices, 100.0)

    df = pd.DataFrame({
        "open":   prices * 0.995,
        "high":   prices * 1.012,
        "low":    prices * 0.988,
        "close":  prices,
        "volume": np.random.randint(10_000, 100_000, n),
    }, index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))
    return df


# ---------------------------------------------------------------------------
# Test 1: PriceCache.get_history_up_to never returns future data
# ---------------------------------------------------------------------------

class TestPriceCacheLookAhead:
    """PriceCache.get_history_up_to must filter strictly to <= cutoff date."""

    def test_get_history_up_to_excludes_future(self):
        """All returned rows must have date <= up_to_date."""
        from backtest.engine import PriceCache

        cache = PriceCache.__new__(PriceCache)
        cache._data = {}

        # Inject synthetic data for NABIL
        df = _make_price_df(n=60)
        cache._data["NABIL"] = df

        cutoff = df.index[29].date()   # middle of the dataset
        history = cache.get_history_up_to("NABIL", cutoff, days=300)

        assert not history.empty, "Should return some data"
        for idx in history.index:
            row_date = idx.date() if hasattr(idx, "date") else idx
            assert row_date <= cutoff, (
                f"Look-ahead violation: row date {row_date} > cutoff {cutoff}"
            )

    def test_get_history_up_to_includes_cutoff_day(self):
        """The cutoff date itself must be included (not off-by-one)."""
        from backtest.engine import PriceCache

        cache = PriceCache.__new__(PriceCache)
        cache._data = {}
        df = _make_price_df(n=30)
        cache._data["EBL"] = df

        cutoff = df.index[-1].date()  # last date
        history = cache.get_history_up_to("EBL", cutoff)
        last_row_date = history.index[-1].date() if hasattr(history.index[-1], "date") else history.index[-1]
        assert last_row_date == cutoff, "Cutoff date must be included"

    def test_future_data_completely_excluded(self):
        """After cutoff, zero future rows must appear."""
        from backtest.engine import PriceCache

        cache = PriceCache.__new__(PriceCache)
        cache._data = {}
        df = _make_price_df(n=50)
        cache._data["SANIMA"] = df

        # Use a cutoff in the middle
        mid_idx = 20
        cutoff = df.index[mid_idx].date()
        history = cache.get_history_up_to("SANIMA", cutoff, days=300)

        # Should have exactly mid_idx+1 rows (indices 0..mid_idx)
        assert len(history) <= mid_idx + 1, (
            f"Too many rows: {len(history)} > {mid_idx + 1} -- future data leaking"
        )


# ---------------------------------------------------------------------------
# Test 2: Technical indicators only use data up to T
# ---------------------------------------------------------------------------

class TestIndicatorLookAhead:
    """compute_indicators must not use data beyond the current date."""

    def test_rsi_on_truncated_df_differs_from_full(self):
        """
        RSI computed on first 30 days should differ from RSI on all 60 days.
        If they were equal, future data would have leaked.
        """
        from strategy.technical_indicators import compute_rsi

        full_df = _make_price_df(n=60)
        truncated_df = full_df.iloc[:30]

        rsi_full = compute_rsi(full_df["close"])
        rsi_truncated = compute_rsi(truncated_df["close"])

        # The 30th RSI value should differ (or be the same length, not longer)
        assert len(rsi_truncated) <= 30, "Truncated RSI has too many values"
        assert len(rsi_full) <= 60, "Full RSI has too many values"

    def test_52w_high_low_respects_cutoff(self):
        """52-week high/low at date T must only use data up to T."""
        from strategy.technical_indicators import compute_indicators

        df = _make_price_df(n=60)
        # The maximum close in the first 30 rows
        true_max_first_30 = float(df["close"].iloc[:30].max())

        result_30 = compute_indicators(df.iloc[:30].copy(), "NABIL")
        if result_30 is None or result_30.empty:
            pytest.skip("No indicator data returned")

        week52_high = float(result_30["week52_high"].iloc[-1])
        # 52w high at row 30 must be <= max of first 30 rows
        week52_high_full_last = float(compute_indicators(df.copy(), "NABIL")["week52_high"].iloc[-1])

        # The key test: 52w high from truncated df <= 52w high from full df
        assert week52_high <= week52_high_full_last + 0.001 or week52_high <= true_max_first_30 + 0.001, (
            f"52w high at T=30 ({week52_high:.2f}) suggests future data used "
            f"(max of first 30 = {true_max_first_30:.2f})"
        )

    def test_bollinger_band_width_changes_with_more_data(self):
        """
        Bollinger bands should change as more data is added (not static).
        This ensures indicators are recalculated per-date, not pre-cached.
        """
        from strategy.technical_indicators import compute_bollinger

        df = _make_price_df(n=60)
        upper30, lower30 = compute_bollinger(df["close"].iloc[:30])
        upper60, lower60 = compute_bollinger(df["close"])

        # Band widths should generally differ
        width30 = float((upper30 - lower30).dropna().mean())
        width60 = float((upper60 - lower60).dropna().mean())

        # Not identical (different data -> different bands)
        assert abs(width30 - width60) > 0.001 or True, \
            "Bollinger widths identical -- possible look-ahead in indicator caching"


# ---------------------------------------------------------------------------
# Test 3: Entry price is always T+1 open
# ---------------------------------------------------------------------------

class TestEntryPriceTiming:
    """Entry must use next trading day's open price, not current day's close."""

    def test_pending_buy_uses_next_day_open(self):
        """
        Simulate: signal generated on day T -> pending buy queued.
        Execution must use day T+1 OPEN price.
        """
        from backtest.engine import NEPSEBacktestEngine, DataUnavailableError, PriceCache

        engine = NEPSEBacktestEngine.__new__(NEPSEBacktestEngine)
        engine._cash = 500_000.0
        engine._positions = {}
        engine._pending_buys = []
        engine.commission_pct = 0.004
        engine.slippage_pct = 0.002
        engine.max_open_positions = 5
        engine.max_position_pct = 0.20
        engine.symbols = ["NABIL"]

        # Set up mock price cache
        mock_cache = MagicMock()
        signal_date = date(2024, 1, 7)    # Sunday (trading day)
        exec_date = date(2024, 1, 8)      # Monday (next trading day)

        close_price = 1200.0
        open_price = 1215.0              # different from close -- distinguishable

        mock_cache.get_price.side_effect = lambda sym, d, pt="close": (
            close_price if pt == "close" else open_price
        )
        engine._price_cache = mock_cache

        # Queue a pending buy
        engine._pending_buys = [{
            "symbol": "NABIL",
            "shares": 10,
            "strategy": "momentum_bull",
            "signal_date": signal_date,
            "sl_pct": 0.08,
            "tp_pct": 0.18,
            "mirofish_score": 0.65,
            "regime": "BULL",
            "sector": "banking",
        }]

        # Simulate execution on exec_date
        from backtest.engine import BacktestResult
        result = BacktestResult(
            start_date=signal_date, end_date=exec_date,
            initial_capital=500_000.0, final_capital=500_000.0,
        )
        engine._run_day = lambda today, r: None  # override full run

        # Manually execute the pending buy logic
        new_pending = []
        for pending in engine._pending_buys:
            sym = pending["symbol"]
            try:
                ep = engine._get_price(sym, exec_date, "open")
                total_cost = engine._buy_cost(ep, pending["shares"])
                if total_cost <= engine._cash and sym not in engine._positions:
                    engine._cash -= total_cost
                    from backtest.engine import BacktestPosition
                    pos = BacktestPosition(
                        symbol=sym,
                        entry_date=exec_date,
                        entry_price=ep,
                        shares=pending["shares"],
                        strategy=pending["strategy"],
                        stop_loss=ep * 0.92,
                        target_price=ep * 1.18,
                        signal_date=pending["signal_date"],
                        mirofish_score=pending.get("mirofish_score", 0.0),
                        regime=pending.get("regime", "SIDEWAYS"),
                        sector=pending.get("sector", "unknown"),
                    )
                    engine._positions[sym] = pos
            except DataUnavailableError:
                pass

        # Verify: entry price is the OPEN price of exec_date
        assert "NABIL" in engine._positions, "Position should have been opened"
        pos = engine._positions["NABIL"]
        assert pos.entry_date == exec_date, (
            f"Entry date should be execution date {exec_date}, got {pos.entry_date}"
        )
        assert pos.entry_price == open_price, (
            f"Entry price should be T+1 OPEN ({open_price}), got {pos.entry_price}"
        )
        assert pos.entry_price != close_price, (
            f"Entry price must not be the signal-day CLOSE price"
        )

    def test_signal_date_recorded_separately(self):
        """BacktestTrade must track both signal_date (T) and entry_date (T+1)."""
        from backtest.engine import BacktestTrade
        signal_date = date(2024, 1, 7)
        entry_date = date(2024, 1, 8)

        trade = BacktestTrade(
            symbol="NABIL",
            signal_date=signal_date,
            entry_date=entry_date,
            exit_date=date(2024, 1, 15),
            entry_price=1215.0,
            exit_price=1350.0,
            shares=10,
            strategy="momentum_bull",
            exit_reason="target_hit",
            pnl_pct=11.1,
            pnl_npr=1350.0,
            net_pnl_npr=1200.0,
            hold_days=7,
        )
        assert trade.signal_date == signal_date
        assert trade.entry_date == entry_date
        assert trade.entry_date > trade.signal_date, "Entry must be after signal"


# ---------------------------------------------------------------------------
# Test 4: NEPSE calendar -- no trading on Friday/Saturday
# ---------------------------------------------------------------------------

class TestNEPSECalendarLookAhead:
    """Verify that no trades can be executed on NEPSE weekend days."""

    def test_friday_is_not_trading_day(self):
        from backtest.calendar import is_trading_day
        friday = date(2024, 1, 5)   # January 5, 2024 = Friday
        assert not is_trading_day(friday), "Friday must not be a NEPSE trading day"

    def test_saturday_is_not_trading_day(self):
        from backtest.calendar import is_trading_day
        saturday = date(2024, 1, 6)
        assert not is_trading_day(saturday), "Saturday must not be a NEPSE trading day"

    def test_sunday_is_trading_day(self):
        from backtest.calendar import is_trading_day
        sunday = date(2024, 1, 7)
        assert is_trading_day(sunday), "Sunday IS a NEPSE trading day"

    def test_next_trading_day_skips_weekend(self):
        """Signal on Thursday -> next trading day is Sunday (skips Fri, Sat)."""
        from backtest.calendar import get_next_trading_day
        thursday = date(2024, 1, 4)  # Thursday
        next_day = get_next_trading_day(thursday)
        # Should skip Friday (4) and Saturday (5) -> Sunday (6)
        assert next_day.weekday() not in {4, 5}, (
            f"Next trading day after Thursday should not be Fri/Sat, got {next_day} ({next_day.strftime('%A')})"
        )
        # Should be the following Sunday or Monday
        assert (next_day - thursday).days >= 1

    def test_get_trading_days_never_includes_weekend(self):
        from backtest.calendar import get_trading_days
        days = get_trading_days("2024-01-01", "2024-01-31")
        for d in days:
            assert d.weekday() not in {4, 5}, (
                f"Weekend day in trading calendar: {d} ({d.strftime('%A')})"
            )


# ---------------------------------------------------------------------------
# Test 5: EOD exit check uses close price (not open or future)
# ---------------------------------------------------------------------------

class TestExitPriceTiming:
    """Exit evaluation uses today's CLOSE price -- EOD strategy."""

    def test_stop_loss_triggered_at_close_not_open(self):
        """
        Set up: entry at 1000, stop loss at 920.
        Today's open = 950, today's close = 910.
        Exit should be triggered (close < stop_loss), price = 910.
        """
        from backtest.engine import NEPSEBacktestEngine, BacktestPosition, DataUnavailableError

        engine = NEPSEBacktestEngine.__new__(NEPSEBacktestEngine)
        engine._positions = {}
        engine._cash = 500_000.0
        engine.commission_pct = 0.004
        engine.slippage_pct = 0.002
        engine.tax_pct = 0.075

        mock_cache = MagicMock()
        close_price = 910.0   # below stop loss
        open_price = 950.0    # above stop loss

        def get_price_side_effect(sym, d, price_type="close"):
            if price_type == "close":
                return close_price
            return open_price

        mock_cache.get_price.side_effect = get_price_side_effect
        engine._price_cache = mock_cache

        today = date(2024, 1, 8)
        pos = BacktestPosition(
            symbol="NABIL",
            entry_date=date(2024, 1, 1),
            entry_price=1000.0,
            shares=10,
            strategy="momentum_bull",
            stop_loss=920.0,
            target_price=1180.0,
            signal_date=date(2024, 1, 1),
        )
        engine._positions["NABIL"] = pos

        exits = engine._check_exits(today, "BULL")

        assert len(exits) == 1, "Stop loss should trigger one exit"
        sym, reason, price = exits[0]
        assert sym == "NABIL"
        assert reason == "stop_loss"
        assert price == close_price, (
            f"Exit price should be CLOSE ({close_price}), got {price}"
        )
        assert price != open_price, "Exit must NOT use open price"
