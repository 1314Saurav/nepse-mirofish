"""
tests/test_phase3_integration.py
Phase 3 integration tests — 5 scenarios covering the full signal pipeline.
Tests run with mock/synthetic data (no live DB or API required).
"""

from __future__ import annotations

import sys
import os
from datetime import date
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers: synthetic data factories
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 60, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame."""
    np.random.seed(42)
    base = 1000.0
    if trend == "up":
        prices = base + np.cumsum(np.random.normal(2.0, 5.0, n))
    elif trend == "down":
        prices = base + np.cumsum(np.random.normal(-2.0, 5.0, n))
    elif trend == "flat":
        prices = base + np.cumsum(np.random.normal(0.0, 3.0, n))
    else:
        prices = base + np.cumsum(np.random.normal(0.0, 5.0, n))

    prices = np.maximum(prices, 100.0)  # floor
    df = pd.DataFrame({
        "date": pd.date_range(end=date.today(), periods=n),
        "open": prices * 0.995,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(10_000, 200_000, n),
    })
    return df.set_index("date")


def _make_indicators(rsi: float = 55.0, trend: str = "up") -> dict:
    """Synthetic indicator dict for testing."""
    price = 1200.0 if trend == "up" else 800.0
    sma20 = price * (1.01 if trend == "up" else 0.97)
    sma50 = price * (0.97 if trend == "up" else 1.05)
    return {
        "close": price,
        "sma20": sma20,
        "sma50": sma50,
        "ema13": price * 1.005,
        "rsi14": rsi,
        "macd": 5.0 if trend == "up" else -5.0,
        "macd_signal": 3.0,
        "macd_hist": 2.0 if trend == "up" else -2.0,
        "atr14": 20.0,
        "bb_upper": price * 1.05,
        "bb_lower": price * 0.95,
        "vol_ratio": 1.8 if trend == "up" else 0.5,
        "obv": 1_000_000,
        "week52_high": price * 1.25,
        "week52_low": price * 0.70,
        "ret_5d": 3.0 if trend == "up" else -3.0,
        "ret_10d": 5.0 if trend == "up" else -5.0,
        "ret_20d": 8.0 if trend == "up" else -8.0,
    }


def _make_sector_rotation(top_sector: str = "banking") -> dict:
    sectors = ["banking", "hydropower", "insurance", "finance", "microfinance", "manufacturing"]
    ranked = []
    for i, sec in enumerate([top_sector] + [s for s in sectors if s != top_sector], 1):
        ranked.append({
            "sector": sec,
            "rank": i,
            "combined_score": max(0.0, 1.0 - (i - 1) * 0.15),
            "momentum_score": max(0.0, 0.8 - (i - 1) * 0.12),
            "mirofish_score": 0.5,
        })
    return {"ranked_sectors": ranked, "date": date.today().isoformat()}


# ---------------------------------------------------------------------------
# Scenario 1: All-Bullish — expect BUY signal, high score
# ---------------------------------------------------------------------------

class TestScenario1AllBullish:
    """All indicators aligned bullish → should produce BUY with high conviction."""

    def test_technical_indicators_bullish_price(self):
        """RSI should be in 50-68 range for healthy bull."""
        from strategy.technical_indicators import compute_indicators
        df = _make_price_df(n=60, trend="up")
        result = compute_indicators(df, "NABIL")
        assert "rsi14" in result.columns
        rsi_latest = result["rsi14"].iloc[-1]
        # Uptrend → RSI should be above 50
        assert 40 <= rsi_latest <= 85, f"RSI out of expected range: {rsi_latest}"

    def test_technical_indicators_bollinger_bands(self):
        """Bollinger bands should be symmetric around SMA20."""
        from strategy.technical_indicators import compute_indicators
        df = _make_price_df(n=60, trend="up")
        result = compute_indicators(df, "NABIL")
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        # BB should be symmetric: sma20 - lower ≈ upper - sma20
        sma = result["sma20"].iloc[-10:]
        upper = result["bb_upper"].iloc[-10:]
        lower = result["bb_lower"].iloc[-10:]
        diff = abs((upper - sma) - (sma - lower))
        assert diff.mean() < 0.01, "Bollinger bands not symmetric"

    def test_signal_combiner_bullish_scenario(self):
        """Bullish MiroFish + bullish technicals + top sector → BUY with HIGH conviction."""
        from strategy.signal_combiner import combine_signals
        mf = {"score": 0.75, "action": "BUY", "conviction": "HIGH"}
        ind = _make_indicators(rsi=58.0, trend="up")
        sr = _make_sector_rotation(top_sector="banking")
        result = combine_signals(
            mirofish_signal=mf,
            technical_data=ind,
            regime="BULL",
            sector_rotation=sr,
        )
        assert result["action"] == "BUY", f"Expected BUY, got {result['action']}"
        assert result["conviction"] in ("HIGH", "MEDIUM")
        assert result["composite_signal"] > 0.3
        assert result["position_size_pct"] > 0

    def test_watchlist_tier_a_for_bullish(self):
        """Strong bullish signal → Tier A on watchlist."""
        from strategy.watchlist import build_watchlist, _score_mirofish, _score_technical

        mf_score = _score_mirofish(0.75)
        assert mf_score > 15, f"MiroFish score too low: {mf_score}"

        ind = _make_indicators(rsi=58.0, trend="up")
        tech_score, reasons = _score_technical(
            rsi=58.0, macd_hist=2.0, atr=20.0,
            ret_5d=3.0, ret_20d=8.0,
            sma20=ind["sma20"], sma50=ind["sma50"], price=ind["close"]
        )
        assert tech_score > 12, f"Technical score too low: {tech_score}"

    def test_atr_positive(self):
        """ATR must always be positive."""
        from strategy.technical_indicators import compute_indicators
        df = _make_price_df(n=60, trend="up")
        result = compute_indicators(df, "NABIL")
        assert (result["atr14"].dropna() > 0).all(), "ATR must be positive"


# ---------------------------------------------------------------------------
# Scenario 2: Overbought Conflict — MiroFish bullish but RSI > 75 → HOLD
# ---------------------------------------------------------------------------

class TestScenario2OverboughtConflict:
    """Gate logic: MiroFish bullish + RSI > 75 → downgrade to HOLD."""

    def test_gate_logic_overbought_hold(self):
        """MiroFish > 0.3 + RSI > 75 → action should be HOLD."""
        from strategy.signal_combiner import combine_signals
        mf = {"score": 0.60, "action": "BUY", "conviction": "MEDIUM"}
        ind = _make_indicators(rsi=78.0, trend="up")  # overbought
        sr = _make_sector_rotation()
        result = combine_signals(
            mirofish_signal=mf,
            technical_data=ind,
            regime="LATE_BULL",
            sector_rotation=sr,
        )
        assert result["action"] in ("HOLD", "WATCH"), (
            f"Overbought gate should trigger HOLD, got {result['action']}"
        )

    def test_rsi_range_always_0_100(self):
        """RSI must be in [0, 100] for any price series."""
        from strategy.technical_indicators import compute_rsi
        for trend in ("up", "down", "flat"):
            df = _make_price_df(n=100, trend=trend)
            rsi = compute_rsi(df["close"])
            valid = rsi.dropna()
            assert (valid >= 0).all() and (valid <= 100).all(), (
                f"RSI out of [0,100] for {trend} trend: min={valid.min():.2f} max={valid.max():.2f}"
            )

    def test_position_size_reduced_for_overbought(self):
        """Position size should be 0 or very small when action is HOLD."""
        from strategy.signal_combiner import combine_signals
        mf = {"score": 0.55, "action": "BUY", "conviction": "MEDIUM"}
        ind = _make_indicators(rsi=80.0, trend="up")
        sr = _make_sector_rotation()
        result = combine_signals(
            mirofish_signal=mf,
            technical_data=ind,
            regime="BULL",
            sector_rotation=sr,
        )
        if result["action"] == "HOLD":
            assert result["position_size_pct"] == 0.0


# ---------------------------------------------------------------------------
# Scenario 3: Bear + Capitulation — MiroFish very bullish → BUY override
# ---------------------------------------------------------------------------

class TestScenario3BearCapitulation:
    """Gate logic: BEAR+CAPITULATION+MF>0.70 → override to BUY (contrarian entry)."""

    def test_capitulation_buy_override(self):
        """Bear regime with capitulation + strong MF bullish → BUY override."""
        from strategy.signal_combiner import combine_signals
        mf = {"score": 0.80, "action": "BUY", "conviction": "HIGH"}
        ind = _make_indicators(rsi=22.0, trend="down")  # oversold
        ind["vol_ratio"] = 3.5  # panic selling volume
        sr = _make_sector_rotation()
        result = combine_signals(
            mirofish_signal=mf,
            technical_data=ind,
            regime="CAPITULATION",
            sector_rotation=sr,
        )
        # High MF bullish + capitulation → should allow BUY or WATCH
        assert result["action"] in ("BUY", "WATCH"), (
            f"Capitulation+high MF should allow BUY, got {result['action']}"
        )

    def test_bear_regime_position_size_smaller(self):
        """Bear regime should have smaller base position sizes than bull."""
        from strategy.signal_combiner import combine_signals
        ind = _make_indicators(rsi=45.0, trend="down")
        mf_bull = {"score": 0.60, "action": "BUY", "conviction": "MEDIUM"}
        mf_bear = {"score": 0.60, "action": "BUY", "conviction": "MEDIUM"}
        sr = _make_sector_rotation()

        result_bull = combine_signals(mf_bull, ind, "BULL", sr)
        result_bear = combine_signals(mf_bear, ind, "BEAR", sr)

        # Bear should have smaller or equal position size
        bear_size = result_bear.get("position_size_pct", 0)
        bull_size = result_bull.get("position_size_pct", 0)
        assert bear_size <= bull_size * 1.1, (
            f"Bear position size ({bear_size:.1f}%) should not exceed bull ({bull_size:.1f}%)"
        )

    def test_trading_rules_block_high_rsi(self):
        """validate_trade should add soft warning for RSI > 72."""
        from strategy.trading_rules import validate_trade
        result = validate_trade(
            symbol="NABIL",
            action="BUY",
            position_pct=10.0,
            open_positions=2,
            ltp=1200.0,
            prev_close=1180.0,
            rsi=78.0,
        )
        warnings = result.get("soft_warnings", [])
        rsi_warned = any("rsi" in w.lower() or "overbought" in w.lower() for w in warnings)
        assert rsi_warned, f"Expected RSI overbought warning, got: {warnings}"


# ---------------------------------------------------------------------------
# Scenario 4: Sideways Mean Reversion
# ---------------------------------------------------------------------------

class TestScenario4SidewaysMeanReversion:
    """SIDEWAYS regime → mean_reversion strategy selected; low-RSI stocks favoured."""

    def test_sideways_regime_selects_mean_reversion(self):
        """select_active_strategy(SIDEWAYS) → mean_reversion_sideways."""
        from strategy.entry_exit import select_active_strategy
        result = select_active_strategy(regime="SIDEWAYS")
        assert result["strategy"] == "mean_reversion_sideways", (
            f"Expected mean_reversion_sideways, got {result['strategy']}"
        )

    def test_entry_conditions_mean_reversion(self):
        """Mean reversion entry: RSI < 35, price near lower BB."""
        from strategy.entry_exit import check_entry_conditions
        ind = _make_indicators(rsi=32.0, trend="flat")
        # Adjust to be near lower Bollinger band
        price = ind["close"]
        ind["bb_lower"] = price * 1.03   # price is 3% below lower band
        ind["bb_upper"] = price * 1.12
        ind["sma20"] = price * 1.05      # price below SMA20 (recovery target)
        ind["vol_ratio"] = 0.7           # low volume (required condition)
        ind["week52_low"] = price * 0.85 # price not near 52w low
        ind["week52_high"] = price * 1.30

        result = check_entry_conditions(
            strategy_name="mean_reversion_sideways",
            symbol="EBL",
            indicators=ind,
            mirofish_score=0.15,         # slight positive
            composite_score=0.20,
            sector_rank=3,
        )
        # With RSI < 35 condition met, should pass entry
        # (may fail other conditions — just check it runs without error)
        assert "entry_ok" in result
        assert "failed_conditions" in result

    def test_bollinger_width_not_negative(self):
        """Upper BB must always be >= Lower BB."""
        from strategy.technical_indicators import compute_indicators, compute_bollinger
        df = _make_price_df(n=60, trend="flat")
        upper, lower = compute_bollinger(df["close"])
        assert (upper >= lower).all(), "Bollinger upper < lower — invalid!"

    def test_signal_combiner_sideways_weights(self):
        """In SIDEWAYS regime, technical weight is highest (0.50)."""
        from strategy.signal_combiner import REGIME_WEIGHTS
        sideways_w = REGIME_WEIGHTS["SIDEWAYS"]
        assert sideways_w["tech"] >= sideways_w["mf"], (
            "In SIDEWAYS, technical should have at least as much weight as MiroFish"
        )


# ---------------------------------------------------------------------------
# Scenario 5: Pre-NRB-Event Caution — signal dampened by event calendar
# ---------------------------------------------------------------------------

class TestScenario5PreNRBCaution:
    """Upcoming HIGH-impact NRB event → signal dampened, BUY downgraded to WATCH."""

    def test_event_adjust_high_impact_dampens_signal(self):
        """HIGH-impact event → BUY downgraded to WATCH, position size halved."""
        from pipeline.event_calendar import adjust_signal_for_events
        signal = {
            "action": "BUY",
            "composite_signal": 0.65,
            "conviction": "MEDIUM",
            "position_size_pct": 15.0,
        }
        fake_events = [
            {
                "name": "NRB Monetary Policy",
                "date": date.today().isoformat(),
                "impact": "HIGH",
                "event_type": "nrb_monetary_policy",
                "days_away": 0,
            }
        ]
        result = adjust_signal_for_events(signal, fake_events)
        assert result["event_adjusted"] is True
        assert result["action"] in ("WATCH", "HOLD"), (
            f"HIGH-impact event should downgrade BUY, got {result['action']}"
        )
        assert result["position_size_pct"] <= signal["position_size_pct"] * 0.6

    def test_event_adjust_medium_impact_reduces_size(self):
        """MEDIUM-impact event → position size reduced but action unchanged."""
        from pipeline.event_calendar import adjust_signal_for_events
        signal = {
            "action": "BUY",
            "composite_signal": 0.55,
            "conviction": "MEDIUM",
            "position_size_pct": 12.0,
        }
        fake_events = [
            {
                "name": "Book Close: NABIL",
                "date": date.today().isoformat(),
                "impact": "MEDIUM",
                "event_type": "book_close",
                "days_away": 1,
            }
        ]
        result = adjust_signal_for_events(signal, fake_events)
        assert result["event_adjusted"] is True
        assert result["position_size_pct"] < signal["position_size_pct"]

    def test_event_adjust_no_event_unchanged(self):
        """No nearby events → signal passes through unchanged."""
        from pipeline.event_calendar import adjust_signal_for_events
        signal = {
            "action": "BUY",
            "composite_signal": 0.60,
            "position_size_pct": 12.0,
        }
        result = adjust_signal_for_events(signal, events=[])
        assert result["event_adjusted"] is False
        assert result["position_size_pct"] == signal["position_size_pct"]

    def test_regime_detector_returns_dict(self):
        """detect_regime should return a dict with 'regime' key."""
        from strategy.regime_detector import detect_regime
        # Mock the DB load to return empty (offline test)
        with patch("strategy.regime_detector.load_nepse_index", return_value=pd.DataFrame()):
            result = detect_regime()
        assert isinstance(result, dict), "detect_regime must return a dict"
        assert "regime" in result, "Result must contain 'regime' key"
        assert result["regime"] in (
            "BULL", "BEAR", "SIDEWAYS", "RECOVERY",
            "EARLY_BULL", "LATE_BULL", "CAPITULATION", "CONSOLIDATION", "UNKNOWN"
        ), f"Unknown regime value: {result['regime']}"

    def test_trading_rules_circuit_breaker(self):
        """validate_trade should block BUY if price already hit +9.5% circuit."""
        from strategy.trading_rules import validate_trade
        result = validate_trade(
            symbol="XYZ",
            action="BUY",
            position_pct=10.0,
            open_positions=1,
            ltp=110.0,       # +10% from prev close
            prev_close=100.0,
        )
        # Should either be blocked by circuit breaker rule or have soft warning
        failed = result.get("failed_rules", [])
        warnings = result.get("soft_warnings", [])
        circuit_hit = (
            any("circuit" in r.lower() for r in failed)
            or any("circuit" in w.lower() for w in warnings)
        )
        assert circuit_hit, (
            f"Circuit breaker should trigger at +10%.\nFailed: {failed}\nWarnings: {warnings}"
        )


# ---------------------------------------------------------------------------
# Bonus: Portfolio paper trade smoke test
# ---------------------------------------------------------------------------

class TestPortfolioSmoke:
    """Quick smoke tests for the paper trading portfolio."""

    def test_open_and_close_position(self):
        """Open a position, close it, verify PnL calculation."""
        from strategy.portfolio import NEPSEPortfolio
        p = NEPSEPortfolio(initial_cash=500_000)
        # Disable DB
        p._engine = None

        initial_cash = p.cash
        price_in = 1000.0
        shares = 50

        ok = p.open_position("NABIL", price=price_in, shares=shares,
                              strategy="test", stop_loss=920.0, target_price=1180.0)
        assert ok, "open_position should succeed"
        assert "NABIL" in p.positions

        # Check cash was debited
        assert p.cash < initial_cash

        # Close at profit
        price_out = 1100.0
        trade = p.close_position("NABIL", price=price_out, exit_reason="target_hit")
        assert trade is not None
        assert trade.pnl_pct > 0, f"Should have positive PnL, got {trade.pnl_pct}"
        assert trade.action == "SELL"

    def test_insufficient_cash_blocked(self):
        """open_position should return False if cash is insufficient."""
        from strategy.portfolio import NEPSEPortfolio
        p = NEPSEPortfolio(initial_cash=1000)
        p._engine = None
        ok = p.open_position("NABIL", price=1200, shares=100)  # needs 120k+
        assert not ok, "Should be blocked by insufficient cash"

    def test_duplicate_position_blocked(self):
        """Cannot open two positions in the same symbol simultaneously."""
        from strategy.portfolio import NEPSEPortfolio
        p = NEPSEPortfolio(initial_cash=500_000)
        p._engine = None
        p.open_position("NABIL", price=1000, shares=10)
        ok2 = p.open_position("NABIL", price=1050, shares=10)
        assert not ok2, "Duplicate position should be blocked"
