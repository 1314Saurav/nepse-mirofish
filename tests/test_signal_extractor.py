"""
tests/test_signal_extractor.py

Unit tests for pipeline/signal_extractor.py

Tests 3 mock simulation outputs: bullish, bearish, neutral.
Run with:  python -m pytest tests/test_signal_extractor.py -v
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.signal_extractor import extract_trading_signal, format_signal_table


# ── Mock simulation data factories ────────────────────────────────────────────

def _make_action(content: str, platform: str = "twitter",
                 round_num: int = 1, agent_id: str = "Broker_001") -> dict:
    return {
        "agent_id":  agent_id,
        "content":   content,
        "text":      content,
        "platform":  platform,
        "round_num": round_num,
    }


def _make_sim(actions: list[dict], date: str = "2026-03-26",
              sim_id: str = "test-sim-001") -> dict:
    """Wrap actions into a minimal simulation output dict."""
    # Build synthetic timeline from actions
    rounds = sorted(set(a.get("round_num", 0) for a in actions))
    timeline = [{"round": r, "action_count": sum(1 for a in actions if a.get("round_num") == r)}
                for r in rounds]
    return {
        "simulation_id": sim_id,
        "date":          date,
        "actions":       actions,
        "timeline":      timeline,
        "agent_stats":   {},
        "sim_config":    {},
    }


# ── Bullish mock ───────────────────────────────────────────────────────────────

BULLISH_ACTIONS = [
    # Round 1 — strong bullish retail
    _make_action("NEPSE rally today! Huge gains across banking sector. Buy accumulate now!", "twitter", 1, "RetailTrader_001"),
    _make_action("NRB repo rate cut driving positive sentiment. Banking stocks surge. Strong growth signal.", "reddit", 1, "PolicyWatcher_001"),
    _make_action("Hydropower sector positive. NEA PPA rate increase. UPPER NHPC BPCL outperform NEPSE. Rally.", "twitter", 1, "HydroAnalyst_001"),
    _make_action("IPO oversubscribed by 200x. Bullish retail sentiment high. Growth opportunity in microfinance.", "reddit", 1, "RetailTrader_002"),
    _make_action("Good EPS earnings beat. PE ratio undervalued. Institutional accumulate banking stocks. Rise.", "reddit", 1, "Broker_001"),

    # Round 2 — continuation
    _make_action("NEPSE higher again. Banking outperform. NRB positive on credit growth. Strong buy signal.", "twitter", 2, "RetailTrader_003"),
    _make_action("Monsoon forecast good. Hydropower generation surge expected. UPPER target rise.", "reddit", 2, "HydroAnalyst_002"),
    _make_action("Dividend declared positive. Remittance high. USD strong. Growth opportunity confirmed.", "twitter", 2, "NRN_001"),
    _make_action("Coalition stable government positive. Budget infrastructure announcement. Rise rally.", "reddit", 2, "PoliAnalyst_001"),
    _make_action("Circuit upper hit. Accumulate on pullbacks. Positive trend. Bullish momentum.", "twitter", 2, "Broker_002"),

    # Round 3 — sustained
    _make_action("SEBON approval bonus share. Strong rally continues. Buy opportunity microfinance positive.", "reddit", 3, "RetailTrader_004"),
    _make_action("NRN investment inflow high. Remittance peak. Dividend yield attractive. Growth signal.", "twitter", 3, "NRN_002"),
    _make_action("Earnings beat quarterly results. EPS growth strong. Banking positive. Rise higher.", "reddit", 3, "Broker_003"),
    _make_action("PPA rate increase confirmed. NEA export deal. Hydro rally strong growth.", "twitter", 3, "HydroAnalyst_003"),
    _make_action("Foreign buy signal. Oversubscribed IPO. Positive momentum accumulate gains.", "reddit", 3, "Broker_004"),
]


# ── Bearish mock ───────────────────────────────────────────────────────────────

BEARISH_ACTIONS = [
    # Round 1 — strong bearish signals
    _make_action("NEPSE crash incoming. CD ratio too high. NRB will tighten credit. Sell dump now.", "twitter", 1, "PolicyWatcher_002"),
    _make_action("Banking stocks fall. NPL high. Provision increase hits earnings. EPS decline bearish.", "reddit", 1, "Broker_005"),
    _make_action("Coalition collapse government fall. Political crisis uncertainty. Risk-off. Sell.", "twitter", 1, "PoliAnalyst_002"),
    _make_action("Dry season power shortage load shedding. Hydro generation drop. NHPC fall.", "reddit", 1, "HydroAnalyst_004"),
    _make_action("Inflation high NPR depreciate. Credit tight. Market decline risk overvalued expensive.", "twitter", 1, "RetailTrader_005"),

    # Round 2 — panic selling
    _make_action("Margin call forced sale panic selling. Portfolio down. Circuit lower hit. Dump.", "reddit", 2, "RetailTrader_006"),
    _make_action("NRB tighten CD ratio concern bank rate hike. Banking sector weak. Fall.", "twitter", 2, "PolicyWatcher_003"),
    _make_action("Dry season load shedding increase. Hydro output negative. UPPER fall decline.", "reddit", 2, "HydroAnalyst_005"),
    _make_action("Government fall no coalition. Budget delay. Infrastructure concern. Risk sell.", "twitter", 2, "PoliAnalyst_003"),
    _make_action("Bad earnings quarterly results. EPS decline NPL provisioning. Overvalued sell.", "reddit", 2, "Broker_006"),

    # Round 3 — continuing decline
    _make_action("SEBON warning. Margin lending concern. Circuit lower again. Market weak bearish.", "twitter", 3, "Broker_007"),
    _make_action("Foreign sell pressure. Gulf economy weak remittance drop. NRN reduce exposure.", "reddit", 3, "NRN_003"),
    _make_action("Repo rate hike fear. CD ratio breach warning. Banking stocks crash. Bear market.", "twitter", 3, "PolicyWatcher_004"),
    _make_action("Political crisis no resolution. Election uncertainty negative. Risk-off dump.", "reddit", 3, "PoliAnalyst_004"),
    _make_action("Overvalued expensive PE ratio high. Earnings miss. Sell pressure weak market.", "twitter", 3, "Broker_008"),
]


# ── Neutral mock ───────────────────────────────────────────────────────────────

NEUTRAL_ACTIONS = [
    # Mixed signals across rounds
    _make_action("NEPSE stable today. Banking sideways. Some positive sentiment but concerns remain.", "twitter", 1, "Broker_009"),
    _make_action("Mixed signals. Gainers and losers balanced. Market looking for direction.", "reddit", 1, "RetailTrader_007"),
    _make_action("NRB repo rate unchanged. CD ratio stable. Banking holding. Watch for catalyst.", "reddit", 1, "PolicyWatcher_005"),
    _make_action("Hydro sector sideways. Waiting for monsoon data. PPA rate review pending.", "twitter", 1, "HydroAnalyst_006"),
    _make_action("IPO applications average. Retail sentiment neutral. No major catalyst today.", "reddit", 1, "RetailTrader_008"),

    # Round 2 — continued neutral
    _make_action("Market digesting recent moves. Some buying interest offset by selling pressure.", "twitter", 2, "Broker_010"),
    _make_action("Coalition holding but fragile. Political risk moderate. Budget timeline uncertain.", "reddit", 2, "PoliAnalyst_005"),
    _make_action("Remittance stable. USD neutral. NRN holding positions. No major change.", "twitter", 2, "NRN_004"),
    _make_action("Earnings in line with expectations. EPS stable. PE ratio fair value zone.", "reddit", 2, "Broker_011"),
    _make_action("Banking sector consolidating. CD ratio comfortable. Wait for data release.", "twitter", 2, "PolicyWatcher_006"),

    # Round 3 — slight positive tilt
    _make_action("Slight positive tilt. Some accumulation at support. Watching for breakout.", "reddit", 3, "Broker_012"),
    _make_action("Monsoon outlook normal. Hydro sector neutral to slightly positive outlook.", "twitter", 3, "HydroAnalyst_007"),
    _make_action("Mixed news day. Some positive on dividends but political uncertainty caps gains.", "reddit", 3, "PoliAnalyst_006"),
    _make_action("Market neutral. Retail indecisive. Waiting for clear signal from NRB or SEBON.", "twitter", 3, "RetailTrader_009"),
    _make_action("Fair value territory. No strong buy or sell signal. Hold current positions.", "reddit", 3, "Broker_013"),
]


# ── Test cases ─────────────────────────────────────────────────────────────────

class TestBullishScenario:
    """Bullish simulation: strong buy signals across all platforms."""

    @pytest.fixture
    def bullish_signal(self):
        sim = _make_sim(BULLISH_ACTIONS, date="2026-03-26", sim_id="bull-test")
        return extract_trading_signal(sim)

    def test_direction_is_bullish(self, bullish_signal):
        assert bullish_signal["direction"] == "BULLISH", \
            f"Expected BULLISH, got {bullish_signal['direction']}"

    def test_bull_bear_score_positive(self, bullish_signal):
        assert bullish_signal["bull_bear_score"] > 0, \
            f"Expected positive score, got {bullish_signal['bull_bear_score']}"

    def test_bull_bear_score_not_extreme(self, bullish_signal):
        # Should not be maxed out (groupthink check)
        assert bullish_signal["bull_bear_score"] < 1.0

    def test_confidence_reasonable(self, bullish_signal):
        assert 30.0 <= bullish_signal["confidence_pct"] <= 100.0

    def test_has_sector_signals(self, bullish_signal):
        assert len(bullish_signal["sector_signals"]) >= 2

    def test_hydropower_bullish(self, bullish_signal):
        hydro_score = bullish_signal["sector_signals"].get("Hydropower", 0)
        assert hydro_score >= 0.0, f"Expected non-negative hydropower, got {hydro_score}"

    def test_has_key_themes(self, bullish_signal):
        assert len(bullish_signal["key_themes"]) >= 1

    def test_has_top_drivers(self, bullish_signal):
        assert len(bullish_signal["top_driver_agent_types"]) >= 1

    def test_has_round_scores(self, bullish_signal):
        assert len(bullish_signal["raw_round_scores"]) >= 1

    def test_signal_fields_present(self, bullish_signal):
        required = [
            "date", "bull_bear_score", "confidence_pct", "direction",
            "sentiment_velocity", "platform_agreement", "sector_signals",
            "top_driver_agent_types", "key_themes", "raw_round_scores",
            "signal_strength", "total_actions",
        ]
        for field in required:
            assert field in bullish_signal, f"Missing field: {field}"

    def test_total_actions_correct(self, bullish_signal):
        assert bullish_signal["total_actions"] == len(BULLISH_ACTIONS)

    def test_format_table_runs(self, bullish_signal):
        table = format_signal_table(bullish_signal)
        assert "BULLISH" in table
        assert "NEPSE" in table


class TestBearishScenario:
    """Bearish simulation: strong sell signals, political/NRB concerns."""

    @pytest.fixture
    def bearish_signal(self):
        sim = _make_sim(BEARISH_ACTIONS, date="2026-03-26", sim_id="bear-test")
        return extract_trading_signal(sim)

    def test_direction_is_bearish(self, bearish_signal):
        assert bearish_signal["direction"] == "BEARISH", \
            f"Expected BEARISH, got {bearish_signal['direction']}"

    def test_bull_bear_score_negative(self, bearish_signal):
        assert bearish_signal["bull_bear_score"] < 0, \
            f"Expected negative score, got {bearish_signal['bull_bear_score']}"

    def test_score_not_extreme(self, bearish_signal):
        assert bearish_signal["bull_bear_score"] > -1.0

    def test_confidence_reasonable(self, bearish_signal):
        assert 30.0 <= bearish_signal["confidence_pct"] <= 100.0

    def test_banking_sector_bearish_or_neutral(self, bearish_signal):
        bank_score = bearish_signal["sector_signals"].get("Commercial Banks", 0)
        assert bank_score <= 0.0, f"Expected non-positive banks, got {bank_score}"

    def test_has_driver_types(self, bearish_signal):
        assert len(bearish_signal["top_driver_agent_types"]) >= 1

    def test_fields_complete(self, bearish_signal):
        assert bearish_signal["signal_strength"] in ("STRONG", "MODERATE", "WEAK")
        assert bearish_signal["sentiment_velocity"] in ("ACCELERATING", "DECELERATING", "STABLE")

    def test_format_table_shows_bearish(self, bearish_signal):
        table = format_signal_table(bearish_signal)
        assert "BEARISH" in table


class TestNeutralScenario:
    """Neutral simulation: mixed signals, no clear direction."""

    @pytest.fixture
    def neutral_signal(self):
        sim = _make_sim(NEUTRAL_ACTIONS, date="2026-03-26", sim_id="neutral-test")
        return extract_trading_signal(sim)

    def test_direction_is_neutral_or_mild(self, neutral_signal):
        # Neutral actions should not produce a strong directional signal
        abs_score = abs(neutral_signal["bull_bear_score"])
        assert abs_score < 0.5, \
            f"Expected abs score < 0.5 for neutral, got {abs_score}"

    def test_signal_is_not_strong(self, neutral_signal):
        # Neutral scenario should not produce STRONG signal
        assert neutral_signal["signal_strength"] in ("WEAK", "MODERATE"), \
            f"Expected WEAK or MODERATE for neutral, got {neutral_signal['signal_strength']}"

    def test_confidence_reasonable(self, neutral_signal):
        assert 10.0 <= neutral_signal["confidence_pct"] <= 100.0

    def test_has_round_scores(self, neutral_signal):
        assert len(neutral_signal["raw_round_scores"]) >= 1

    def test_format_table_runs(self, neutral_signal):
        table = format_signal_table(neutral_signal)
        assert "NEPSE" in table
        assert "2026-03-26" in table


class TestEdgeCases:
    """Edge cases: empty actions, single action, platform-only signals."""

    def test_empty_actions(self):
        sim = _make_sim([], date="2026-01-01")
        sig = extract_trading_signal(sim)
        assert sig["direction"] == "NEUTRAL"
        assert sig["bull_bear_score"] == 0.0
        assert sig["total_actions"] == 0

    def test_single_bullish_action(self):
        sim = _make_sim([_make_action("Strong buy rally surge gains up!", "twitter", 1)])
        sig = extract_trading_signal(sim)
        assert sig["direction"] in ("BULLISH", "NEUTRAL")
        assert sig["total_actions"] == 1

    def test_single_bearish_action(self):
        sim = _make_sim([_make_action("Sell crash drop decline bearish weak down!", "reddit", 1)])
        sig = extract_trading_signal(sim)
        assert sig["direction"] in ("BEARISH", "NEUTRAL")

    def test_date_preserved(self):
        sim = _make_sim([], date="2025-12-31")
        sig = extract_trading_signal(sim)
        assert sig["date"] == "2025-12-31"

    def test_output_schema_complete(self):
        sim = _make_sim(BULLISH_ACTIONS[:5])
        sig = extract_trading_signal(sim)
        schema_keys = [
            "date", "simulation_id", "extracted_at",
            "bull_bear_score", "confidence_pct", "direction",
            "sentiment_velocity", "platform_agreement",
            "twitter_score", "reddit_score",
            "sector_signals", "top_driver_agent_types",
            "key_themes", "raw_round_scores",
            "signal_strength", "total_actions", "quality_flags",
        ]
        for key in schema_keys:
            assert key in sig, f"Missing key in signal output: '{key}'"

    def test_quality_flags_is_list(self):
        sim = _make_sim(BULLISH_ACTIONS)
        sig = extract_trading_signal(sim)
        assert isinstance(sig["quality_flags"], list)

    def test_sector_signals_is_dict(self):
        sim = _make_sim(BEARISH_ACTIONS)
        sig = extract_trading_signal(sim)
        assert isinstance(sig["sector_signals"], dict)
        for k, v in sig["sector_signals"].items():
            assert isinstance(k, str)
            assert isinstance(v, float), f"Sector score for {k} is not float: {v}"


# ── Quick smoke-test run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    BOLD  = "\033[1m"; GREEN = "\033[92m"; RED = "\033[91m"
    CYAN  = "\033[96m"; RESET = "\033[0m"

    scenarios = [
        ("BULLISH",  BULLISH_ACTIONS,  "bull-smoke"),
        ("BEARISH",  BEARISH_ACTIONS,  "bear-smoke"),
        ("NEUTRAL",  NEUTRAL_ACTIONS,  "neutral-smoke"),
    ]

    print(f"\n{BOLD}Signal Extractor Smoke Test{RESET}")
    print("=" * 60)

    all_ok = True
    for label, actions, sim_id in scenarios:
        sim = _make_sim(actions, date="2026-03-26", sim_id=sim_id)
        sig = extract_trading_signal(sim)
        ok  = (
            label == "BULLISH" and sig["direction"] == "BULLISH" or
            label == "BEARISH" and sig["direction"] == "BEARISH" or
            label == "NEUTRAL" and abs(sig["bull_bear_score"]) < 0.5
        )
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        all_ok = all_ok and ok
        print(f"  [{status}] {label:<10} score={sig['bull_bear_score']:+.3f}  "
              f"dir={sig['direction']:<8}  conf={sig['confidence_pct']:.0f}%  "
              f"strength={sig['signal_strength']}")

    print()
    print(format_signal_table(extract_trading_signal(
        _make_sim(BULLISH_ACTIONS, date="2026-03-26", sim_id="demo")
    )))

    if all_ok:
        print(f"\n{GREEN}All smoke tests passed.{RESET}")
    else:
        print(f"\n{RED}Some smoke tests failed — check signal_extractor.py{RESET}")
        sys.exit(1)
