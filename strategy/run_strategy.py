"""
strategy/run_strategy.py
Full daily NEPSE strategy cycle — 15 steps.
Orchestrates: regime detection → technical analysis → sector rotation →
MiroFish signals → signal combining → rule validation → watchlist →
portfolio monitoring → event filtering → dashboard refresh → Telegram.

Run:
  python -m strategy.run_strategy
  make strategy
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe of symbols to analyse (NEPSE blue chips + sector leaders)
# ---------------------------------------------------------------------------

ANALYSIS_UNIVERSE = [
    # Banking
    "NABIL", "NICA", "EBL", "SANIMA", "PRVU",
    "KBL", "MBL", "SBI", "CBL", "HBL",
    # Hydropower
    "NHPC", "CHCL", "UPPER", "BPCL", "RRHP",
    # Insurance
    "NLIC", "LICN", "SICL", "PIC", "SGIC",
    # Finance
    "ICFC", "GMFIL", "MFIL",
    # Microfinance
    "SKBBL", "CBBL",
    # Manufacturing
    "UNL", "SHIVM", "BNL",
    # Hotels
    "SHL", "OHL",
]


# ---------------------------------------------------------------------------
# Step result tracking
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step: int
    name: str
    status: str          # OK / WARN / ERROR / SKIP
    duration_s: float = 0.0
    output: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class CycleResult:
    run_date: str
    steps: list[StepResult] = field(default_factory=list)
    regime: str = "UNKNOWN"
    watchlist_count: int = 0
    signals_generated: int = 0
    trades_executed: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def step01_check_market_open() -> StepResult:
    """Step 1: Verify NEPSE is open today (not a holiday/weekend)."""
    today = date.today()
    # NEPSE: Sunday–Thursday; closed Friday, Saturday + public holidays
    is_weekday = today.weekday() not in (4, 5)  # 4=Fri, 5=Sat

    status = "OK" if is_weekday else "WARN"
    return StepResult(
        step=1, name="Check market open",
        status=status,
        output={"date": today.isoformat(), "weekday": today.strftime("%A"),
                "market_open": is_weekday},
        error="" if is_weekday else "Market closed today (weekend/holiday)",
    )


def step02_load_price_data(symbols: list[str]) -> StepResult:
    """Step 2: Load/validate price data from DB for all symbols."""
    t0 = time.time()
    loaded = []
    failed = []
    try:
        from strategy.technical_indicators import load_price_history
        for sym in symbols:
            try:
                df = load_price_history(sym, days=60)
                if len(df) >= 20:
                    loaded.append(sym)
                else:
                    failed.append(sym)
            except Exception:
                failed.append(sym)
    except ImportError as exc:
        return StepResult(step=2, name="Load price data", status="ERROR",
                          error=str(exc))

    status = "OK" if loaded else "ERROR"
    if failed:
        status = "WARN"
    return StepResult(
        step=2, name="Load price data", status=status,
        duration_s=time.time() - t0,
        output={"loaded": len(loaded), "failed": len(failed), "failed_symbols": failed[:5]},
    )


def step03_compute_indicators(symbols: list[str]) -> tuple[StepResult, dict]:
    """Step 3: Compute technical indicators for all symbols."""
    t0 = time.time()
    indicators: dict = {}
    errors: list[str] = []

    try:
        from strategy.technical_indicators import get_latest_indicators
    except ImportError as exc:
        return StepResult(step=3, name="Compute indicators", status="ERROR",
                          error=str(exc)), {}

    for sym in symbols:
        try:
            ind = get_latest_indicators(sym)
            if ind:
                indicators[sym] = ind
        except Exception as exc:
            errors.append(f"{sym}: {exc}")

    status = "OK" if indicators else "ERROR"
    return StepResult(
        step=3, name="Compute indicators", status=status,
        duration_s=time.time() - t0,
        output={"computed": len(indicators), "errors": len(errors)},
        error="; ".join(errors[:3]),
    ), indicators


def step04_detect_regime() -> tuple[StepResult, dict]:
    """Step 4: Detect current market regime."""
    t0 = time.time()
    try:
        from strategy.regime_detector import detect_regime
        result = detect_regime()
        if not isinstance(result, dict):
            result = {"regime": str(result), "confidence": 0.5, "sub_regime": None}
        return StepResult(
            step=4, name="Detect regime", status="OK",
            duration_s=time.time() - t0,
            output=result,
        ), result
    except Exception as exc:
        return StepResult(step=4, name="Detect regime", status="ERROR",
                          error=str(exc)), {"regime": "SIDEWAYS", "confidence": 0.5}


def step05_sector_rotation() -> tuple[StepResult, dict]:
    """Step 5: Compute sector rotation scores."""
    t0 = time.time()
    try:
        from strategy.sector_rotation import get_rotation_signal
        result = get_rotation_signal(save=True)
        n_sectors = len(result.get("ranked_sectors", []))
        return StepResult(
            step=5, name="Sector rotation", status="OK",
            duration_s=time.time() - t0,
            output={"sectors_ranked": n_sectors},
        ), result
    except Exception as exc:
        return StepResult(step=5, name="Sector rotation", status="WARN",
                          error=str(exc)), {}


def step06_fetch_mirofish_signals(symbols: list[str], regime: str) -> tuple[StepResult, dict]:
    """Step 6: Pull latest MiroFish signals from the bridge/API."""
    t0 = time.time()
    mirofish_signals: dict = {}
    try:
        from pipeline.mirofish_bridge import MiroFishBridge
        bridge = MiroFishBridge()
        # Get latest simulation results for each symbol
        for sym in symbols[:10]:  # limit API calls
            try:
                result = bridge.get_symbol_signal(sym)
                if result:
                    mirofish_signals[sym] = result
            except Exception:
                # Fallback: neutral signal
                mirofish_signals[sym] = {"score": 0.0, "action": "HOLD", "conviction": "LOW"}
    except Exception as exc:
        logger.warning("MiroFish bridge unavailable: %s", exc)
        # Neutral fallback for all symbols
        for sym in symbols:
            mirofish_signals[sym] = {"score": 0.0, "action": "HOLD", "conviction": "LOW"}

    return StepResult(
        step=6, name="Fetch MiroFish signals", status="OK",
        duration_s=time.time() - t0,
        output={"signals_fetched": len(mirofish_signals)},
    ), mirofish_signals


def step07_combine_signals(
    symbols: list[str],
    indicators: dict,
    mirofish_signals: dict,
    regime_data: dict,
    sector_rotation: dict,
) -> tuple[StepResult, dict]:
    """Step 7: Combine all signals into composite scores."""
    t0 = time.time()
    combined: dict = {}
    errors = []

    try:
        from strategy.signal_combiner import combine_signals
    except ImportError as exc:
        return StepResult(step=7, name="Combine signals", status="ERROR",
                          error=str(exc)), {}

    regime = regime_data.get("regime", "SIDEWAYS")

    for sym in symbols:
        ind = indicators.get(sym, {})
        mf = mirofish_signals.get(sym, {"score": 0.0})
        try:
            combined[sym] = combine_signals(
                mirofish_signal=mf,
                technical_data=ind,
                regime=regime,
                sector_rotation=sector_rotation,
            )
        except Exception as exc:
            errors.append(sym)
            combined[sym] = {"action": "HOLD", "composite_signal": 0.0,
                             "conviction": "LOW", "position_size_pct": 0.0}

    return StepResult(
        step=7, name="Combine signals", status="OK",
        duration_s=time.time() - t0,
        output={"combined": len(combined), "errors": len(errors)},
    ), combined


def step08_apply_event_filter(combined_signals: dict) -> tuple[StepResult, dict]:
    """Step 8: Dampen signals for upcoming market events."""
    t0 = time.time()
    adjusted: dict = {}
    event_count = 0

    try:
        from pipeline.event_calendar import get_events_for_date, adjust_signal_for_events
        events = get_events_for_date()
        event_count = len(events)
        for sym, sig in combined_signals.items():
            adjusted[sym] = adjust_signal_for_events(sig, events)
    except Exception as exc:
        logger.warning("Event filter unavailable: %s", exc)
        adjusted = combined_signals

    return StepResult(
        step=8, name="Event filter", status="OK",
        duration_s=time.time() - t0,
        output={"events_active": event_count, "signals_adjusted": len(adjusted)},
    ), adjusted


def step09_validate_trades(
    combined_signals: dict,
    indicators: dict,
    portfolio,
) -> tuple[StepResult, dict]:
    """Step 9: Apply NEPSE trading rules to each BUY signal."""
    t0 = time.time()
    validated: dict = {}
    approved = 0
    blocked = 0

    try:
        from strategy.trading_rules import validate_trade
    except ImportError as exc:
        return StepResult(step=9, name="Validate trades", status="ERROR",
                          error=str(exc)), {}

    open_positions = len(portfolio.get_open_positions()) if portfolio else 0

    for sym, sig in combined_signals.items():
        if sig.get("action") != "BUY":
            validated[sym] = {**sig, "trade_approved": False, "skip_reason": "not BUY signal"}
            continue

        ind = indicators.get(sym, {})
        try:
            val = validate_trade(
                symbol=sym,
                action="BUY",
                position_pct=sig.get("position_size_pct", 10.0),
                open_positions=open_positions,
                ltp=ind.get("close", 100.0),
                prev_close=ind.get("close", 100.0),
                avg_turnover_npr=ind.get("avg_turnover_npr", 1_000_000),
                rsi=ind.get("rsi14", 50.0),
            )
            validated[sym] = {**sig, "trade_validation": val,
                               "trade_approved": val.get("approved", False)}
            if val.get("approved"):
                approved += 1
            else:
                blocked += 1
        except Exception as exc:
            validated[sym] = {**sig, "trade_approved": False,
                               "skip_reason": str(exc)}
            blocked += 1

    return StepResult(
        step=9, name="Validate trades", status="OK",
        duration_s=time.time() - t0,
        output={"approved": approved, "blocked": blocked},
    ), validated


def step10_select_strategy(regime_data: dict) -> tuple[StepResult, str]:
    """Step 10: Select the active trading strategy for today."""
    t0 = time.time()
    try:
        from strategy.entry_exit import select_active_strategy
        regime = regime_data.get("regime", "SIDEWAYS")
        strategy_result = select_active_strategy(regime=regime)
        strategy_name = strategy_result.get("strategy", "mean_reversion_sideways")
        return StepResult(
            step=10, name="Select strategy", status="OK",
            duration_s=time.time() - t0,
            output=strategy_result,
        ), strategy_name
    except Exception as exc:
        return StepResult(step=10, name="Select strategy", status="WARN",
                          error=str(exc)), "mean_reversion_sideways"


def step11_generate_watchlist(
    symbols: list[str],
    combined_signals: dict,
    sector_rotation: dict,
    regime: str,
) -> tuple[StepResult, list]:
    """Step 11: Build and save the daily watchlist."""
    t0 = time.time()
    try:
        from strategy.watchlist import generate_daily_watchlist
        entries = generate_daily_watchlist(
            symbols=symbols,
            mirofish_signals=combined_signals,
            sector_rotation=sector_rotation,
            regime=regime,
            top_n=10,
            send_to_telegram=True,
            save=True,
        )
        return StepResult(
            step=11, name="Generate watchlist", status="OK",
            duration_s=time.time() - t0,
            output={"entries": len(entries),
                    "tier_a": sum(1 for e in entries if e.tier == "A")},
        ), entries
    except Exception as exc:
        return StepResult(step=11, name="Generate watchlist", status="ERROR",
                          error=str(exc)), []


def step12_check_portfolio_exits(
    portfolio,
    indicators: dict,
    combined_signals: dict,
    regime: str,
) -> StepResult:
    """Step 12: Check all open positions for exit signals."""
    t0 = time.time()
    if portfolio is None:
        return StepResult(step=12, name="Check exits", status="SKIP",
                          error="Portfolio unavailable")

    current_prices = {sym: ind.get("close", 0) for sym, ind in indicators.items()}
    current_rsi = {sym: ind.get("rsi14", 50) for sym, ind in indicators.items()}
    mirofish_scores = {sym: sig.get("mirofish_score", 0) for sym, sig in combined_signals.items()}

    exits = portfolio.check_exit_conditions(
        current_prices=current_prices,
        current_rsi=current_rsi,
        mirofish_scores=mirofish_scores,
        regime=regime,
    )

    for exit_info in exits:
        sym = exit_info["symbol"]
        price = exit_info["current_price"]
        reason = exit_info["reason"]
        logger.info("EXIT SIGNAL: %s @ %.2f | %s", sym, price, reason)
        portfolio.close_position(sym, price=price, exit_reason=reason)

    return StepResult(
        step=12, name="Check exits", status="OK",
        duration_s=time.time() - t0,
        output={"exits_triggered": len(exits),
                "symbols_exited": [e["symbol"] for e in exits]},
    )


def step13_execute_paper_trades(
    validated_signals: dict,
    indicators: dict,
    portfolio,
    strategy_name: str,
    watchlist_entries: list,
) -> StepResult:
    """Step 13: Execute paper trades for approved BUY signals (top watchlist only)."""
    t0 = time.time()
    if portfolio is None:
        return StepResult(step=13, name="Execute trades", status="SKIP",
                          error="Portfolio unavailable")

    tier_a_symbols = {e.symbol for e in watchlist_entries if e.tier == "A"}
    trades_placed = 0

    for sym in tier_a_symbols:
        sig = validated_signals.get(sym, {})
        if not sig.get("trade_approved", False):
            continue
        ind = indicators.get(sym, {})
        price = ind.get("close", 0.0)
        if price <= 0:
            continue

        # Calculate shares from position size
        portfolio_value = portfolio.portfolio_value({sym: price})
        position_pct = sig.get("position_size_pct", 10.0) / 100.0
        budget = portfolio_value * position_pct
        shares = max(1, int(budget / price))

        try:
            from strategy.entry_exit import check_entry_conditions
            entry_result = check_entry_conditions(
                strategy_name=strategy_name,
                symbol=sym,
                indicators=ind,
                mirofish_score=sig.get("mirofish_score", 0.0),
                composite_score=sig.get("composite_signal", 0.0),
                sector_rank=1,
            )
        except Exception:
            entry_result = {"entry_ok": True}

        if not entry_result.get("entry_ok", False):
            logger.info("Entry conditions not met for %s: %s",
                        sym, entry_result.get("failed_conditions", []))
            continue

        sl = price * 0.92
        tp = price * 1.18
        success = portfolio.open_position(
            symbol=sym,
            price=price,
            shares=shares,
            strategy=strategy_name,
            stop_loss=sl,
            target_price=tp,
            meta={"composite_signal": sig.get("composite_signal", 0),
                  "regime": sig.get("regime", "SIDEWAYS")},
        )
        if success:
            trades_placed += 1

    return StepResult(
        step=13, name="Execute trades", status="OK",
        duration_s=time.time() - t0,
        output={"trades_placed": trades_placed,
                "tier_a_candidates": len(tier_a_symbols)},
    )


def step14_refresh_dashboard(regime: str, cycle_result: CycleResult) -> StepResult:
    """Step 14: Write updated HTML dashboard."""
    t0 = time.time()
    try:
        from strategy.dashboard import _build_html_dashboard
        html = _build_html_dashboard(refresh=300)
        path = Path("data/dashboard.html")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return StepResult(
            step=14, name="Refresh dashboard", status="OK",
            duration_s=time.time() - t0,
            output={"path": str(path)},
        )
    except Exception as exc:
        return StepResult(step=14, name="Refresh dashboard", status="WARN",
                          error=str(exc))


def step15_save_cycle_summary(cycle_result: CycleResult) -> StepResult:
    """Step 15: Persist cycle summary to JSON."""
    t0 = time.time()
    try:
        output_dir = Path("data/cycles")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"cycle_{cycle_result.run_date}.json"

        summary = {
            "run_date": cycle_result.run_date,
            "regime": cycle_result.regime,
            "watchlist_count": cycle_result.watchlist_count,
            "signals_generated": cycle_result.signals_generated,
            "trades_executed": cycle_result.trades_executed,
            "errors": cycle_result.errors,
            "steps": [
                {
                    "step": s.step,
                    "name": s.name,
                    "status": s.status,
                    "duration_s": round(s.duration_s, 3),
                    "output": s.output,
                    "error": s.error,
                }
                for s in cycle_result.steps
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        return StepResult(
            step=15, name="Save cycle summary", status="OK",
            duration_s=time.time() - t0,
            output={"path": str(path)},
        )
    except Exception as exc:
        return StepResult(step=15, name="Save cycle summary", status="WARN",
                          error=str(exc))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_full_strategy_cycle(
    target_date: Optional[date] = None,
    symbols: Optional[list[str]] = None,
    skip_market_check: bool = False,
    dry_run: bool = False,
) -> CycleResult:
    """
    Execute the full 15-step daily NEPSE strategy cycle.

    Parameters
    ----------
    target_date       : Date to run for (default: today)
    symbols           : Override the default analysis universe
    skip_market_check : Continue even if market is closed
    dry_run           : Run all steps but don't actually open positions

    Returns
    -------
    CycleResult with all step outcomes
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    target_date = target_date or date.today()
    symbols = symbols or ANALYSIS_UNIVERSE
    cycle = CycleResult(run_date=target_date.isoformat())

    print(f"\n{'='*65}")
    print(f"  🐟  MiroFish NEPSE Strategy Cycle  —  {target_date}")
    print(f"{'='*65}\n")

    def run_step(result: StepResult) -> bool:
        """Log step result; return False if ERROR."""
        cycle.steps.append(result)
        icon = {"OK": "✅", "WARN": "⚠️ ", "ERROR": "❌", "SKIP": "⏭️ "}.get(result.status, "?")
        duration = f"{result.duration_s:.2f}s" if result.duration_s else ""
        print(f"  Step {result.step:02d}: {icon} {result.name:<30}  {duration}")
        if result.error:
            print(f"           └─ {result.error}")
        if result.status == "ERROR":
            cycle.errors += 1
        return result.status != "ERROR"

    # ── Step 1: Market open check ──────────────────────────────────────────
    r1 = step01_check_market_open()
    run_step(r1)
    if r1.status == "WARN" and not skip_market_check:
        print("\n  Market closed today. Use --skip-market-check to run anyway.\n")
        return cycle

    # ── Step 2: Load price data ────────────────────────────────────────────
    r2 = step02_load_price_data(symbols)
    if not run_step(r2):
        print("\n  Cannot proceed without price data.\n")
        return cycle

    # ── Step 3: Compute indicators ─────────────────────────────────────────
    r3, indicators = step03_compute_indicators(symbols)
    run_step(r3)

    # ── Step 4: Detect regime ──────────────────────────────────────────────
    r4, regime_data = step04_detect_regime()
    run_step(r4)
    cycle.regime = regime_data.get("regime", "SIDEWAYS")

    # ── Step 5: Sector rotation ────────────────────────────────────────────
    r5, sector_rotation = step05_sector_rotation()
    run_step(r5)

    # ── Step 6: MiroFish signals ───────────────────────────────────────────
    r6, mirofish_signals = step06_fetch_mirofish_signals(symbols, cycle.regime)
    run_step(r6)

    # ── Step 7: Combine signals ────────────────────────────────────────────
    r7, combined_signals = step07_combine_signals(
        symbols, indicators, mirofish_signals, regime_data, sector_rotation
    )
    run_step(r7)
    cycle.signals_generated = len(combined_signals)

    # ── Step 8: Event filter ───────────────────────────────────────────────
    r8, filtered_signals = step08_apply_event_filter(combined_signals)
    run_step(r8)

    # ── Step 9: Validate trades ────────────────────────────────────────────
    # Load portfolio
    portfolio = None
    try:
        from strategy.portfolio import load_portfolio
        portfolio = load_portfolio()
    except Exception as exc:
        logger.warning("Portfolio load failed: %s", exc)

    r9, validated_signals = step09_validate_trades(
        filtered_signals, indicators, portfolio
    )
    run_step(r9)

    # ── Step 10: Select strategy ───────────────────────────────────────────
    r10, strategy_name = step10_select_strategy(regime_data)
    run_step(r10)

    # ── Step 11: Generate watchlist ────────────────────────────────────────
    r11, watchlist_entries = step11_generate_watchlist(
        symbols, combined_signals, sector_rotation, cycle.regime
    )
    run_step(r11)
    cycle.watchlist_count = len(watchlist_entries)

    # ── Step 12: Check portfolio exits ─────────────────────────────────────
    r12 = step12_check_portfolio_exits(
        portfolio, indicators, combined_signals, cycle.regime
    )
    run_step(r12)

    # ── Step 13: Execute paper trades ──────────────────────────────────────
    if not dry_run:
        r13 = step13_execute_paper_trades(
            validated_signals, indicators, portfolio,
            strategy_name, watchlist_entries
        )
        run_step(r13)
        cycle.trades_executed = r13.output.get("trades_placed", 0)
    else:
        print("  Step 13: ⏭️  Execute trades             [DRY RUN — skipped]")

    # ── Step 14: Refresh dashboard ─────────────────────────────────────────
    r14 = step14_refresh_dashboard(cycle.regime, cycle)
    run_step(r14)

    # ── Step 15: Save cycle summary ────────────────────────────────────────
    r15 = step15_save_cycle_summary(cycle)
    run_step(r15)

    # ── Final summary ──────────────────────────────────────────────────────
    ok = sum(1 for s in cycle.steps if s.status == "OK")
    warn = sum(1 for s in cycle.steps if s.status == "WARN")
    err = sum(1 for s in cycle.steps if s.status == "ERROR")

    print(f"\n{'='*65}")
    print(f"  Cycle complete  —  Regime: {cycle.regime}")
    print(f"  Steps: {ok} OK / {warn} WARN / {err} ERROR")
    print(f"  Watchlist: {cycle.watchlist_count}  |  Signals: {cycle.signals_generated}"
          f"  |  Trades: {cycle.trades_executed}")
    print(f"{'='*65}\n")

    return cycle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MiroFish daily NEPSE strategy cycle")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--skip-market-check", action="store_true",
                        help="Run even if market is closed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse but don't place paper trades")
    parser.add_argument("--symbols", nargs="*",
                        help="Override symbol universe")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    symbols = args.symbols or None

    cycle = run_full_strategy_cycle(
        target_date=target_date,
        symbols=symbols,
        skip_market_check=args.skip_market_check,
        dry_run=args.dry_run,
    )
    sys.exit(0 if cycle.errors == 0 else 1)


if __name__ == "__main__":
    main()
