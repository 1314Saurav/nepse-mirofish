"""
backtest/signal_attribution.py
Break down strategy performance to understand contribution of each signal.
Compares: MiroFish-only, Technical-only, Sector-only, Full-Hybrid backtests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attribution component configurations
# ---------------------------------------------------------------------------

ATTRIBUTION_COMPONENTS = {
    "mirofish_only": {
        "label": "MiroFish only",
        "description": "MiroFish signals at full weight; technical and sector set to neutral (0.5)",
        "weights": {"mf": 1.0, "tech": 0.0, "sector": 0.0},
        "mf_score_override": None,         # use real MF score
        "technical_score_override": 0.5,   # neutral
        "sector_score_override": 0.5,      # neutral
    },
    "technical_only": {
        "label": "Technical only",
        "description": "Technical signals at full weight; MiroFish and sector set to neutral",
        "weights": {"mf": 0.0, "tech": 1.0, "sector": 0.0},
        "mf_score_override": 0.5,
        "technical_score_override": None,
        "sector_score_override": 0.5,
    },
    "sector_only": {
        "label": "Sector only",
        "description": "Sector rotation signals at full weight; others neutral",
        "weights": {"mf": 0.0, "tech": 0.0, "sector": 1.0},
        "mf_score_override": 0.5,
        "technical_score_override": 0.5,
        "sector_score_override": None,
    },
    "full_hybrid": {
        "label": "Full hybrid",
        "description": "Standard combined strategy (MF 40% / Tech 35% / Sector 25%)",
        "weights": {"mf": 0.40, "tech": 0.35, "sector": 0.25},
        "mf_score_override": None,
        "technical_score_override": None,
        "sector_score_override": None,
    },
}


@dataclass
class ComponentResult:
    component: str
    label: str
    annual_return_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    max_drawdown_pct: float
    alpha_pct: float


@dataclass
class AttributionResult:
    components: dict[str, ComponentResult] = field(default_factory=dict)
    benchmark_annual_return: float = 0.0
    mirofish_contribution_pp: float = 0.0
    technical_contribution_pp: float = 0.0
    sector_contribution_pp: float = 0.0
    synergy_bonus_pp: float = 0.0
    top_mf_correct_trades: list[dict] = field(default_factory=list)
    top_mf_wrong_trades: list[dict] = field(default_factory=list)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Backtest runner with component override
# ---------------------------------------------------------------------------

def _run_component_backtest(
    start: str,
    end: str,
    component_config: dict,
    capital: float = 500_000.0,
) -> dict:
    """
    Run a backtest with one signal component at full weight, others neutralised.
    Returns metrics dict.
    """
    try:
        from backtest.engine import NEPSEBacktestEngine
        from backtest.metrics import compute_metrics

        weights = component_config["weights"]
        strategy_config = {
            "signal_weights": weights,
            "mf_score_override": component_config.get("mf_score_override"),
            "technical_score_override": component_config.get("technical_score_override"),
            "sector_score_override": component_config.get("sector_score_override"),
        }

        engine = NEPSEBacktestEngine(
            start_date=start,
            end_date=end,
            initial_capital_npr=capital,
            strategy_config=strategy_config,
        )
        result = engine.run()

        if len(result.daily_portfolio_values) < 5:
            return {"annualised_return_pct": 0.0, "sharpe_ratio": 0.0,
                    "win_rate_pct": 0.0, "total_trades": 0,
                    "max_drawdown_pct": 99.0, "alpha_pct": 0.0,
                    "trades": []}

        trades_dicts = [
            {
                "pnl_pct": t.pnl_pct, "regime": t.regime,
                "sector": t.sector, "hold_days": t.hold_days,
                "mirofish_score": t.mirofish_score, "action": "SELL",
                "symbol": t.symbol, "entry_date": t.entry_date.isoformat(),
            }
            for t in result.trades
        ]
        metrics = compute_metrics(
            daily_portfolio_values=result.daily_portfolio_values,
            trades=trades_dicts,
            nepse_index_values=result.nepse_index_values or result.daily_portfolio_values,
        )
        metrics["trades"] = trades_dicts
        return metrics

    except Exception as exc:
        logger.warning("Component backtest failed (%s): %s", component_config["label"], exc)
        return {"annualised_return_pct": 0.0, "sharpe_ratio": 0.0,
                "win_rate_pct": 0.0, "total_trades": 0,
                "max_drawdown_pct": 99.0, "alpha_pct": 0.0, "trades": []}


# ---------------------------------------------------------------------------
# Trade-level attribution
# ---------------------------------------------------------------------------

def _find_mf_best_worst(trades: list[dict], n: int = 10) -> tuple[list[dict], list[dict]]:
    """
    Find trades where MiroFish signal was most/least predictive.
    Score = mirofish_score x pnl_pct (high score + high return = MF was right)
    """
    scored = []
    for t in trades:
        mf = float(t.get("mirofish_score", 0))
        pnl = float(t.get("pnl_pct", 0))
        attribution_score = mf * pnl  # both positive = correct; opposite signs = wrong
        scored.append({**t, "mf_attribution_score": round(attribution_score, 4)})

    scored.sort(key=lambda x: x["mf_attribution_score"], reverse=True)
    best = scored[:n]
    worst = scored[-n:]
    return best, worst


# ---------------------------------------------------------------------------
# Main attribution analysis
# ---------------------------------------------------------------------------

def run_attribution_analysis(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    capital: float = 500_000.0,
) -> AttributionResult:
    """
    Run 4 component backtests and compute contribution of each signal source.
    Prints a formatted attribution table.
    """
    print(f"\n{'='*62}")
    print(f"  Signal Attribution Analysis ({start} to {end})")
    print(f"{'='*62}")

    result = AttributionResult()
    component_metrics: dict[str, dict] = {}

    for comp_id, config in ATTRIBUTION_COMPONENTS.items():
        logger.info("Running %s attribution backtest...", config["label"])
        print(f"  Running: {config['label']}...", end=" ", flush=True)

        metrics = _run_component_backtest(start, end, config, capital)
        component_metrics[comp_id] = metrics

        cr = ComponentResult(
            component=comp_id,
            label=config["label"],
            annual_return_pct=round(metrics.get("annualised_return_pct", 0.0), 2),
            sharpe_ratio=round(metrics.get("sharpe_ratio", 0.0), 3),
            win_rate_pct=round(metrics.get("win_rate_pct", 0.0), 1),
            total_trades=metrics.get("total_trades", 0),
            max_drawdown_pct=round(metrics.get("max_drawdown_pct", 0.0), 2),
            alpha_pct=round(metrics.get("alpha_pct", 0.0), 2),
        )
        result.components[comp_id] = cr
        print(f"done  (Sharpe={cr.sharpe_ratio:.2f}  Return={cr.annual_return_pct:+.1f}%)")

    # Compute contributions
    hybrid = result.components.get("full_hybrid")
    mf_only = result.components.get("mirofish_only")
    tech_only = result.components.get("technical_only")
    sector_only = result.components.get("sector_only")

    if hybrid and mf_only and tech_only and sector_only:
        hybrid_r = hybrid.annual_return_pct
        mf_r = mf_only.annual_return_pct
        tech_r = tech_only.annual_return_pct
        sec_r = sector_only.annual_return_pct

        # Simple additive attribution (weighted contributions)
        mf_contrib = hybrid_r * 0.40 - mf_r * 0.40  # delta from using full MF vs neutral
        tech_contrib = hybrid_r * 0.35 - tech_r * 0.35
        sec_contrib = hybrid_r * 0.25 - sec_r * 0.25
        # Sum of components vs actual hybrid (interaction/synergy)
        synergy = hybrid_r - (mf_r * 0.40 + tech_r * 0.35 + sec_r * 0.25)

        result.mirofish_contribution_pp = round(mf_contrib, 2)
        result.technical_contribution_pp = round(tech_contrib, 2)
        result.sector_contribution_pp = round(sec_contrib, 2)
        result.synergy_bonus_pp = round(synergy, 2)

    # MiroFish trade-level attribution
    hybrid_trades = component_metrics.get("full_hybrid", {}).get("trades", [])
    if hybrid_trades:
        result.top_mf_correct_trades, result.top_mf_wrong_trades = \
            _find_mf_best_worst(hybrid_trades)

    # Generate recommendation
    result.recommendation = _generate_recommendation(result)

    _print_attribution_table(result)
    return result


def _generate_recommendation(result: AttributionResult) -> str:
    mf = result.components.get("mirofish_only")
    if not mf:
        return "Insufficient data for recommendation."

    if mf.sharpe_ratio >= 0.8:
        return (
            "MiroFish adds significant alpha (Sharpe > 0.8 standalone). "
            "Invest in improving agent personas and increasing simulation count."
        )
    elif mf.sharpe_ratio >= 0.6:
        return (
            "MiroFish is moderately useful (Sharpe 0.6-0.8). "
            "Focus on improving NRB policy watcher and institutional broker agents."
        )
    else:
        return (
            "MiroFish standalone Sharpe < 0.6. Agent personas need richer prompts "
            "and more NEPSE-specific training data before deploying real capital."
        )


def _print_attribution_table(result: AttributionResult) -> None:
    G, R, Y, B = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

    print(f"\n{'='*62}")
    print(f"  {'Component':<22} {'Ann.Return':>11} {'Sharpe':>7} {'WinRate':>8}")
    print(f"  {'-'*60}")

    for comp_id in ["mirofish_only", "technical_only", "sector_only", "full_hybrid"]:
        cr = result.components.get(comp_id)
        if not cr:
            continue
        is_hybrid = comp_id == "full_hybrid"
        ret_c = G if cr.annual_return_pct > 0 else R
        sharp_c = G if cr.sharpe_ratio >= 1.0 else (Y if cr.sharpe_ratio >= 0.6 else R)
        bold = "\033[1m" if is_hybrid else ""
        print(
            f"  {bold}{cr.label:<22}{B} "
            f"{ret_c}{cr.annual_return_pct:>+10.2f}%{B} "
            f"{sharp_c}{cr.sharpe_ratio:>7.3f}{B} "
            f"{cr.win_rate_pct:>7.1f}%"
        )

    print(f"  {'-'*60}")
    print(f"\n  Contribution breakdown (percentage points of annual return):")
    mf_c = G if result.mirofish_contribution_pp > 0 else R
    t_c = G if result.technical_contribution_pp > 0 else R
    s_c = G if result.sector_contribution_pp > 0 else R
    print(f"    MiroFish contribution:   {mf_c}{result.mirofish_contribution_pp:>+7.2f}pp{B}")
    print(f"    Technical contribution:  {t_c}{result.technical_contribution_pp:>+7.2f}pp{B}")
    print(f"    Sector contribution:     {s_c}{result.sector_contribution_pp:>+7.2f}pp{B}")
    print(f"    Synergy bonus:           {result.synergy_bonus_pp:>+7.2f}pp")

    print(f"\n  Recommendation: {result.recommendation}")
    print(f"{'='*62}\n")

    if result.top_mf_correct_trades:
        print("  Top 3 'MiroFish was right' trades:")
        for t in result.top_mf_correct_trades[:3]:
            print(f"    {t.get('symbol','?'):<8} MF={t.get('mirofish_score',0):+.2f}  "
                  f"PnL={t.get('pnl_pct',0):+.1f}%  "
                  f"Score={t.get('mf_attribution_score',0):+.3f}")

    if result.top_mf_wrong_trades:
        print("  Top 3 'MiroFish was wrong' trades:")
        for t in result.top_mf_wrong_trades[:3]:
            print(f"    {t.get('symbol','?'):<8} MF={t.get('mirofish_score',0):+.2f}  "
                  f"PnL={t.get('pnl_pct',0):+.1f}%  "
                  f"Score={t.get('mf_attribution_score',0):+.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Signal attribution analysis")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=500_000.0)
    args = parser.parse_args()
    run_attribution_analysis(args.start, args.end, args.capital)

if __name__ == "__main__":
    main()
