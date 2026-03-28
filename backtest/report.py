"""
backtest/report.py
Generate comprehensive backtest reports: markdown summary (via Claude),
equity curve chart, monthly returns heatmap, trade analysis chart,
and JSON data export.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report prompt for Claude
# ---------------------------------------------------------------------------

REPORT_PROMPT_TEMPLATE = """\
Write a professional quantitative strategy backtest report for the \
NEPSE MiroFish trading strategy. Use the following results data:

## Performance Summary
- Period: {start_date} to {end_date}
- Total Return: {total_return_pct:+.2f}%
- Annualised Return (CAGR): {annualised_return_pct:+.2f}%
- NEPSE Benchmark Return: {benchmark_return_pct:+.2f}%
- Alpha vs Benchmark: {alpha_pct:+.2f}pp

## Risk Metrics
- Max Drawdown: {max_drawdown_pct:.2f}%
- Annual Volatility: {volatility_annual_pct:.2f}%
- Sharpe Ratio: {sharpe_ratio:.3f}
- Sortino Ratio: {sortino_ratio:.3f}
- Calmar Ratio: {calmar_ratio:.3f}

## Trade Statistics
- Total Trades: {total_trades}
- Win Rate: {win_rate_pct:.1f}%
- Average Win: {avg_win_pct:+.2f}%
- Average Loss: {avg_loss_pct:+.2f}%
- Profit Factor: {profit_factor:.2f}
- Average Hold Duration: {avg_hold_days:.1f} days

## Walk-Forward Validation
- Windows Passed: {wf_windows_passed}/{wf_windows_total}
- Optimal Weights: MiroFish {mf_weight:.0%} / Technical {tech_weight:.0%} / Sector {sec_weight:.0%}

## Stress Tests
- Tests Passed: {stress_passed}/{stress_total}
- Monte Carlo P5 Final Value: NPR {mc_p5:,.0f} ({mc_p5_pct:+.1f}% vs initial)

## Signal Attribution
- MiroFish Contribution: {mf_contribution:+.2f}pp
- Technical Contribution: {tech_contribution:+.2f}pp
- MiroFish Standalone Sharpe: {mf_sharpe:.3f}

Write a structured report with these exact sections:

1. Executive Summary (3-4 sentences, clear verdict on deployment readiness)
2. Methodology (brief description of MiroFish + technical + sector combination)
3. Performance Analysis (detailed commentary on returns, drawdown, risk-adjusted metrics)
4. Walk-Forward Validation (did strategy generalise? key findings)
5. Regime Analysis (which regimes did the strategy work best/worst in?)
6. Stress Test Results (major risks identified, how strategy responded)
7. Limitations & Risks (Nepal-specific risks: liquidity, NRB policy, data quality)
8. Deployment Recommendation (specific actionable recommendation with conditions)

Tone: factual, professional, like a hedge fund risk report. No hype.
Length: 600-900 words.

IMPORTANT DISCLAIMER (include verbatim at the end):
"This report is for research and simulation purposes only and does not \
constitute financial advice. Past backtested performance does not guarantee \
future results. All trading involves risk of loss."
"""


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def _save_equity_curve(
    daily_dates: list,
    portfolio_values: list[float],
    nepse_values: list[float],
    daily_snapshots: list,
    output_dir: Path,
) -> Optional[str]:
    """Generate equity curve chart with drawdown periods and regime labels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), height_ratios=[3, 1], sharex=True
        )
        fig.patch.set_facecolor("#0d1117")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            ax.spines[:].set_color("#30363d")

        x = list(range(len(portfolio_values)))
        pv = np.array(portfolio_values, dtype=float)
        nv = np.array(nepse_values[:len(pv)], dtype=float) if nepse_values else pv * 0

        # Normalise both to 100 at start
        pv_norm = pv / pv[0] * 100 if pv[0] > 0 else pv
        if len(nv) > 0 and nv[0] > 0:
            nv_norm = nv / nv[0] * 100
        else:
            nv_norm = np.full(len(pv_norm), 100.0)

        ax1.plot(x, pv_norm, color="#3fb950", linewidth=1.8, label="MiroFish Strategy")
        ax1.plot(x, nv_norm, color="#58a6ff", linewidth=1.2,
                 linestyle="--", label="NEPSE Index", alpha=0.8)
        ax1.axhline(y=100, color="#8b949e", linestyle=":", alpha=0.4)

        # Shade drawdown periods
        peak = pv_norm[0]
        for i in range(len(pv_norm)):
            if pv_norm[i] > peak:
                peak = pv_norm[i]
            dd = (peak - pv_norm[i]) / peak
            if dd > 0.05:
                ax1.axvspan(max(0, i-1), i, color="#f85149", alpha=0.08)

        ax1.set_ylabel("Indexed Value (Base=100)", color="#8b949e")
        ax1.set_title("NEPSE MiroFish Strategy -- Equity Curve vs Benchmark",
                      color="#c9d1d9", fontsize=12)
        ax1.legend(facecolor="#161b22", labelcolor="#c9d1d9")
        ax1.yaxis.grid(True, color="#21262d", linewidth=0.5)

        # Drawdown subplot
        drawdown = np.array([(pv_norm[:i+1].max() - v) / pv_norm[:i+1].max() * 100
                              for i, v in enumerate(pv_norm)])
        ax2.fill_between(x, -drawdown, 0, color="#f85149", alpha=0.6)
        ax2.set_ylabel("Drawdown %", color="#8b949e")
        ax2.yaxis.grid(True, color="#21262d", linewidth=0.5)

        plt.tight_layout()
        path = output_dir / "equity_curve.png"
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        logger.info("Equity curve saved -> %s", path)
        return str(path)
    except Exception as exc:
        logger.warning("Equity curve chart failed: %s", exc)
        return None


def _save_monthly_heatmap(
    daily_dates: list,
    portfolio_values: list[float],
    output_dir: Path,
) -> Optional[str]:
    """Generate monthly returns heatmap (calendar format)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # Compute monthly returns
        if not daily_dates or len(portfolio_values) < 2:
            return None

        df = pd.DataFrame({"date": daily_dates, "value": portfolio_values})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        monthly = df["value"].resample("ME").last()
        monthly_returns = monthly.pct_change().dropna() * 100

        if monthly_returns.empty:
            return None

        years = sorted(monthly_returns.index.year.unique())
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        data = np.full((len(years), 12), np.nan)
        for idx, row in monthly_returns.items():
            y_i = years.index(idx.year)
            m_i = idx.month - 1
            data[y_i, m_i] = row

        fig, ax = plt.subplots(figsize=(14, max(3, len(years) * 1.5)))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # Custom diverging colormap (red-white-green)
        from matplotlib.colors import TwoSlopeNorm
        vmax = max(abs(np.nanmax(data)), abs(np.nanmin(data)), 5)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

        # Labels
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_labels, color="#8b949e")
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels([str(y) for y in years], color="#8b949e")

        # Value annotations
        for y in range(len(years)):
            for m in range(12):
                val = data[y, m]
                if not np.isnan(val):
                    ax.text(m, y, f"{val:.1f}%", ha="center", va="center",
                            fontsize=8, color="white" if abs(val) > vmax * 0.5 else "#0d1117")

        plt.colorbar(im, ax=ax, label="Monthly Return %",
                     shrink=0.8).ax.tick_params(colors="#8b949e")
        ax.set_title("Monthly Returns Heatmap", color="#c9d1d9", fontsize=12)
        ax.spines[:].set_color("#30363d")

        plt.tight_layout()
        path = output_dir / "monthly_returns_heatmap.png"
        plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()
        logger.info("Monthly heatmap saved -> %s", path)
        return str(path)
    except Exception as exc:
        logger.warning("Monthly heatmap failed: %s", exc)
        return None


def _save_trade_analysis(trades: list[dict], output_dir: Path) -> Optional[str]:
    """4-panel trade analysis chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        if not trades:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes.flat:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            ax.spines[:].set_color("#30363d")

        pnls = [float(t.get("pnl_pct", 0)) for t in trades]
        hold_days = [float(t.get("hold_days", 0)) for t in trades]
        regimes = [t.get("regime", "UNKNOWN") for t in trades]
        mf_scores = [float(t.get("mirofish_score", 0)) for t in trades]

        # Panel 1: Win/loss distribution
        ax = axes[0, 0]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        ax.hist(wins, bins=15, color="#3fb950", alpha=0.7, label=f"Wins ({len(wins)})")
        ax.hist(losses, bins=15, color="#f85149", alpha=0.7, label=f"Losses ({len(losses)})")
        ax.set_title("Win/Loss Distribution", color="#c9d1d9")
        ax.set_xlabel("Return %", color="#8b949e")
        ax.legend(facecolor="#161b22", labelcolor="#c9d1d9")

        # Panel 2: Hold duration
        ax = axes[0, 1]
        ax.hist(hold_days, bins=20, color="#58a6ff", alpha=0.8)
        ax.set_title("Hold Duration Distribution", color="#c9d1d9")
        ax.set_xlabel("Days Held", color="#8b949e")
        ax.axvline(x=np.mean(hold_days) if hold_days else 0,
                   color="#d29922", linestyle="--", label=f"Mean={np.mean(hold_days):.1f}d")
        ax.legend(facecolor="#161b22", labelcolor="#c9d1d9")

        # Panel 3: Return by regime
        ax = axes[1, 0]
        regime_returns: dict[str, list[float]] = {}
        for r, p in zip(regimes, pnls):
            regime_returns.setdefault(r, []).append(p)
        reg_names = list(regime_returns.keys())
        reg_means = [np.mean(regime_returns[r]) for r in reg_names]
        colours = ["#3fb950" if m > 0 else "#f85149" for m in reg_means]
        ax.bar(reg_names, reg_means, color=colours, alpha=0.8)
        ax.axhline(y=0, color="#8b949e", linewidth=0.8)
        ax.set_title("Average Return by Regime", color="#c9d1d9")
        ax.set_ylabel("Avg Return %", color="#8b949e")
        ax.tick_params(axis="x", rotation=30)

        # Panel 4: MiroFish score vs return scatter
        ax = axes[1, 1]
        ax.scatter(mf_scores, pnls, alpha=0.5, color="#58a6ff", s=25)
        ax.axhline(y=0, color="#8b949e", linewidth=0.6)
        ax.axvline(x=0, color="#8b949e", linewidth=0.6)
        if len(mf_scores) > 2:
            z = np.polyfit(mf_scores, pnls, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(min(mf_scores), max(mf_scores), 100)
            ax.plot(x_line, p_fit(x_line), "#d29922", linewidth=1.5, label="Trend")
        ax.set_title("MiroFish Score vs Trade Return", color="#c9d1d9")
        ax.set_xlabel("MiroFish Score", color="#8b949e")
        ax.set_ylabel("Trade Return %", color="#8b949e")
        ax.legend(facecolor="#161b22", labelcolor="#c9d1d9")

        plt.suptitle("Trade Analysis -- NEPSE MiroFish Strategy",
                     color="#c9d1d9", fontsize=12)
        plt.tight_layout()
        path = output_dir / "trade_analysis.png"
        plt.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()
        logger.info("Trade analysis chart saved -> %s", path)
        return str(path)
    except Exception as exc:
        logger.warning("Trade analysis chart failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Claude report writer
# ---------------------------------------------------------------------------

def _generate_report_text(metrics: dict, wf: dict, stress: dict, mc: dict, attr: dict) -> str:
    """Call Claude to generate the narrative report text."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or "your_" in api_key:
        return _fallback_report_text(metrics, wf, stress, mc, attr)

    wf_weights = wf.get("avg_optimal_weights", {"mf": 0.40, "tech": 0.35, "sector": 0.25})
    mc_p5 = mc.get("p5_final_value", 1_000_000)
    initial = mc.get("initial_capital", 1_000_000)
    mc_p5_pct = (mc_p5 / initial - 1) * 100 if initial > 0 else 0

    prompt = REPORT_PROMPT_TEMPLATE.format(
        start_date=metrics.get("start_date", "2022-01-01"),
        end_date=metrics.get("end_date", "2024-12-31"),
        total_return_pct=metrics.get("total_return_pct", 0),
        annualised_return_pct=metrics.get("annualised_return_pct", 0),
        benchmark_return_pct=metrics.get("benchmark_return_pct", 0),
        alpha_pct=metrics.get("alpha_pct", 0),
        max_drawdown_pct=metrics.get("max_drawdown_pct", 0),
        volatility_annual_pct=metrics.get("volatility_annual_pct", 0),
        sharpe_ratio=metrics.get("sharpe_ratio", 0),
        sortino_ratio=metrics.get("sortino_ratio", 0),
        calmar_ratio=metrics.get("calmar_ratio", 0),
        total_trades=metrics.get("total_trades", 0),
        win_rate_pct=metrics.get("win_rate_pct", 0),
        avg_win_pct=metrics.get("avg_win_pct", 0),
        avg_loss_pct=metrics.get("avg_loss_pct", 0),
        profit_factor=metrics.get("profit_factor", 0),
        avg_hold_days=metrics.get("avg_hold_days", 0),
        wf_windows_passed=wf.get("windows_passed", 0),
        wf_windows_total=wf.get("windows_total", 5),
        mf_weight=wf_weights.get("mf", 0.40),
        tech_weight=wf_weights.get("tech", 0.35),
        sec_weight=wf_weights.get("sector", 0.25),
        stress_passed=stress.get("passed", 0),
        stress_total=stress.get("total", 5),
        mc_p5=mc_p5,
        mc_p5_pct=mc_p5_pct,
        mf_contribution=attr.get("mirofish_contribution_pp", 0),
        tech_contribution=attr.get("technical_contribution_pp", 0),
        mf_sharpe=attr.get("mf_sharpe", 0),
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as exc:
        logger.warning("Claude report generation failed: %s", exc)
        return _fallback_report_text(metrics, wf, stress, mc, attr)


def _fallback_report_text(metrics: dict, wf: dict, stress: dict, mc: dict, attr: dict) -> str:
    """Generate a structured template report when Claude is unavailable."""
    return f"""## NEPSE MiroFish Strategy -- Backtest Report

### 1. Executive Summary
Backtest period: {metrics.get('start_date', 'N/A')} to {metrics.get('end_date', 'N/A')}.
Strategy achieved {metrics.get('total_return_pct', 0):+.2f}% total return with Sharpe ratio \
{metrics.get('sharpe_ratio', 0):.3f} vs NEPSE benchmark {metrics.get('benchmark_return_pct', 0):+.2f}%.
Max drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%.

### 2. Methodology
Hybrid signal combining MiroFish social sentiment (40%), technical indicators (35%),
and sector rotation (25%). NEPSE-specific rules applied (T+3, circuit breakers, liquidity).

### 3. Performance Analysis
- Total Return: {metrics.get('total_return_pct', 0):+.2f}%
- Annualised: {metrics.get('annualised_return_pct', 0):+.2f}%
- Win Rate: {metrics.get('win_rate_pct', 0):.1f}%
- Profit Factor: {metrics.get('profit_factor', 0):.2f}

### 4. Walk-Forward Validation
{wf.get('windows_passed', 0)}/{wf.get('windows_total', 5)} windows passed (threshold: 4/5).

### 5. Regime Analysis
Strategy performs best in BULL and RECOVERY regimes. See regime_breakdown in metrics.

### 6. Stress Test Results
{stress.get('passed', 0)}/{stress.get('total', 5)} stress scenarios passed.

### 7. Limitations & Risks
- NEPSE liquidity risk: many stocks have < NPR 10M daily turnover
- NRB policy risk: sudden rate changes not predictable by technical analysis
- Data quality: historical seeds reconstructed synthetically for 2022-2024

### 8. Deployment Recommendation
{'Proceed to paper trading with reduced position sizes.' if metrics.get('sharpe_ratio', 0) >= 1.0 else 'Requires further optimisation before deployment.'}

---
*This report is for research and simulation purposes only and does not constitute \
financial advice. Past backtested performance does not guarantee future results. \
All trading involves risk of loss.*
"""


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_backtest_report(
    results,                          # BacktestResult object
    metrics: dict,
    walk_forward: Optional[dict] = None,
    stress_tests: Optional[list] = None,
    monte_carlo: Optional[dict] = None,
    attribution=None,
    output_path: str = "data/processed/backtest",
) -> dict:
    """
    Generate the full backtest report package:
    1. backtest_summary.md
    2. equity_curve.png
    3. monthly_returns_heatmap.png
    4. trade_analysis.png
    5. backtest_data.json
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    wf = walk_forward or {}
    mc = monte_carlo or {}
    attr_data = {}
    if attribution:
        attr_data = {
            "mirofish_contribution_pp": getattr(attribution, "mirofish_contribution_pp", 0),
            "technical_contribution_pp": getattr(attribution, "technical_contribution_pp", 0),
            "mf_sharpe": getattr(
                getattr(attribution, "components", {}).get("mirofish_only", None),
                "sharpe_ratio", 0
            ),
        }

    stress_summary = {}
    if stress_tests:
        stress_summary = {
            "passed": sum(1 for s in stress_tests if getattr(s, "passed", False)),
            "total": len(stress_tests),
        }

    wf_summary = {}
    if hasattr(wf, "windows_passed"):
        wf_summary = {
            "windows_passed": wf.windows_passed,
            "windows_total": wf.windows_total,
            "avg_optimal_weights": wf.avg_optimal_weights,
        }
    elif isinstance(wf, dict):
        wf_summary = wf

    # 1. Markdown report
    metrics_with_dates = {
        **metrics,
        "start_date": str(getattr(results, "start_date", "2022-01-01")),
        "end_date": str(getattr(results, "end_date", "2024-12-31")),
    }
    report_text = _generate_report_text(metrics_with_dates, wf_summary, stress_summary, mc, attr_data)
    md_path = output_dir / "backtest_summary.md"
    header = f"# NEPSE MiroFish -- Backtest Report\n**Generated:** {date.today()}\n\n---\n\n"
    md_path.write_text(header + report_text, encoding="utf-8")
    logger.info("Markdown report -> %s", md_path)
    print(f"  Report saved -> {md_path}")

    # 2. Equity curve
    dates = getattr(results, "daily_dates", [])
    pvs = getattr(results, "daily_portfolio_values", [])
    nepse = getattr(results, "nepse_index_values", [])
    snaps = getattr(results, "daily_snapshots", [])
    _save_equity_curve(dates, pvs, nepse, snaps, output_dir)

    # 3. Monthly heatmap
    _save_monthly_heatmap(dates, pvs, output_dir)

    # 4. Trade analysis
    trades_dicts = [
        {
            "pnl_pct": t.pnl_pct, "hold_days": t.hold_days,
            "regime": t.regime, "mirofish_score": t.mirofish_score,
        }
        for t in getattr(results, "trades", [])
    ]
    _save_trade_analysis(trades_dicts, output_dir)

    # 5. JSON data
    json_path = output_dir / "backtest_data.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "metrics": {k: v for k, v in metrics_with_dates.items()
                        if not isinstance(v, dict) or k in ("regime_breakdown", "sector_breakdown")},
            "walk_forward": wf_summary,
            "stress_tests": stress_summary,
            "monte_carlo": {k: v for k, v in mc.items() if k != "equity_curve_bands"},
            "attribution": attr_data,
        }, fh, indent=2, default=str)
    logger.info("JSON data -> %s", json_path)

    return {
        "report_path": str(md_path),
        "equity_curve": str(output_dir / "equity_curve.png"),
        "monthly_heatmap": str(output_dir / "monthly_returns_heatmap.png"),
        "trade_analysis": str(output_dir / "trade_analysis.png"),
        "json_data": str(json_path),
    }


def main() -> None:
    import argparse, json as _json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Generate backtest report")
    parser.add_argument("--data", default="data/processed/backtest/backtest_data.json")
    parser.add_argument("--output", default="data/processed/backtest")
    args = parser.parse_args()

    try:
        with open(args.data) as fh:
            data = _json.load(fh)
    except FileNotFoundError:
        print(f"ERROR: No backtest data at {args.data}. Run the backtest engine first.")
        return

    metrics = data.get("metrics", {})
    from types import SimpleNamespace
    results = SimpleNamespace(
        start_date=metrics.get("start_date", "2022-01-01"),
        end_date=metrics.get("end_date", "2024-12-31"),
        daily_dates=[], daily_portfolio_values=[],
        nepse_index_values=[], daily_snapshots=[], trades=[],
    )
    paths = generate_backtest_report(results, metrics, output_path=args.output)
    print(f"\nReport generated:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
