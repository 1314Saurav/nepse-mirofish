"""
strategy/sector_rotation.py

NEPSE sector rotation tracker.
Identifies which sectors are leading/lagging and generates rotation signals.
Combines price momentum with MiroFish simulation sector signals.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategy.technical_indicators import get_indicators_for_symbol
from db.models import get_engine, StockPrice
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_PROC_DIR = _ROOT / "data" / "processed"

# ── NEPSE sector definitions ──────────────────────────────────────────────────

NEPSE_SECTORS: dict[str, list[str]] = {
    "banking":       ["NABIL", "NBBL", "EBL", "NIC", "SBI", "PCBL", "KBL", "NIMB",
                      "ADBL", "CZBIL", "HBL", "MBL", "PRVU", "SANIMA", "SCB"],
    "hydropower":    ["NHPC", "UPPER", "BPCL", "RIDI", "NHDL", "GHL", "DORDI",
                      "SHPC", "AKPL", "HDHPC", "KPCL", "NGPL", "MHNL"],
    "insurance":     ["NLIC", "NLICL", "LICN", "SRLI", "ALICL", "SICL",
                      "NICL", "PICL", "PRIN", "RLICL"],
    "finance":       ["GUFL", "ICFC", "JFL", "MPFL", "SFCL",
                      "CFCL", "GFCL", "MFIL", "SIFC"],
    "microfinance":  ["CBBL", "FOWAD", "NMBMF", "SMFBS",
                      "DDBL", "GILB", "JBBL", "MERO", "NMLBBL"],
    "manufacturing": ["SHIVM", "BNT", "RARA", "GCIL", "HDL", "NICA", "NTC"],
    "hotels":        ["OHL", "TRH", "CGH"],
    "trading":       ["BBC", "STC"],
}

# Sector display names
SECTOR_DISPLAY = {
    "banking":       "Commercial Banks",
    "hydropower":    "Hydropower",
    "insurance":     "Insurance",
    "finance":       "Finance Companies",
    "microfinance":  "Microfinance",
    "manufacturing": "Manufacturing & Processing",
    "hotels":        "Hotels & Tourism",
    "trading":       "Trading",
}


# ── Core momentum computation ─────────────────────────────────────────────────

def _load_sector_returns(days: int = 30) -> dict[str, dict[str, float]]:
    """
    Load recent price data for all sector stocks and compute returns.
    Returns: {symbol: {"ret_5d": float, "ret_10d": float, "ret_20d": float,
                       "vol_ratio": float, "volume_avg": float}}
    """
    try:
        engine = get_engine()
        with Session(engine) as session:
            # Get all symbols we care about
            all_symbols = [s for stocks in NEPSE_SECTORS.values() for s in stocks]
            rows = (
                session.query(StockPrice)
                .filter(StockPrice.symbol.in_(all_symbols))
                .order_by(StockPrice.symbol, StockPrice.date.desc())
                .all()
            )
    except Exception as e:
        logger.warning("Could not load sector data from DB: %s", e)
        return {}

    # Group by symbol and compute returns
    from collections import defaultdict
    symbol_rows: dict[str, list] = defaultdict(list)
    for r in rows:
        symbol_rows[r.symbol].append(r)

    result = {}
    for symbol, sym_rows in symbol_rows.items():
        sym_rows_sorted = sorted(sym_rows, key=lambda x: x.date, reverse=True)
        prices = [float(r.close or r.ltp or 0) for r in sym_rows_sorted if (r.close or r.ltp)]
        volumes = [float(r.volume or 0) for r in sym_rows_sorted]

        if len(prices) < 21:
            continue

        def ret(n: int) -> float:
            if len(prices) <= n or prices[n] == 0:
                return 0.0
            return (prices[0] - prices[n]) / prices[n] * 100

        vol_avg = float(np.mean(volumes[:20])) if volumes else 0.0
        vol_ratio = float(volumes[0] / vol_avg) if vol_avg > 0 else 1.0

        result[symbol] = {
            "ret_5d":     ret(5),
            "ret_10d":    ret(10),
            "ret_20d":    ret(20),
            "vol_ratio":  vol_ratio,
            "volume_avg": vol_avg,
            "price":      prices[0],
        }
    return result


def _load_nepse_return(days: int = 20) -> float:
    """Load NEPSE index N-day return for relative strength calculation."""
    try:
        from db.models import MarketSnapshot
        engine = get_engine()
        with Session(engine) as session:
            rows = (
                session.query(MarketSnapshot)
                .filter(MarketSnapshot.nepse_index.isnot(None))
                .order_by(MarketSnapshot.date.desc())
                .limit(days + 1)
                .all()
            )
        if len(rows) >= days + 1:
            latest = float(rows[0].nepse_index or 0)
            prior  = float(rows[days].nepse_index or 0)
            if prior > 0:
                return (latest - prior) / prior * 100
    except Exception as e:
        logger.debug("Could not load NEPSE index return: %s", e)
    return 0.0


def compute_sector_momentum(days: int = 20) -> dict[str, dict]:
    """
    Compute momentum metrics for each NEPSE sector.

    Returns:
        {
            sector_name: {
                "sector_return_pct": float,
                "relative_strength": float,      # >1 = outperforming NEPSE
                "momentum_score": float,          # composite 5d/10d/20d
                "volume_trend": str,              # "rising"|"falling"|"flat"
                "leader_stocks": list[str],       # top 2 stocks
                "stocks_data": dict,
                "coverage": int,                  # number of stocks with data
            }
        }
    Sorted by momentum_score descending.
    """
    symbol_data = _load_sector_returns(days + 5)
    nepse_ret20  = _load_nepse_return(20)
    nepse_ret5   = _load_nepse_return(5)
    nepse_ret10  = _load_nepse_return(10)

    sector_results: dict[str, dict] = {}

    for sector, symbols in NEPSE_SECTORS.items():
        stocks_with_data = {s: symbol_data[s] for s in symbols if s in symbol_data}
        if not stocks_with_data:
            sector_results[sector] = {
                "sector_return_pct": 0.0, "relative_strength": 1.0,
                "momentum_score": 0.0, "volume_trend": "flat",
                "leader_stocks": [], "stocks_data": {}, "coverage": 0,
            }
            continue

        # Average returns across sector stocks
        ret_5d  = float(np.mean([d["ret_5d"]  for d in stocks_with_data.values()]))
        ret_10d = float(np.mean([d["ret_10d"] for d in stocks_with_data.values()]))
        ret_20d = float(np.mean([d["ret_20d"] for d in stocks_with_data.values()]))
        avg_vol = float(np.mean([d["vol_ratio"] for d in stocks_with_data.values()]))

        # Relative strength vs NEPSE
        nepse_ret = nepse_ret20 if nepse_ret20 != 0 else 0.001
        rel_strength = ret_20d / abs(nepse_ret) if nepse_ret != 0 else 1.0

        # Composite momentum score (weighted: recent counts more)
        rel5  = ret_5d  - nepse_ret5
        rel10 = ret_10d - nepse_ret10
        rel20 = ret_20d - nepse_ret20
        momentum_score = (0.50 * rel5 + 0.30 * rel10 + 0.20 * rel20) / 10.0
        # Normalise to roughly -1 to +1
        momentum_score = float(np.clip(momentum_score, -1.0, 1.0))

        # Volume trend
        if avg_vol > 1.2:
            volume_trend = "rising"
        elif avg_vol < 0.8:
            volume_trend = "falling"
        else:
            volume_trend = "flat"

        # Top 2 leader stocks (by 20-day return)
        sorted_stocks = sorted(
            stocks_with_data.items(),
            key=lambda x: x[1]["ret_20d"], reverse=True
        )
        leader_stocks = [sym for sym, _ in sorted_stocks[:2]]

        sector_results[sector] = {
            "sector_return_pct": round(ret_20d, 2),
            "relative_strength": round(rel_strength, 3),
            "momentum_score":    round(momentum_score, 3),
            "volume_trend":      volume_trend,
            "leader_stocks":     leader_stocks,
            "stocks_data":       stocks_with_data,
            "coverage":          len(stocks_with_data),
            "display_name":      SECTOR_DISPLAY.get(sector, sector.title()),
        }

    # Sort by momentum_score descending
    return dict(sorted(
        sector_results.items(),
        key=lambda x: x[1]["momentum_score"], reverse=True
    ))


def get_rotation_signal(
    mirofish_sector_signals: Optional[dict] = None,
    days: int = 20,
    save: bool = True,
) -> dict:
    """
    Generate sector rotation recommendation combining price momentum
    and MiroFish simulation sector signals.

    Args:
        mirofish_sector_signals: From pipeline/signal_extractor.py
                                 {"hydropower": 0.84, "banking": 0.71, ...}
        days: Lookback for momentum calculation.
        save: Save result to data/processed/

    Returns:
        {
            "date": str,
            "overweight": str,       # top sector to increase exposure
            "underweight": str,      # weakest sector to reduce
            "sector_rankings": [...],# ordered list with scores
            "rotation_narrative": str,
        }
    """
    price_momentum = compute_sector_momentum(days)
    today_str = date.today().isoformat()

    # Build combined scores
    rankings = []
    for sector, pm in price_momentum.items():
        # MiroFish boost (if available)
        mf_score = 0.0
        if mirofish_sector_signals:
            # Try sector name and display name
            mf_score = (
                mirofish_sector_signals.get(sector, 0.0) or
                mirofish_sector_signals.get(SECTOR_DISPLAY.get(sector, ""), 0.0)
            )

        combined = (0.65 * pm["momentum_score"]) + (0.35 * float(mf_score))

        rankings.append({
            "sector":         sector,
            "display_name":   pm["display_name"],
            "price_momentum": pm["momentum_score"],
            "mirofish_score": float(mf_score),
            "combined_score": round(combined, 3),
            "sector_return":  pm["sector_return_pct"],
            "volume_trend":   pm["volume_trend"],
            "leader_stocks":  pm["leader_stocks"],
            "coverage":       pm["coverage"],
        })

    rankings.sort(key=lambda x: x["combined_score"], reverse=True)

    overweight   = rankings[0]["display_name"] if rankings else "Unknown"
    underweight  = rankings[-1]["display_name"] if rankings else "Unknown"
    top_sector   = rankings[0] if rankings else {}
    weak_sector  = rankings[-1] if rankings else {}

    narrative = (
        f"Sector rotation favours {overweight} "
        f"(momentum {top_sector.get('price_momentum', 0):+.2f}, "
        f"leaders: {', '.join(top_sector.get('leader_stocks', [])[:2])}). "
        f"Avoid {underweight} "
        f"(momentum {weak_sector.get('price_momentum', 0):+.2f})."
    )

    result = {
        "date":              today_str,
        "overweight":        overweight,
        "overweight_sector": rankings[0]["sector"] if rankings else "",
        "underweight":       underweight,
        "underweight_sector": rankings[-1]["sector"] if rankings else "",
        "sector_rankings":   rankings,
        "rotation_narrative": narrative,
    }

    if save:
        out_path = _PROC_DIR / f"sector_rotation_{today_str}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
        logger.info("Sector rotation saved → %s", out_path)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Computing NEPSE sector rotation...")
    result = get_rotation_signal()
    print(f"\nDate: {result['date']}")
    print(f"OVERWEIGHT:  {result['overweight']}")
    print(f"UNDERWEIGHT: {result['underweight']}")
    print(f"\nSector Rankings:")
    print(f"{'Rank':>4}  {'Sector':20s}  {'Combined':>8}  {'Momentum':>8}  {'Volume':8}  Leaders")
    for i, r in enumerate(result["sector_rankings"], 1):
        leaders = ", ".join(r["leader_stocks"][:2]) if r["leader_stocks"] else "N/A"
        print(f"  {i:2d}. {r['display_name']:20s}  {r['combined_score']:+8.3f}  "
              f"{r['price_momentum']:+8.3f}  {r['volume_trend']:8s}  {leaders}")
    print(f"\n{result['rotation_narrative']}")
