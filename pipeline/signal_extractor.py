"""
pipeline/signal_extractor.py

Parses MiroFish simulation output and produces a clean NEPSE trading signal.

Input:  simulation result dict (from mirofish_bridge.run_simulation_for_seed)
Output: structured signal dict with bull/bear score, sector signals, key themes

The analysis works by:
  1. Collecting all agent actions (posts, comments, reposts)
  2. Scoring each action's sentiment via keyword analysis (no external LLM needed)
  3. Aggregating by round -> per-round sentiment trajectory
  4. Breaking down by platform (Twitter-analog vs Reddit-analog)
  5. Extracting key themes from most-engaged content
  6. Computing sector-level signals from sector-tagged posts
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent

# ── Sentiment lexicon tuned for NEPSE context ─────────────────────────────────

BULLISH_KEYWORDS = [
    # Market direction
    "bullish", "rally", "surge", "gain", "rise", "up", "higher", "positive",
    "growth", "outperform", "beat", "strong", "opportunity", "buy", "accumulate",
    # NEPSE-specific
    "ppa rate increase", "repo rate cut", "ipo allotment", "bonus share",
    "dividend", "good earnings", "eps growth", "low pe", "undervalued",
    "nea export", "monsoon", "coalition stable", "budget infrastructure",
    "remittance high", "nprc depreciation", "usd strong", "foreign buy",
    "right share", "promoter buy", "sebon approval", "nrb relaxed",
    "circuit upper", "target", "oversubscribed",
]

BEARISH_KEYWORDS = [
    # Market direction
    "bearish", "crash", "fall", "decline", "drop", "down", "lower", "negative",
    "sell", "dump", "weak", "concern", "risk", "overvalued", "expensive",
    # NEPSE-specific
    "cd ratio high", "credit tight", "nrb tighten", "rate hike", "repo hike",
    "coalition collapse", "government fall", "political crisis", "election uncertainty",
    "bad earnings", "eps decline", "npl high", "provision increase",
    "dry season", "power shortage", "load shedding increase", "ppa cut",
    "ipo fraud", "sebon warning", "margin call", "forced sale",
    "circuit lower", "foreign sell", "inflation high", "npr depreciate",
]

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Commercial Banks":  ["bank", "banking", "nabil", "nic", "ebl", "sbi", "pcbl", "kbl", "bfi", "cd ratio", "npl", "net interest", "interest rate"],
    "Development Banks": ["development bank", "dbl", "muktinath", "jbl", "srbl", "lbbl"],
    "Finance":           ["finance company", "nfc", "sfc", "gfc", "microfinance", "mbl"],
    "Hydropower":        ["hydro", "hydropower", "nhpc", "upper", "bpcl", "nhdl", "ridi", "ppa", "nea", "mw", "megawatt", "monsoon", "dry season", "power export"],
    "Insurance":         ["insurance", "nlic", "nlicl", "licn", "life insurance", "non-life", "reinsurance", "claim", "premium"],
    "Manufacturing":     ["manufacturing", "unilever", "bottlers", "himalayan", "bottling", "noodles", "cement", "steel", "production"],
    "Hotels":            ["hotel", "tourism", "taan", "yeti", "soaltee", "sheraton", "hospitality"],
    "Microfinance":      ["microfinance", "rmdc", "swbbl", "samriddhi", "nirdhan", "ceberg"],
    "Mutual Fund":       ["mutual fund", "nefin", "kbsl", "global imi", "nabil balance", "siddhartha"],
    "IPO":               ["ipo", "initial public offering", "new listing", "allotment", "oversubscribed", "prospectus"],
}

AGENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "institutional_broker": ["broker", "institutional", "fundamental", "eps", "pe ratio", "earnings", "quarterly"],
    "retail_momentum":      ["retail", "tip", "trending", "fomo", "buy now", "moon", "🚀", "viber group"],
    "nrb_policy_watcher":   ["nrb", "repo rate", "monetary policy", "credit ratio", "cd ratio", "bank rate", "circular"],
    "hydro_analyst":        ["ppa", "nea", "mw", "megawatt", "monsoon", "dry season", "power purchase", "hydro"],
    "political_risk_analyst": ["government", "coalition", "minister", "budget", "parliament", "election", "policy"],
    "diaspora_investor":    ["nrn", "diaspora", "remittance", "usd", "gulf", "abroad", "dollar", "sending money"],
}


# ── Core scoring functions ─────────────────────────────────────────────────────

def _score_text(text: str) -> float:
    """
    Score a text snippet for bullish/bearish sentiment.
    Returns float in range [-1.0, +1.0].
    Positive = bullish, Negative = bearish.
    """
    text_lower = text.lower()
    bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def _classify_sector(text: str) -> Optional[str]:
    """Return the most-matched sector name, or None."""
    text_lower = text.lower()
    scores = {
        sector: sum(1 for kw in kws if kw in text_lower)
        for sector, kws in SECTOR_KEYWORDS.items()
    }
    best_sector, best_score = max(scores.items(), key=lambda x: x[1])
    return best_sector if best_score > 0 else None


def _classify_agent_type(action: dict) -> Optional[str]:
    """Guess agent type from action content or agent_id prefix."""
    agent_id = (action.get("agent_id") or action.get("user_id") or "").lower()
    content  = (action.get("content") or action.get("text") or "").lower()
    combined = agent_id + " " + content

    for atype, kws in AGENT_TYPE_KEYWORDS.items():
        prefix = atype.split("_")[0]
        if prefix in agent_id:
            return atype

    # Fallback: keyword match on content
    scores = {
        atype: sum(1 for kw in kws if kw in combined)
        for atype, kws in AGENT_TYPE_KEYWORDS.items()
    }
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score > 0 else None


def _extract_themes(actions: list[dict], top_n: int = 5) -> list[str]:
    """Extract top N themes by keyword frequency across all actions."""
    theme_keywords = [
        "repo rate", "ppa rate", "ipo", "bonus share", "dividend",
        "coalition", "government", "budget", "monsoon", "dry season",
        "cd ratio", "inflation", "earnings", "eps", "npl",
        "power export", "india deal", "nrb circular", "sebon",
        "right share", "merger", "acquisition", "nrn", "remittance",
    ]
    all_text = " ".join(
        (a.get("content") or a.get("text") or "").lower() for a in actions
    )
    counts = Counter()
    for kw in theme_keywords:
        n = all_text.count(kw)
        if n > 0:
            counts[kw] = n
    return [kw for kw, _ in counts.most_common(top_n)]


# ── Main extraction function ───────────────────────────────────────────────────

def extract_trading_signal(sim_output: dict) -> dict:
    """
    Parse MiroFish simulation output and produce a clean NEPSE trading signal.

    Parameters
    ----------
    sim_output : dict
        Dict with keys: simulation_id, date, actions, timeline, agent_stats, sim_config

    Returns
    -------
    dict with full signal schema
    """
    date_str   = sim_output.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    actions    = sim_output.get("actions",  [])
    timeline   = sim_output.get("timeline", [])

    # ── Per-round scoring ──────────────────────────────────────────────────────
    round_scores: dict[int, list[float]] = defaultdict(list)
    sector_scores: dict[str, list[float]] = defaultdict(list)
    platform_scores: dict[str, list[float]] = {"twitter": [], "reddit": []}
    agent_type_counts: Counter = Counter()
    all_scores: list[float] = []

    for action in actions:
        content  = action.get("content") or action.get("text") or ""
        platform = (action.get("platform") or "unknown").lower()
        round_n  = int(action.get("round_num") or action.get("round") or 0)
        score    = _score_text(content)

        round_scores[round_n].append(score)
        all_scores.append(score)

        if "twitter" in platform:
            platform_scores["twitter"].append(score)
        elif "reddit" in platform:
            platform_scores["reddit"].append(score)

        sector = _classify_sector(content)
        if sector:
            sector_scores[sector].append(score)

        atype = _classify_agent_type(action)
        if atype:
            agent_type_counts[atype] += abs(score) + 0.1   # weight by sentiment strength

    # ── Aggregate scores ───────────────────────────────────────────────────────
    def _mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    raw_round_scores = [
        _mean(round_scores.get(r, [0.0]))
        for r in sorted(round_scores.keys())
    ]

    bull_bear     = _mean(all_scores)   # -1.0 to +1.0
    twitter_score = _mean(platform_scores["twitter"])
    reddit_score  = _mean(platform_scores["reddit"])

    # ── Direction classification ───────────────────────────────────────────────
    if bull_bear >= 0.15:
        direction = "BULLISH"
    elif bull_bear <= -0.15:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # ── Signal strength ────────────────────────────────────────────────────────
    abs_score = abs(bull_bear)
    if abs_score >= 0.5:
        signal_strength = "STRONG"
    elif abs_score >= 0.25:
        signal_strength = "MODERATE"
    else:
        signal_strength = "WEAK"

    # ── Confidence: based on agent agreement rate ──────────────────────────────
    if all_scores:
        agree_sign = sum(1 for s in all_scores
                         if (s > 0) == (bull_bear > 0) or abs(s) < 0.05)
        confidence_pct = round(agree_sign / len(all_scores) * 100, 1)
    else:
        confidence_pct = 50.0

    # ── Velocity: is sentiment accelerating or decelerating? ──────────────────
    if len(raw_round_scores) >= 4:
        first_half = _mean(raw_round_scores[:len(raw_round_scores)//2])
        second_half= _mean(raw_round_scores[len(raw_round_scores)//2:])
        delta = second_half - first_half
        if abs(delta) < 0.05:
            velocity = "STABLE"
        elif (delta > 0 and bull_bear > 0) or (delta < 0 and bull_bear < 0):
            velocity = "ACCELERATING"
        else:
            velocity = "DECELERATING"
    else:
        velocity = "STABLE"

    # ── Platform agreement ─────────────────────────────────────────────────────
    platform_agree = (
        (twitter_score >= 0 and reddit_score >= 0) or
        (twitter_score <= 0 and reddit_score <= 0)
    )

    # ── Sector signals ─────────────────────────────────────────────────────────
    sector_signals = {
        sector: round(_mean(scores), 3)
        for sector, scores in sector_scores.items()
        if scores
    }
    # Include key sectors with 0.0 if no data
    for sector in ["Commercial Banks", "Hydropower", "Insurance", "Finance", "Manufacturing"]:
        if sector not in sector_signals:
            sector_signals[sector] = 0.0

    # ── Top driver agent types ─────────────────────────────────────────────────
    top_drivers = [t for t, _ in agent_type_counts.most_common(3)]

    # ── Key themes ────────────────────────────────────────────────────────────
    key_themes = _extract_themes(actions, top_n=5)

    # ── Build signal ───────────────────────────────────────────────────────────
    signal = {
        "date":                date_str,
        "simulation_id":       sim_output.get("simulation_id", ""),
        "extracted_at":        datetime.now(timezone.utc).isoformat(),
        "bull_bear_score":     round(bull_bear, 3),
        "confidence_pct":      confidence_pct,
        "direction":           direction,
        "sentiment_velocity":  velocity,
        "platform_agreement":  platform_agree,
        "twitter_score":       round(twitter_score, 3),
        "reddit_score":        round(reddit_score, 3),
        "sector_signals":      {k: round(v, 3) for k, v in sorted(
                                    sector_signals.items(),
                                    key=lambda x: abs(x[1]), reverse=True)},
        "top_driver_agent_types": top_drivers,
        "key_themes":          key_themes,
        "raw_round_scores":    [round(s, 3) for s in raw_round_scores],
        "signal_strength":     signal_strength,
        "total_actions":       len(actions),
        "quality_flags":       [],   # populated by simulation_qa.py
    }
    return signal


def format_signal_table(signal: dict) -> str:
    """Format signal as a printable summary table."""
    d   = signal.get("direction", "N/A")
    bb  = signal.get("bull_bear_score", 0)
    con = signal.get("confidence_pct", 0)
    vel = signal.get("sentiment_velocity", "N/A")
    pa  = "YES" if signal.get("platform_agreement") else "NO"
    ss  = signal.get("signal_strength", "N/A")
    tw  = signal.get("twitter_score", 0)
    rd  = signal.get("reddit_score", 0)
    date= signal.get("date", "")

    sector_lines = "\n".join(
        f"    {sec:<24} : {score:+.3f}"
        for sec, score in list(signal.get("sector_signals", {}).items())[:6]
    )
    themes = ", ".join(signal.get("key_themes", []))
    drivers= ", ".join(signal.get("top_driver_agent_types", []))
    flags  = "; ".join(signal.get("quality_flags", [])) or "None"

    sign_char = "+" if bb >= 0 else ""
    return f"""
============================================================
  NEPSE MIROFISH SIGNAL -- {date}
============================================================
  Direction        : {d}
  Bull/Bear Score  : {sign_char}{bb:.3f}
  Signal Strength  : {ss}
  Confidence       : {con:.1f}%
  Velocity         : {vel}
  Platforms agree  : {pa}  (Twitter: {tw:+.3f} | Reddit: {rd:+.3f})
------------------------------------------------------------
  Sector signals:
{sector_lines}
------------------------------------------------------------
  Top drivers      : {drivers}
  Key themes       : {themes}
  Actions analysed : {signal.get('total_actions', 0)}
  Quality flags    : {flags}
============================================================"""


def save_signal(signal: dict, out_dir: Optional[Path] = None) -> Path:
    """Save signal JSON to data/processed/signals/signal_YYYY-MM-DD.json."""
    if out_dir is None:
        out_dir = _ROOT / "data" / "processed" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = signal.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    out_path = out_dir / f"signal_{date_str}.json"
    out_path.write_text(
        json.dumps(signal, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    ap = argparse.ArgumentParser(description="Extract trading signal from simulation output")
    ap.add_argument("--sim",  default=None,
                    help="Path to simulation JSON (default: latest processed/simulations/)")
    ap.add_argument("--save", action="store_true", help="Save signal to disk")
    args = ap.parse_args()

    if args.sim:
        sim_path = Path(args.sim)
    else:
        sim_dir = _ROOT / "data" / "processed" / "simulations"
        sims    = sorted(sim_dir.glob("sim_transcript_*.json"), reverse=True)
        if not sims:
            print("[ERROR] No simulation transcripts found. Run pipeline/run_simulation.py first.")
            sys.exit(1)
        sim_path = sims[0]

    print(f"Loading simulation: {sim_path.name}")
    sim_output = json.loads(sim_path.read_text(encoding="utf-8"))

    signal = extract_trading_signal(sim_output)
    print(format_signal_table(signal))

    if args.save:
        out = save_signal(signal)
        print(f"\nSignal saved -> {out}")
