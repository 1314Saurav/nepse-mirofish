"""
pipeline/generate_simulation_question.py

Dynamically generates the best simulation question for the day using Claude.

The question focuses agents on the single most market-moving event/theme
in today's seed, framing it around investor sentiment evolution — not prices.

Usage:
    from pipeline.generate_simulation_question import generate_simulation_question
    question = generate_simulation_question(seed)

    # Or as CLI:
    python pipeline/generate_simulation_question.py
    python pipeline/generate_simulation_question.py --all   # last 3 seeds
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

SEED_DIR = _ROOT / "data" / "seed"

SYSTEM_PROMPT = """You are a NEPSE (Nepal Stock Exchange) market analyst.
Given today's market seed data, generate ONE simulation question for a
multi-agent sentiment prediction engine.

The question MUST:
1. Focus on the single most market-moving event from today's data
2. Reference specific NEPSE sectors or companies (e.g., banking, hydropower, NABIL, UPPER)
3. Ask how INVESTOR SENTIMENT will EVOLVE — not what the price will be
4. Include a time horizon of 2-5 trading days
5. Be under 80 words
6. Be specific to Nepal's market context (not generic global market language)

Good examples:
- "NRB announced an unexpected 50bps repo rate cut today. How will NEPSE's banking sector investors respond over the next 3 trading days, and will the bullish sentiment spread to hydropower and insurance sectors?"
- "Upper Tamakoshi declared a 35% bonus share. How will retail momentum traders and institutional brokers differ in their hydropower sector sentiment this week?"
- "Nepal's ruling coalition collapsed as the Finance Minister resigned. How will political risk analysts and diaspora investors influence NEPSE market sentiment over the next 4 trading days?"

Return ONLY the question text, no preamble, no quotes."""


def _build_context_prompt(seed: dict) -> str:
    """Build the user prompt from seed data for Claude."""
    ms   = seed.get("market_summary", {})
    mc   = seed.get("macro_context", {})
    ipos = seed.get("ipo_events", [])
    news = seed.get("news_articles", [])
    date = seed.get("date", "today")
    pol  = seed.get("political_context", "")

    # Extract top news titles
    top_news = [a.get("title", "") for a in news[:10] if a.get("title")]
    news_str = "\n".join(f"  - {t}" for t in top_news)

    idx = ms.get("nepse_index", "N/A")
    pct = ms.get("nepse_pct_change", "N/A")

    gainers = [g["symbol"] for g in ms.get("top_gainers", [])[:3] if isinstance(g, dict)]
    losers  = [l["symbol"] for l in ms.get("top_losers",  [])[:3] if isinstance(l, dict)]

    prompt = f"""Generate a simulation question for NEPSE market sentiment simulation on {date}.

TODAY'S KEY DATA:
- NEPSE Index: {idx} (change: {pct}%)
- Top gainers: {', '.join(gainers) if gainers else 'N/A'}
- Top losers:  {', '.join(losers) if losers else 'N/A'}
- NRB Repo Rate: {mc.get('repo_rate', 'N/A')}%
- Bank Rate: {mc.get('bank_rate', 'N/A')}%
- USD/NPR: {mc.get('usd_npr', 'N/A')}
- CD Ratio: {mc.get('credit_deposit_ratio', 'N/A')}%
- IPO Events: {'; '.join(ipos[:3]) if ipos else 'None'}
- Political context: {pol or 'None noted'}

TOP NEWS HEADLINES:
{news_str or '  - No major headlines available'}

Based on the above, identify the single most market-moving factor and write the simulation question."""

    return prompt


def generate_simulation_question(seed: dict,
                                  fallback: bool = True) -> str:
    """
    Use Claude to generate a context-aware simulation question from seed data.

    Falls back to a rule-based question if Claude API is unavailable.

    Parameters
    ----------
    seed     : the daily seed dict
    fallback : if True, use rule-based fallback when API is down

    Returns
    -------
    str — the simulation question (under 80 words)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key or api_key.startswith("your_"):
        if fallback:
            return _rule_based_question(seed)
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        user_prompt = _build_context_prompt(seed)
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        question = resp.content[0].text.strip()

        # Enforce 80-word limit (soft — truncate at sentence boundary)
        words = question.split()
        if len(words) > 90:
            question = " ".join(words[:80]).rstrip(",;") + "?"

        return question

    except Exception as exc:
        print(f"  [WARN] Claude API failed for question generation: {exc}",
              file=sys.stderr)
        if fallback:
            return _rule_based_question(seed)
        raise


def _rule_based_question(seed: dict) -> str:
    """
    Fallback: build a rule-based simulation question from seed fields.
    Priority: IPO events > NRB rate changes > index performance > generic
    """
    ms   = seed.get("market_summary", {})
    mc   = seed.get("macro_context", {})
    ipos = seed.get("ipo_events", [])
    news = seed.get("news_articles", [])
    date = seed.get("date", "today")

    # IPO event question
    open_ipos = [e for e in ipos if "OPEN IPO" in e.upper()]
    if open_ipos:
        company = open_ipos[0].split(":")[1].strip().split("@")[0].strip()
        return (f"With {company} IPO currently open, how will NEPSE retail traders "
                f"and institutional brokers differ in sentiment toward new listings "
                f"vs existing stocks over the next 3 trading days?")

    # NRB rate context
    repo = mc.get("repo_rate")
    bank = mc.get("bank_rate")
    cd   = mc.get("credit_deposit_ratio")
    if cd and float(cd) > 85:
        return (f"Nepal's credit-to-deposit ratio stands at {cd}%, near NRB's 90% ceiling. "
                f"How will NRB policy watchers and institutional brokers influence banking "
                f"sector sentiment in NEPSE over the next 3 trading days?")

    # Index performance
    pct = ms.get("nepse_pct_change", 0)
    try:
        pct_f = float(pct)
    except (TypeError, ValueError):
        pct_f = 0.0

    if abs(pct_f) > 1.5:
        direction = "surged" if pct_f > 0 else "dropped"
        return (f"NEPSE {direction} {abs(pct_f):.1f}% today. "
                f"How will this momentum affect retail trader FOMO and institutional "
                f"broker caution over the next 3 trading days?")

    # Generic NEPSE question
    return (f"Given today's NEPSE market conditions on {date} — including current "
            f"NRB repo rate ({repo}%), CD ratio ({cd}%), and forex levels — "
            f"how will different investor groups position themselves over the next "
            f"3 trading days and what sector will see the most sentiment shift?")


def generate_for_recent_seeds(n: int = 3) -> list[dict]:
    """
    Generate simulation questions for the last N seed files.
    Returns list of {date, question, source} dicts.
    """
    seeds    = sorted(SEED_DIR.glob("seed_*.json"), reverse=True)[:n]
    results  = []
    for seed_path in seeds:
        seed     = json.loads(seed_path.read_text(encoding="utf-8"))
        question = generate_simulation_question(seed, fallback=True)
        source   = "claude" if os.getenv("ANTHROPIC_API_KEY","").startswith("sk-ant") else "rule-based"
        results.append({
            "date":     seed.get("date"),
            "question": question,
            "source":   source,
        })
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    BOLD = "\033[1m"; CYAN = "\033[96m"; GREEN = "\033[92m"; RESET = "\033[0m"

    ap = argparse.ArgumentParser(description="Generate NEPSE simulation question from seed data")
    ap.add_argument("--seed", default=None, help="Path to seed JSON")
    ap.add_argument("--all",  action="store_true", help="Generate for last 3 seeds")
    args = ap.parse_args()

    if args.all:
        print(f"\n{BOLD}Generating simulation questions for last 3 seeds...{RESET}\n")
        results = generate_for_recent_seeds(3)
        for i, r in enumerate(results, 1):
            print(f"{CYAN}[{i}] {r['date']} ({r['source']}){RESET}")
            print(f"  {r['question']}")
            print()

    else:
        if args.seed:
            seed_path = Path(args.seed)
        else:
            seeds = sorted(SEED_DIR.glob("seed_*.json"), reverse=True)
            if not seeds:
                print("[ERROR] No seed files found in data/seed/")
                sys.exit(1)
            seed_path = seeds[0]

        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        print(f"\n{BOLD}Generating simulation question for {seed.get('date')}...{RESET}\n")
        question = generate_simulation_question(seed)
        print(f"{CYAN}Question:{RESET}")
        print(f"  {question}")
        print()
