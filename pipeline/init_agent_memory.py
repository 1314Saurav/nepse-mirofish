"""
pipeline/init_agent_memory.py

Pre-seeds Zep Cloud memory for each NEPSE agent type with historical context.

Zep Cloud memory is stored at the user level. Since MiroFish creates Zep
users during profile preparation, we inject memories immediately AFTER
a simulation is prepared but BEFORE it is started. If no simulation is
running, we create temporary Zep users to hold the memories.

Usage:
    python pipeline/init_agent_memory.py                    # all types, no sim
    python pipeline/init_agent_memory.py --sim-id SIM_XYZ   # attach to sim
    python pipeline/init_agent_memory.py --type nrb_policy_watcher  # one type only
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT))

BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"

# ── Historical memory corpus ───────────────────────────────────────────────────

AGENT_MEMORIES: dict[str, list[str]] = {
    "institutional_broker": [
        "NEPSE crashed from 3200 to 1800 (2021-2022) when NRB tightened credit policies via a 90% CD ratio ceiling. Banking stocks led the decline by 35%.",
        "Hydropower stocks rallied 80% in 2023-2024 as NEA increased PPA rates and cross-border export deals with India were signed via the Dhalkebar-Muzaffarpur line.",
        "NEPSE uses T+3 settlement and has ±10% daily circuit breakers on individual stocks. Factor this into liquidity risk when placing large orders.",
        "The insurance sector mandatory merger (2022-2023) reduced listed companies but increased individual company size. Merged insurers are now better capitalised.",
        "SEBON's free float policy requires promoters to reduce stakes to 51% over time. This creates periodic secondary market supply pressure.",
        "Average P/E ratio for NEPSE commercial banks historically ranges 15-25x. Above 25x is expensive; below 12x signals value unless earnings are deteriorating.",
        "Bonus shares temporarily increase supply, causing 5-15% price corrections within 2 weeks of ex-date. The correction is usually a buying opportunity.",
    ],
    "retail_momentum": [
        "IPO allotment in Nepal is lottery-based. Applying via multiple family DEMAT accounts increases chances. Most oversubscribed IPOs are 50-300x.",
        "Bonus shares (stock dividends) cause temporary price spikes of 10-20% on announcement, followed by corrections after ex-date.",
        "Facebook groups and Viber communities spread stock tips faster than any formal channel in Nepal. Tips often move small-cap stocks 5-10% within hours.",
        "NEPSE floor price (circuit breaker) is ±10% per day. Stocks hitting upper circuit for 3+ consecutive days attract retail attention as momentum plays.",
        "The NRB relaxed margin lending limits in 2023 — traders can now borrow up to 70% against pledged shares at licensed margin lenders.",
    ],
    "nrb_policy_watcher": [
        "NRB credit-to-deposit (CD) ratio ceiling of 90% has been the main macro constraint on bank lending since 2021. When banks approach 88-90% CD ratio, they slow lending, which tightens market liquidity.",
        "Bank rate is the ceiling of NRB's interest rate corridor. Repo rate is the operational rate for short-term liquidity injection. Reverse repo is the floor.",
        "NRB announces monetary policy twice yearly: mid-July (for full fiscal year) and mid-January (mid-term review). Markets price in expectations 2-3 weeks before.",
        "When NRB cuts repo rate by 25bps, banking stocks typically gain 3-7% within one week as lending margins improve and credit growth expectations rise.",
        "Nepal's INR/NPR peg is fixed at approximately NPR 1.6 per INR. Any deviation signals currency risk and rattles import-heavy sectors (manufacturing, hotels).",
        "NRB foreign exchange reserves cover approximately 10-12 months of imports — a key stability metric. Below 6 months would trigger emergency measures.",
        "BFI (Banks and Financial Institutions) circulars on provisioning requirements directly impact bank earnings. Increased provisioning reduces EPS for 1-2 quarters.",
    ],
    "hydro_analyst": [
        "Nepal's dry season (Nov-May) reduces hydro output by 60-70% vs wet season. Hydropower stocks typically underperform Nov-March and outperform May-September.",
        "Upper Tamakoshi (456MW) and Trishuli-3A are bellwether projects. Their generation reports and dividend history set the tone for the entire sector.",
        "NEA's Power Purchase Agreement (PPA) rate directly determines project profitability. Current PPA rates: wet season ~NPR 4.80/unit, dry season ~NPR 8.40/unit.",
        "Nepal's cross-border power export to India via the Dhalkebar-Muzaffarpur line is limited to 600-900 MW. UPPER, NHPC benefit most from export allocation.",
        "IPPAN (Independent Power Producers Association Nepal) releases generation reports monthly. Significant shortfall vs projections is a negative catalyst.",
        "Seasonal rainfall data from DHM (Dept of Hydrology and Meteorology) in June-July is the most watched indicator for hydro sector summer performance.",
        "Project-level events: Commercial Operation Date (COD) declarations are major positive catalysts. Delays to COD cause 5-15% stock price declines.",
    ],
    "political_risk_analyst": [
        "Nepal has had 10+ Prime Ministers in 15 years (2008-2023). Each government transition causes a 5-15% NEPSE correction due to policy uncertainty.",
        "The annual budget (typically presented in May/June) is the single biggest fiscal event for NEPSE. Infrastructure allocation determines construction and hydro sector sentiment.",
        "The 2022 elections caused NEPSE to fall 15% in 3 months due to uncertainty. Post-election stability rallies historically last 6-12 months if a stable coalition forms.",
        "Pro-China vs pro-India policy shifts affect import costs, manufacturing margins, and cross-border investment flows. Watch for treaty announcements.",
        "Nepal's 2015 earthquake reconstruction created a multi-year infrastructure boom. Similar infrastructure spending pledges are positive catalysts for construction stocks.",
        "SEBON policy changes (free float rules, margin requirements, circuit breaker adjustments) are approved through the Ministry of Finance — track MoF budget speeches.",
        "Major bilateral agreements (India-Nepal power trade, China BRI projects) typically cause 2-5% sector-specific moves on announcement day.",
    ],
    "diaspora_investor": [
        "Nepal receives approximately $9-10 billion USD in annual remittances, constituting roughly 25% of GDP. This creates structural buying support in NEPSE.",
        "NRN (Non-Resident Nepali) investment regulations allow direct DEMAT account ownership since 2021. NRNs can now hold shares without a local proxy.",
        "Gulf Cooperation Council (GCC) economic health directly impacts Nepali worker remittances. Oil price below $60/barrel triggers GCC austerity, reducing Nepal remittances.",
        "Remittance inflows to Nepal historically peak in Q1 (Oct-Dec) around Dashain/Tihar festivals and Q3 (Apr-Jun). NEPSE volume spikes correlate with these peaks.",
        "When NPR depreciates vs USD by 1%, remittance-backed investment into NEPSE increases as the same USD buys more shares. This creates a natural hedge.",
        "Dividend-paying stocks (banks, insurance) are preferred by diaspora investors as they provide regular NPR income to send to family or reinvest.",
        "NEPSE trading is accessible online via brokers' mobile apps since 2021. Diaspora can now trade directly without relying on family, increasing participation.",
    ],
}


def _get_zep_client():
    """Return an authenticated Zep Cloud client, or None if key missing."""
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    api_key = os.getenv("ZEP_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None
    try:
        from zep_cloud.client import Zep
        return Zep(api_key=api_key)
    except Exception as exc:
        print(f"  {YELLOW}[WARN]{RESET} Could not connect to Zep: {exc}")
        return None


def _get_sim_profiles(sim_id: str) -> list[dict]:
    """Fetch OASIS agent profiles from a running MiroFish simulation."""
    import requests
    base = "http://localhost:5001"
    profiles = []
    for platform in ["reddit", "twitter"]:
        try:
            r = requests.get(f"{base}/simulation/{sim_id}/profiles",
                             params={"platform": platform}, timeout=15)
            if r.ok:
                d = r.json().get("data") or []
                profiles.extend(d if isinstance(d, list) else [])
        except Exception:
            pass
    return profiles


def init_memories_for_type(zep, agent_type: str,
                            user_ids: Optional[list[str]] = None,
                            dry_run: bool = False) -> int:
    """
    Initialise Zep memory for a given agent type.

    If user_ids is None, creates temporary Zep users named
    {type}_seed_0, {type}_seed_1, ... (one per memory entry).

    Returns the number of memory entries written.
    """
    memories = AGENT_MEMORIES.get(agent_type, [])
    if not memories:
        print(f"  {YELLOW}No memories defined for type: {agent_type}{RESET}")
        return 0

    written = 0
    for i, memory_text in enumerate(memories):
        uid = (user_ids[i] if user_ids and i < len(user_ids)
               else f"nepse_{agent_type}_seed_{i}")

        if dry_run:
            print(f"    [DRY] would write memory to user {uid}: {memory_text[:60]}...")
            written += 1
            continue

        try:
            # Ensure Zep user exists
            try:
                zep.user.get(uid)
            except Exception:
                zep.user.add(user_id=uid,
                             first_name=agent_type.replace("_", " ").title(),
                             metadata={"agent_type": agent_type,
                                       "source":     "nepse_memory_init"})

            # Add memory as a message session
            from zep_cloud.types import Message
            session_id = f"{uid}_history_{i}"
            try:
                zep.memory.add_session(session_id=session_id, user_id=uid)
            except Exception:
                pass  # session may already exist

            zep.memory.add(
                session_id=session_id,
                messages=[
                    Message(
                        role="system",
                        role_type="system",
                        content=f"Historical knowledge for NEPSE investor type '{agent_type}': {memory_text}",
                    )
                ],
            )
            written += 1
            time.sleep(0.2)   # rate limit
        except Exception as exc:
            print(f"    {YELLOW}[WARN]{RESET} Failed to write memory {i} for {uid}: {exc}")

    return written


def init_all_agent_memories(sim_id: Optional[str] = None,
                             filter_type: Optional[str] = None,
                             dry_run: bool = False) -> dict:
    """
    Initialise memory for all 6 agent types.

    If sim_id is provided, fetches OASIS user IDs from the simulation
    and maps memories to real agent users.

    Returns summary dict.
    """
    zep = _get_zep_client()
    if zep is None and not dry_run:
        return {
            "status": "skipped",
            "reason": "ZEP_API_KEY not set or invalid — set it in .env to enable agent memory",
        }

    types = [filter_type] if filter_type else list(AGENT_MEMORIES.keys())
    sim_profiles: list[dict] = []
    if sim_id:
        print(f"  Fetching profiles from simulation {sim_id}...")
        sim_profiles = _get_sim_profiles(sim_id)
        print(f"  Found {len(sim_profiles)} agent profiles")

    summary = {
        "status":       "ok",
        "sim_id":       sim_id,
        "dry_run":      dry_run,
        "types_seeded": [],
        "total_entries": 0,
    }

    for agent_type in types:
        print(f"  Seeding memories for: {CYAN}{agent_type}{RESET}...")

        # Extract user_ids for this agent type from simulation profiles
        user_ids = None
        if sim_profiles:
            type_profiles = [
                p for p in sim_profiles
                if agent_type.replace("_", " ").lower() in
                   (p.get("persona", "") + p.get("bio", "")).lower()
            ]
            if type_profiles:
                user_ids = [p.get("user_id", p.get("username", "")) for p in type_profiles]

        n = init_memories_for_type(
            zep       = zep,
            agent_type= agent_type,
            user_ids  = user_ids,
            dry_run   = dry_run,
        )
        summary["types_seeded"].append(agent_type)
        summary["total_entries"] += n
        print(f"    {GREEN}+{n} entries{RESET}")

    total_agents  = 1000   # from config
    total_entries = summary["total_entries"]
    print(f"\n  {GREEN}Memory initialised:{RESET} {len(types)} types, {total_entries} memory entries")
    if not dry_run and zep:
        print(f"  (memories stored in Zep Cloud as user-level historical facts)")
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    ap = argparse.ArgumentParser(description="Initialise NEPSE agent memories in Zep Cloud")
    ap.add_argument("--sim-id",   default=None, help="Attach memories to simulation profiles")
    ap.add_argument("--type",     default=None, help="Only seed one agent type")
    ap.add_argument("--dry-run",  action="store_true", help="Preview without writing to Zep")
    import argparse
    args = ap.parse_args()

    print(f"\n{BOLD}NEPSE Agent Memory Initialisation{RESET}")
    print("=" * 50)

    result = init_all_agent_memories(
        sim_id      = args.sim_id,
        filter_type = args.type,
        dry_run     = args.dry_run,
    )

    print()
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result.get("status") == "skipped":
        print(f"\n  {YELLOW}Action required:{RESET} Add your ZEP_API_KEY to .env")
        print("  Get a free Zep Cloud key at: https://app.getzep.com/")
