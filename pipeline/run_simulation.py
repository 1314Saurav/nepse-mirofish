"""
pipeline/run_simulation.py

High-level simulation runner that orchestrates:
  1. Load today's seed file
  2. Load agent config and build simulation_requirement
  3. Submit to MiroFish and stream round-by-round output
  4. Extract and save the simulation transcript

Usage:
    python pipeline/run_simulation.py
    python pipeline/run_simulation.py --seed data/seed/seed_2026-03-26.json
    python pipeline/run_simulation.py --rounds 6 --no-stream
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pipeline.mirofish_bridge import MiroFishBridge, seed_to_document, MIROFISH_BASE_URL
from pipeline.load_agents import load_agent_config, build_simulation_requirement

PROC_DIR  = _ROOT / "data" / "processed"
SEED_DIR  = _ROOT / "data" / "seed"
SIM_DIR   = PROC_DIR / "simulations"
SIM_DIR.mkdir(parents=True, exist_ok=True)

BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"


def find_latest_seed() -> Path:
    seeds = sorted(SEED_DIR.glob("seed_*.json"), reverse=True)
    if not seeds:
        raise FileNotFoundError("No seed files found in data/seed/")
    return seeds[0]


def stream_simulation_progress(bridge: MiroFishBridge, sim_id: str,
                                max_rounds: int, timeout: int = 1200) -> None:
    """Stream round-by-round output to terminal as simulation runs."""
    deadline   = time.time() + timeout
    last_round = -1
    last_tweet = 0
    last_post  = 0

    while time.time() < deadline:
        try:
            status = bridge._get(f"/simulation/{sim_id}/run-status")
            d      = (status.get("data") or status)
            cur    = d.get("current_round", 0)
            total  = d.get("total_rounds", max_rounds)
            pct    = d.get("progress_percent", 0)
            done   = d.get("is_complete", False)

            twitter_acts = d.get("twitter_action_count", 0)
            reddit_acts  = d.get("reddit_action_count", 0)

            if cur != last_round:
                bar_len  = 30
                filled   = int(bar_len * pct / 100) if pct else 0
                bar      = "#" * filled + "-" * (bar_len - filled)
                new_tw   = twitter_acts - last_tweet
                new_rd   = reddit_acts  - last_post
                print(f"\r  Round {cur:>2}/{total} [{bar}] {pct:>3.0f}%  "
                      f"| Twitter: +{new_tw} | Reddit: +{new_rd}    ", flush=True)
                last_round = cur
                last_tweet = twitter_acts
                last_post  = reddit_acts

            if done or (total and cur >= total):
                print(f"\n  {GREEN}Simulation complete!{RESET}", flush=True)
                break

        except Exception as exc:
            print(f"\r  [stream error: {exc}]  ", flush=True)

        time.sleep(10)


def run_simulation(seed_path: Path, max_rounds: int = 12,
                   stream: bool = True) -> dict:
    """
    Full simulation run: load seed → submit to MiroFish → stream → collect results.
    Returns the complete simulation result dict.
    """
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    date_str = seed.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    print(f"\n{BOLD}NEPSE MiroFish Simulation{RESET}")
    print("=" * 60)
    print(f"  Seed date     : {date_str}")
    print(f"  Seed file     : {seed_path.name}")
    print(f"  Max rounds    : {max_rounds}")
    print(f"  MiroFish URL  : {MIROFISH_BASE_URL}")
    print("=" * 60)

    bridge = MiroFishBridge(MIROFISH_BASE_URL)

    if not bridge.is_alive():
        print(f"\n{RED}[ERROR]{RESET} MiroFish backend is not running at {MIROFISH_BASE_URL}")
        print("Start it with:  cd mirofish && npm run backend")
        sys.exit(1)

    # Load agent config and augment simulation_requirement
    agent_cfg  = load_agent_config()
    agent_req  = build_simulation_requirement(agent_cfg)
    doc_text   = seed_to_document(seed)
    full_req   = seed.get("simulation_question",
                           "How will NEPSE move in the next 3 trading days?")

    print(f"\n{CYAN}Step 1/6{RESET} Generating ontology from seed + agent config...")
    onto = bridge.generate_ontology(
        doc_text=doc_text + "\n\n---\n\n" + agent_req,
        sim_requirement=full_req,
        project_name=f"NEPSE-{date_str}",
    )
    project_id = onto.get("project_id")
    print(f"  project_id = {project_id}")

    print(f"\n{CYAN}Step 2/6{RESET} Building Zep knowledge graph...")
    graph_data = bridge.build_graph(project_id, f"NEPSE-Graph-{date_str}")
    graph_id   = graph_data.get("graph_id")
    print(f"  graph_id = {graph_id}")

    print(f"\n{CYAN}Step 3/6{RESET} Creating simulation...")
    sim_data = bridge.create_simulation(project_id, graph_id)
    sim_id   = sim_data.get("simulation_id") or sim_data.get("id")
    print(f"  simulation_id = {sim_id}")

    print(f"\n{CYAN}Step 4/6{RESET} Generating OASIS agent profiles...")
    bridge.prepare_simulation(sim_id, use_llm=True, parallel_count=10)
    print(f"  Profiles ready.")

    print(f"\n{CYAN}Step 5/6{RESET} Starting parallel simulation ({max_rounds} rounds)...")
    bridge.start_simulation(sim_id, platform="parallel", max_rounds=max_rounds)

    print(f"\n{CYAN}Step 6/6{RESET} Streaming progress...")
    if stream:
        stream_simulation_progress(bridge, sim_id, max_rounds)
    else:
        bridge.wait_for_completion(sim_id)

    print(f"\n{CYAN}Collecting results...{RESET}")
    actions     = bridge.get_actions(sim_id, limit=5000)
    timeline    = bridge.get_timeline(sim_id)
    agent_stats = bridge.get_agent_stats(sim_id)
    sim_config  = bridge.get_sim_config(sim_id)

    result = {
        "simulation_id": sim_id,
        "project_id":    project_id,
        "graph_id":      graph_id,
        "date":          date_str,
        "seed_file":     str(seed_path),
        "max_rounds":    max_rounds,
        "ran_at":        datetime.now(timezone.utc).isoformat(),
        "actions":       actions,
        "timeline":      timeline,
        "agent_stats":   agent_stats,
        "sim_config":    sim_config,
    }

    # Save transcript
    transcript_path = SIM_DIR / f"sim_transcript_{date_str}.json"
    transcript_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    result["transcript_path"] = str(transcript_path)

    print(f"\n{GREEN}Simulation transcript saved -> {transcript_path}{RESET}")
    print(f"  Actions collected : {len(actions)}")
    print(f"  Timeline rounds   : {len(timeline)}")
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    ap = argparse.ArgumentParser(description="Run NEPSE MiroFish simulation")
    ap.add_argument("--seed",      default=None,
                    help="Path to seed JSON (default: latest)")
    ap.add_argument("--rounds",    type=int, default=12,
                    help="Number of simulation rounds (default: 12)")
    ap.add_argument("--no-stream", action="store_true",
                    help="Disable streaming output (just poll silently)")
    args = ap.parse_args()

    seed_path = Path(args.seed) if args.seed else find_latest_seed()
    result = run_simulation(seed_path, args.rounds, stream=not args.no_stream)

    print(f"\n{BOLD}Done.{RESET} simulation_id = {result['simulation_id']}")
    print(f"Transcript: {result['transcript_path']}")
