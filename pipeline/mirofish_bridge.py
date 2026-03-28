"""
pipeline/mirofish_bridge.py

Bridge between nepse-mirofish data pipeline and MiroFish simulation engine.

Workflow (mirrors MiroFish's 5-step API):
  1. POST /graph/ontology/generate  — upload seed doc + get project_id
  2. POST /graph/build              — build Zep knowledge graph (async)
  3. POST /simulation/create        — create simulation from graph
  4. POST /simulation/prepare       — generate OASIS agent profiles (async)
  5. POST /simulation/start         — run parallel simulation
  6. Poll /simulation/<id>/run-status until done
  7. GET  /simulation/<id>/actions  — collect agent actions for signal extraction

Usage:
    from pipeline.mirofish_bridge import MiroFishBridge, run_simulation_for_seed

    result = run_simulation_for_seed("data/seed/seed_2026-03-26.json")
    # result contains: simulation_id, actions, timeline, summary
"""

from __future__ import annotations

import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ── Config ────────────────────────────────────────────────────────────────────

_ROOT     = Path(__file__).resolve().parent.parent
_PROC_DIR = _ROOT / "data" / "processed"
_PROC_DIR.mkdir(parents=True, exist_ok=True)

MIROFISH_BASE_URL  = "http://localhost:5001/api"
POLL_INTERVAL_SEC  = 10
SIM_TIMEOUT_SEC    = 20 * 60   # 20 minutes
MAX_ROUNDS_DEFAULT = 12        # 12 rounds × 6h = 3 trading days of simulated time


# ── Seed → MiroFish document ───────────────────────────────────────────────────

def seed_to_document(seed: dict) -> str:
    """
    Convert a structured seed JSON into a rich markdown document
    suitable for MiroFish's ontology generation step.

    MiroFish ingests natural-language documents — this function
    is the translation layer between our structured data and
    MiroFish's document-based input.
    """
    date    = seed.get("date", "Unknown")
    ms      = seed.get("market_summary", {})
    mc      = seed.get("macro_context", {})
    news    = seed.get("news_articles", [])
    ipos    = seed.get("ipo_events", [])
    pol_ctx = seed.get("political_context", "")
    brief   = seed.get("brief_text", "")
    sim_q   = seed.get("simulation_question",
                        "How might NEPSE move in the next 3 trading days?")

    lines = [
        f"# NEPSE Market Intelligence Report — {date}",
        "",
        "## Executive Summary",
        brief or f"Market data for {date}. See sections below for details.",
        "",
        "## Simulation Objective",
        sim_q,
        "",
        "## 1. Market Performance",
    ]

    if ms:
        idx    = ms.get("nepse_index", "N/A")
        pct    = ms.get("nepse_pct_change", "N/A")
        turn   = ms.get("total_turnover_npr")
        turn_s = f"NPR {turn:,.0f}" if isinstance(turn, (int, float)) else "N/A"
        lines += [
            f"- NEPSE Index: **{idx}** ({pct:+.2f}% change)" if isinstance(pct, float) else f"- NEPSE Index: {idx} (change: {pct}%)",
            f"- Total Turnover: {turn_s}",
            f"- Scrips Traded: {ms.get('scrips_traded', 'N/A')}",
            f"- Total Transactions: {ms.get('total_transactions', 'N/A')}",
        ]
        gainers = ms.get("top_gainers", [])
        if gainers:
            g_str = ", ".join(f"{g['symbol']} (+{g.get('pct_change',0):.1f}%)"
                               for g in gainers[:5] if isinstance(g, dict))
            lines.append(f"- **Top Gainers**: {g_str}")
        losers = ms.get("top_losers", [])
        if losers:
            l_str = ", ".join(f"{l['symbol']} ({l.get('pct_change',0):+.1f}%)"
                               for l in losers[:5] if isinstance(l, dict))
            lines.append(f"- **Top Losers**: {l_str}")
    else:
        lines.append("- Market data unavailable for this session.")

    lines += [
        "",
        "## 2. NRB Macro Context",
        f"- Bank Rate: {mc.get('bank_rate', 'N/A')}%",
        f"- Repo Rate: {mc.get('repo_rate', 'N/A')}%",
        f"- Reverse Repo: {mc.get('reverse_repo_rate', 'N/A')}%",
        f"- CRR: {mc.get('crr_pct', 'N/A')}%  |  SLR: {mc.get('slr_pct', 'N/A')}%",
        f"- Credit/Deposit Ratio: {mc.get('credit_deposit_ratio', 'N/A')}%  (as of {mc.get('cd_ratio_as_of', 'N/A')})",
        f"- USD/NPR: {mc.get('usd_npr', 'N/A')}  |  EUR/NPR: {mc.get('eur_npr', 'N/A')}",
        f"- INR/NPR: {mc.get('inr_npr', 'N/A')}  |  CNY/NPR: {mc.get('cny_npr', 'N/A')}",
        "",
        "## 3. IPO and Corporate Events",
    ]

    if ipos:
        for ev in ipos:
            lines.append(f"- {ev}")
    else:
        lines.append("- No active IPO or bonus share events today.")

    lines += ["", "## 4. News and Market Sentiment"]

    for art in news[:20]:
        title = art.get("title", "")
        cat   = art.get("category", "general_market")
        src   = art.get("source", "")
        body  = (art.get("body_excerpt", "") or "")[:300]
        lines += [
            f"### [{cat.upper()}] {title}  _(Source: {src})_",
            body,
            "",
        ]

    if pol_ctx:
        lines += ["## 5. Political Context", pol_ctx, ""]

    lines += [
        "## 6. Investor Archetypes in This Market",
        "This simulation involves six distinct investor groups:",
        "- **Institutional Brokers** (200): Data-driven, risk-averse, follow NRB circulars and EPS.",
        "- **Retail Momentum Traders** (200): FOMO-driven, chase trending stocks and IPOs.",
        "- **NRB Policy Watchers** (150): Macro-focused, monitor repo rate and CD ratio obsessively.",
        "- **Hydropower Analysts** (150): Track PPA rates, monsoon, NEA power export deals.",
        "- **Political Risk Analysts** (150): Watch coalition stability, budget announcements.",
        "- **Diaspora / NRN Investors** (150): USD-income mindset, influenced by remittance flows.",
        "",
        "## 7. Historical Context",
        "- NEPSE crashed from 3,200 to 1,800 (2021-2022) when NRB tightened credit.",
        "- Hydropower stocks rallied 80% in 2023-2024 following NEA PPA rate increases.",
        "- IPO allotment in Nepal is lottery-based, driving speculative retail participation.",
        "- Nepal receives ~$10B USD/year in remittances (~25% of GDP), creating a structural bid.",
        "- NEPSE uses T+3 settlement with ±10% daily circuit breakers.",
        "- Government instability (5 PMs in 5 years) historically triggers 5-15% corrections.",
    ]

    return "\n".join(lines)


# ── MiroFish API client ────────────────────────────────────────────────────────

class MiroFishBridge:
    """
    Programmatic client for the MiroFish backend API.
    All calls are synchronous; async operations are polled until completion.
    """

    def __init__(self, base_url: str = MIROFISH_BASE_URL):
        self.base  = base_url.rstrip("/")
        self.sess  = requests.Session()
        # NOTE: do NOT set Content-Type globally — multipart file uploads
        # need requests to set their own boundary-based content type.

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get(self, path: str, **kwargs) -> dict:
        r = self.sess.get(f"{self.base}{path}", **kwargs)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json_body: Optional[dict] = None,
              files=None, data=None, **kwargs) -> dict:
        if files:
            # multipart/form-data — let requests set the boundary automatically
            r = self.sess.post(f"{self.base}{path}", files=files,
                               data=data, **kwargs)
        else:
            # Pure JSON — set content-type explicitly
            headers = kwargs.pop("headers", {})
            headers["Content-Type"] = "application/json"
            r = self.sess.post(f"{self.base}{path}", json=json_body,
                               headers=headers, **kwargs)
        r.raise_for_status()
        return r.json()

    def _poll_task(self, task_id: str, label: str,
                   timeout: int = SIM_TIMEOUT_SEC) -> dict:
        """Poll /graph/task/<task_id> until status is completed or failed."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._get(f"/graph/task/{task_id}")
            status = (resp.get("data") or resp).get("status", "")
            stage  = (resp.get("data") or resp).get("stage", "")
            pct    = (resp.get("data") or resp).get("progress",
                     (resp.get("data") or resp).get("progress_percent", 0))
            print(f"    [{label}] status={status} stage={stage} progress={pct}%",
                  flush=True)
            if status in ("completed", "failed", "error"):
                return resp
            time.sleep(POLL_INTERVAL_SEC)
        raise TimeoutError(f"Task {task_id} ({label}) timed out after {timeout}s")

    def _poll_sim_prepare(self, task_id: str,
                          timeout: int = SIM_TIMEOUT_SEC) -> dict:
        """Poll /simulation/prepare/status until preparation is done."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._post("/simulation/prepare/status",
                              {"task_id": task_id})
            d      = resp.get("data") or resp
            status = d.get("status", "")
            stage  = d.get("stage", "")
            print(f"    [prepare] status={status} stage={stage}", flush=True)
            if status in ("completed", "failed", "error"):
                return resp
            time.sleep(POLL_INTERVAL_SEC)
        raise TimeoutError(f"Simulation prepare timed out after {timeout}s")

    def _poll_sim_run(self, sim_id: str,
                      timeout: int = SIM_TIMEOUT_SEC) -> dict:
        """Poll /simulation/<id>/run-status until rounds complete."""
        deadline = time.time() + timeout
        last_round = -1
        while time.time() < deadline:
            resp     = self._get(f"/simulation/{sim_id}/run-status")
            d        = resp.get("data") or resp
            cur      = d.get("current_round", 0)
            total    = d.get("total_rounds", 0)
            pct      = d.get("progress_percent", 0)
            complete = d.get("is_complete", False)
            if cur != last_round:
                print(f"    [sim run] round {cur}/{total} ({pct:.0f}%)",
                      flush=True)
                last_round = cur
            if complete or (total and cur >= total):
                return resp
            time.sleep(POLL_INTERVAL_SEC)
        raise TimeoutError(f"Simulation {sim_id} run timed out after {timeout}s")

    # ── Step 1: Ontology generation ────────────────────────────────────────────

    def generate_ontology(self, doc_text: str, sim_requirement: str,
                          project_name: str = "NEPSE Daily Simulation") -> dict:
        """
        Upload the seed document and generate an entity/relationship ontology.
        Returns: {project_id, ontology, ...}
        """
        doc_bytes = doc_text.encode("utf-8")
        files = {
            "files": ("nepse_seed.md", doc_bytes, "text/markdown"),
        }
        data = {
            "simulation_requirement": sim_requirement,
            "project_name":           project_name,
        }
        resp = self._post("/graph/ontology/generate", files=files, data=data)
        return resp.get("data", resp)

    # ── Step 2: Graph build ────────────────────────────────────────────────────

    def build_graph(self, project_id: str,
                    graph_name: str = "NEPSE Knowledge Graph") -> dict:
        """
        Kick off async graph construction from the uploaded documents.
        Polls until complete.
        """
        resp    = self._post("/graph/build",
                             {"project_id": project_id,
                              "graph_name": graph_name})
        d       = resp.get("data", resp)
        task_id = d.get("task_id")
        if not task_id:
            raise RuntimeError(f"Graph build returned no task_id: {resp}")
        print(f"    Graph build started (task={task_id})...", flush=True)
        result = self._poll_task(task_id, "graph-build")
        return result.get("data", result)

    # ── Step 3: Create simulation ──────────────────────────────────────────────

    def create_simulation(self, project_id: str,
                          graph_id: Optional[str] = None) -> dict:
        """
        Create a new simulation object linked to the project/graph.
        Returns: {simulation_id, status, ...}
        """
        body = {"project_id": project_id, "enable_twitter": True,
                "enable_reddit": True}
        if graph_id:
            body["graph_id"] = graph_id
        resp = self._post("/simulation/create", body)
        return resp.get("data", resp)

    # ── Step 4: Prepare profiles ───────────────────────────────────────────────

    def prepare_simulation(self, sim_id: str,
                           use_llm: bool = True,
                           parallel_count: int = 10) -> dict:
        """
        Generate OASIS agent profiles from Zep entities.
        Polls until preparation completes.
        """
        resp    = self._post("/simulation/prepare",
                             {"simulation_id":       sim_id,
                              "use_llm_for_profiles": use_llm,
                              "parallel_profile_count": parallel_count})
        d       = resp.get("data", resp)
        task_id = d.get("task_id")
        if not task_id:
            raise RuntimeError(f"Simulation prepare returned no task_id: {resp}")
        print(f"    Profile preparation started (task={task_id})...", flush=True)
        result = self._poll_sim_prepare(task_id)
        return result.get("data", result)

    # ── Step 5: Start simulation ───────────────────────────────────────────────

    def start_simulation(self, sim_id: str,
                         platform: str = "parallel",
                         max_rounds: int = MAX_ROUNDS_DEFAULT,
                         enable_graph_memory: bool = True) -> dict:
        """
        Launch the actual OASIS social simulation.
        platform: "twitter" | "reddit" | "parallel"
        """
        resp = self._post("/simulation/start",
                          {"simulation_id":              sim_id,
                           "platform":                   platform,
                           "max_rounds":                 max_rounds,
                           "enable_graph_memory_update": enable_graph_memory})
        return resp.get("data", resp)

    # ── Step 6: Poll run status ────────────────────────────────────────────────

    def wait_for_completion(self, sim_id: str) -> dict:
        return self._poll_sim_run(sim_id)

    # ── Step 7: Collect results ────────────────────────────────────────────────

    def get_actions(self, sim_id: str, limit: int = 2000) -> list:
        resp = self._get(f"/simulation/{sim_id}/actions",
                         params={"limit": limit, "offset": 0})
        d = resp.get("data", resp)
        return d if isinstance(d, list) else d.get("actions", [])

    def get_timeline(self, sim_id: str) -> list:
        resp = self._get(f"/simulation/{sim_id}/timeline")
        d = resp.get("data", resp)
        return d if isinstance(d, list) else d.get("rounds", [])

    def get_agent_stats(self, sim_id: str) -> dict:
        resp = self._get(f"/simulation/{sim_id}/agent-stats")
        return resp.get("data", resp)

    def get_sim_config(self, sim_id: str) -> dict:
        resp = self._get(f"/simulation/{sim_id}/config")
        return resp.get("data", resp)

    # ── Connectivity check ─────────────────────────────────────────────────────

    def is_alive(self) -> bool:
        try:
            r = requests.get("http://localhost:5001/health", timeout=5)
            return r.status_code < 500
        except Exception:
            return False


# ── Master function ────────────────────────────────────────────────────────────

def run_simulation_for_seed(seed_path: str | Path,
                             max_rounds: int = MAX_ROUNDS_DEFAULT,
                             base_url:   str = MIROFISH_BASE_URL) -> dict:
    """
    End-to-end: load a seed file → run MiroFish simulation → return results.

    Returns a dict with keys:
      simulation_id, project_id, graph_id, actions, timeline,
      agent_stats, sim_config, output_path
    """
    seed_path = Path(seed_path)
    seed      = json.loads(seed_path.read_text(encoding="utf-8"))
    date_str  = seed.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    bridge = MiroFishBridge(base_url)

    # ── Guard: is MiroFish running? ────────────────────────────────────────────
    if not bridge.is_alive():
        raise ConnectionError(
            "MiroFish backend is not reachable at "
            f"{base_url}.\n"
            "Start it with:\n"
            "  cd mirofish && npm run backend"
        )

    print(f"\n[MiroFish] Starting simulation for seed: {seed_path.name}", flush=True)
    t_start = time.time()

    # ── 1. Document + ontology ─────────────────────────────────────────────────
    print("[1/6] Generating ontology from seed document...", flush=True)
    doc_text  = seed_to_document(seed)
    sim_req   = seed.get("simulation_question",
                          "How might NEPSE move in the next 3 trading days?")
    proj_name = f"NEPSE-{date_str}"
    onto      = bridge.generate_ontology(doc_text, sim_req, proj_name)
    project_id = onto.get("project_id")
    if not project_id:
        raise RuntimeError(f"Ontology generation failed: {onto}")
    print(f"    project_id = {project_id}", flush=True)

    # ── 2. Graph build ─────────────────────────────────────────────────────────
    print("[2/6] Building Zep knowledge graph...", flush=True)
    graph_data = bridge.build_graph(project_id, f"NEPSE-Graph-{date_str}")
    graph_id   = graph_data.get("graph_id")
    print(f"    graph_id = {graph_id}", flush=True)

    # ── 3. Create simulation ───────────────────────────────────────────────────
    print("[3/6] Creating simulation object...", flush=True)
    sim_data = bridge.create_simulation(project_id, graph_id)
    sim_id   = sim_data.get("simulation_id") or sim_data.get("id")
    if not sim_id:
        raise RuntimeError(f"Simulation create failed: {sim_data}")
    print(f"    simulation_id = {sim_id}", flush=True)

    # ── 4. Prepare profiles ────────────────────────────────────────────────────
    print("[4/6] Generating OASIS agent profiles...", flush=True)
    bridge.prepare_simulation(sim_id, use_llm=True, parallel_count=10)

    # ── 5. Start simulation ────────────────────────────────────────────────────
    print(f"[5/6] Starting parallel simulation ({max_rounds} rounds)...", flush=True)
    bridge.start_simulation(sim_id, platform="parallel", max_rounds=max_rounds)

    # ── 6. Wait for completion ─────────────────────────────────────────────────
    print("[6/6] Simulation running — polling for completion...", flush=True)
    bridge.wait_for_completion(sim_id)

    # ── Collect results ────────────────────────────────────────────────────────
    print("Collecting simulation results...", flush=True)
    actions     = bridge.get_actions(sim_id, limit=5000)
    timeline    = bridge.get_timeline(sim_id)
    agent_stats = bridge.get_agent_stats(sim_id)
    sim_config  = bridge.get_sim_config(sim_id)

    elapsed = time.time() - t_start

    result = {
        "simulation_id": sim_id,
        "project_id":    project_id,
        "graph_id":      graph_id,
        "date":          date_str,
        "elapsed_sec":   round(elapsed),
        "actions":       actions,
        "timeline":      timeline,
        "agent_stats":   agent_stats,
        "sim_config":    sim_config,
    }

    # Save raw output
    out_path = _PROC_DIR / f"simulation_{date_str}.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    result["output_path"] = str(out_path)
    print(f"Simulation complete in {elapsed:.0f}s. Saved -> {out_path}", flush=True)
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    sys.stdout = __import__("io").TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    ap = argparse.ArgumentParser(description="Run MiroFish simulation for a seed file")
    ap.add_argument("--seed",   default=None,
                    help="Path to seed JSON (default: latest data/seed/seed_*.json)")
    ap.add_argument("--rounds", type=int, default=MAX_ROUNDS_DEFAULT)
    ap.add_argument("--url",    default=MIROFISH_BASE_URL)
    args = ap.parse_args()

    if args.seed:
        seed_file = Path(args.seed)
    else:
        seeds = sorted((_ROOT / "data" / "seed").glob("seed_*.json"), reverse=True)
        if not seeds:
            print("[ERROR] No seed files found in data/seed/")
            sys.exit(1)
        seed_file = seeds[0]

    print(f"Using seed: {seed_file}")
    try:
        result = run_simulation_for_seed(seed_file, args.rounds, args.url)
        print(f"\nDone. Actions collected: {len(result['actions'])}")
        print(f"Output: {result['output_path']}")
    except ConnectionError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        raise
