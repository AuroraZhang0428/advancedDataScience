"""
NestAI Offline Benchmark
========================
Standalone terminal script — do NOT import from app.py or the frontend.

Runs 15 fixed queries through all 3 systems and prints a TravelPlanner-style
comparison table for the final report.

Usage
-----
    # Filter baseline only (no API key needed):
    python benchmark/run_benchmark.py --skip-llm

    # Full 3-way comparison:
    python benchmark/run_benchmark.py --api-key sk-...
    # or:  export OPENAI_API_KEY=sk-... && python benchmark/run_benchmark.py

Systems compared
----------------
    baseline-filter   Regex parsing + price-sort, zero LLM calls
    baseline-llm      Plain GPT-4o-mini chatbot, no structured pipeline
    nestai-agent      Full LangGraph ReAct agent (4-iteration loop)

Metrics (TravelPlanner-style)
-----------------------------
    Delivery Rate           Did the system return ≥1 result without crashing?
    Hard Constraint Micro   Fraction of hard constraints satisfied (avg over queries)
    Hard Constraint Macro   % of queries where ALL hard constraints passed
    Commonsense Micro       Fraction of commonsense checks passed
    Commonsense Macro       % of queries where ALL commonsense checks passed
    Tool-use Failures       # queries where agent crashed on a tool call
    Final Pass Rate ★       % of queries passing EVERY check (headline metric)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ════════════════════════════════════════════════════════════════════════════
# 1.  QUERY SUITE  (15 fixed queries, hardcoded — TA requirement)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Query:
    qid: str
    difficulty: str          # easy | medium | hard
    text: str                # natural-language query sent to the system

    # Hard constraints (ground-truth — used by evaluator, NOT sent to system)
    max_price:    float | None = None
    room_type:    str   | None = None   # exact substring to match room_type field
    min_rating:   float | None = None   # 0.0–5.0 scale
    neighbourhood: str  | None = None   # borough or neighbourhood name
    room_rule:    str   | None = None   # rule that must NOT appear in house_rules


QUERIES: list[Query] = [
    # ── EASY (budget + neighbourhood, ≤2 hard constraints) ─────────────────
    Query("E01", "easy",
          "Find me a private room in Brooklyn under $80 per night.",
          max_price=80, room_type="Private room", neighbourhood="Brooklyn"),

    Query("E02", "easy",
          "I need an entire apartment in Manhattan for under $200 a night.",
          max_price=200, room_type="Entire home/apt", neighbourhood="Manhattan"),

    Query("E03", "easy",
          "Show me affordable listings in Queens, budget $70 per night.",
          max_price=70, neighbourhood="Queens"),

    Query("E04", "easy",
          "Cheap shared room anywhere in the Bronx, max $50/night.",
          max_price=50, room_type="Shared room", neighbourhood="Bronx"),

    Query("E05", "easy",
          "Any listing in Brooklyn under $100 a night.",
          max_price=100, neighbourhood="Brooklyn"),

    # ── MEDIUM (budget + room_type + one extra constraint) ──────────────────
    Query("M01", "medium",
          "Entire apartment in Brooklyn, no smoking, under $150 per night.",
          max_price=150, room_type="Entire home/apt",
          neighbourhood="Brooklyn", room_rule="no smoking"),

    Query("M02", "medium",
          "Private room in Manhattan, rating at least 4.5 stars, max $120/night.",
          max_price=120, room_type="Private room",
          neighbourhood="Manhattan", min_rating=4.5),

    Query("M03", "medium",
          "Entire place in Staten Island, no parties, under $130 a night.",
          max_price=130, room_type="Entire home/apt",
          neighbourhood="Staten Island", room_rule="no parties"),

    Query("M04", "medium",
          "Private room in Queens with a rating above 4.3, max $90/night.",
          max_price=90, room_type="Private room",
          neighbourhood="Queens", min_rating=4.3),

    Query("M05", "medium",
          "Entire apartment in the Bronx under $110, no pets allowed.",
          max_price=110, room_type="Entire home/apt",
          neighbourhood="Bronx", room_rule="no pets"),

    # ── HARD (budget + room_type + rule + rating, all four) ─────────────────
    Query("H01", "hard",
          "Entire place in Williamsburg Brooklyn, max $180/night, "
          "no smoking, rating above 4.5.",
          max_price=180, room_type="Entire home/apt",
          neighbourhood="Brooklyn", room_rule="no smoking", min_rating=4.5),

    Query("H02", "hard",
          "Private room in Manhattan, remote-work friendly, under $120/night, "
          "no pets, rating above 4.4.",
          max_price=120, room_type="Private room",
          neighbourhood="Manhattan", room_rule="no pets", min_rating=4.4),

    Query("H03", "hard",
          "Entire apartment in Brooklyn under $160, no parties, "
          "review score at least 4.6.",
          max_price=160, room_type="Entire home/apt",
          neighbourhood="Brooklyn", room_rule="no parties", min_rating=4.6),

    Query("H04", "hard",
          "Shared room in Queens, max $55/night, no smoking, rating above 4.2.",
          max_price=55, room_type="Shared room",
          neighbourhood="Queens", room_rule="no smoking", min_rating=4.2),

    Query("H05", "hard",
          "Entire home in Staten Island, under $140/night, no children under 10, "
          "rating at least 4.5.",
          max_price=140, room_type="Entire home/apt",
          neighbourhood="Staten Island",
          room_rule="no children", min_rating=4.5),
]


# ════════════════════════════════════════════════════════════════════════════
# 2.  RESULT + EVAL DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    qid: str
    system: str
    delivered: bool
    listings: list[dict[str, Any]]
    steps: int = 0
    tool_use_failure: bool = False   # tool crashed mid-loop
    error: str | None = None


@dataclass
class EvalResult:
    qid: str
    system: str
    difficulty: str
    delivered: bool
    tool_use_failure: bool

    hard: dict[str, bool] = field(default_factory=dict)
    hard_micro: float = 0.0
    hard_macro: bool = False

    cs: dict[str, bool] = field(default_factory=dict)
    cs_micro: float = 0.0
    cs_macro: bool = False

    final_pass: bool = False


# ════════════════════════════════════════════════════════════════════════════
# 3.  CONSTRAINT CHECKERS
# ════════════════════════════════════════════════════════════════════════════

def _check_budget(listings: list[dict], max_price: float) -> bool:
    return all(float(l.get("price") or 0) <= max_price for l in listings)


def _check_room_type(listings: list[dict], room_type: str) -> bool:
    rt = room_type.lower()
    return all(rt in (l.get("room_type") or "").lower() for l in listings)


def _check_min_rating(listings: list[dict], min_rating: float) -> bool:
    return all(float(l.get("review_rating") or 0) >= min_rating for l in listings)


def _check_neighbourhood(listings: list[dict], neighbourhood: str) -> bool:
    nb = neighbourhood.lower()
    for l in listings:
        area = " ".join([
            str(l.get("neighborhood") or ""),
            str(l.get("neighborhood_group") or ""),
        ]).lower()
        if nb not in area:
            return False
    return True


def _check_room_rule(listings: list[dict], rule: str) -> bool:
    """Pass if NONE of the listings have the forbidden rule in their raw house_rules."""
    rule_lc = rule.lower()
    for l in listings:
        raw = l.get("raw") or {}
        house_rules = str(raw.get("house_rules") or "").lower()
        if rule_lc in house_rules:
            return False
    return True


# Commonsense checks
def _cs_no_duplicates(listings: list[dict]) -> bool:
    ids = [str(l.get("id", "")) for l in listings]
    return len(ids) == len(set(ids))


def _cs_within_sandbox(listings: list[dict], dataset_ids: set[str]) -> bool:
    if not dataset_ids:
        return True
    for l in listings:
        lid = str(l.get("id", ""))
        if lid and lid not in dataset_ids:
            return False
    return True


def _cs_diverse_areas(listings: list[dict]) -> bool:
    if len(listings) < 3:
        return True
    areas = set(
        (l.get("neighborhood") or l.get("neighborhood_group") or "").lower()
        for l in listings
    )
    return len(areas) > 1


def _cs_complete_info(listings: list[dict]) -> bool:
    for l in listings:
        if not (l.get("title") or l.get("name")):
            return False
        if not l.get("price"):
            return False
    return True


def _cs_sane_price(listings: list[dict]) -> bool:
    for l in listings:
        p = float(l.get("price") or 0)
        if p <= 0 or p > 5000:
            return False
    return True


# ════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

def evaluate(query: Query, run: RunResult, dataset_ids: set[str]) -> EvalResult:
    ev = EvalResult(
        qid=query.qid,
        system=run.system,
        difficulty=query.difficulty,
        delivered=run.delivered,
        tool_use_failure=run.tool_use_failure,
    )

    if not run.delivered or not run.listings:
        return ev

    ls = run.listings

    # Hard constraints
    hc: dict[str, bool] = {}
    if query.max_price is not None:
        hc["budget"] = _check_budget(ls, query.max_price)
    if query.room_type is not None:
        hc["room_type"] = _check_room_type(ls, query.room_type)
    if query.min_rating is not None:
        hc["min_rating"] = _check_min_rating(ls, query.min_rating)
    if query.neighbourhood is not None:
        hc["neighbourhood"] = _check_neighbourhood(ls, query.neighbourhood)
    if query.room_rule is not None:
        hc["room_rule"] = _check_room_rule(ls, query.room_rule)

    ev.hard = hc
    ev.hard_micro = (sum(hc.values()) / len(hc)) if hc else 1.0
    ev.hard_macro = all(hc.values()) if hc else True

    # Commonsense
    cs: dict[str, bool] = {
        "no_duplicates":  _cs_no_duplicates(ls),
        "within_sandbox": _cs_within_sandbox(ls, dataset_ids),
        "diverse_areas":  _cs_diverse_areas(ls),
        "complete_info":  _cs_complete_info(ls),
        "sane_price":     _cs_sane_price(ls),
    }
    ev.cs = cs
    ev.cs_micro = sum(cs.values()) / len(cs)
    ev.cs_macro = all(cs.values())

    ev.final_pass = ev.hard_macro and ev.cs_macro
    return ev


# ════════════════════════════════════════════════════════════════════════════
# 5.  SYSTEM RUNNERS
# ════════════════════════════════════════════════════════════════════════════

def run_filter(listings: list[dict], q: Query) -> RunResult:
    """Baseline 1 — regex filter + price sort, zero LLM calls."""
    from agent.baselines.filter_search import run_filter_baseline
    try:
        resp = run_filter_baseline(listings, q.text)
        recs = resp.get("recommendations", [])
        return RunResult(q.qid, "baseline-filter", bool(recs), recs, steps=1)
    except Exception as exc:
        return RunResult(q.qid, "baseline-filter", False, [],
                         tool_use_failure=False, error=str(exc))


def run_llm(listings: list[dict], q: Query, api_key: str) -> RunResult:
    """Baseline 2 — plain GPT-4o-mini chatbot, no structured pipeline."""
    from agent.baselines.llm_chatbot import run_llm_chatbot_baseline
    try:
        resp = run_llm_chatbot_baseline(listings, q.text, api_key=api_key)
        recs = resp.get("recommendations", [])
        return RunResult(q.qid, "baseline-llm", bool(recs), recs, steps=1)
    except Exception as exc:
        return RunResult(q.qid, "baseline-llm", False, [],
                         tool_use_failure=False, error=str(exc))


def run_agent(q: Query, dataset_path: str) -> RunResult:
    """Full NestAI ReAct agent — LangGraph 4-iteration loop."""
    from agent.graph import build_graph
    tool_failure = False
    try:
        graph = build_graph()
        state = graph.invoke({
            "user_query": q.text,
            "dataset_path": dataset_path,
            "attempt_count": 0,
            "relaxation_history": [],
            "questions_asked": [],
        })
        recs  = state.get("final_recommendations") or []
        msgs  = state.get("orchestrator_messages") or []
        steps = len([m for m in msgs if m.get("role") == "assistant"])

        # Detect tool-use failures: tool messages that contain error text
        for m in msgs:
            if m.get("role") == "tool":
                content = str(m.get("content", "")).lower()
                if any(w in content for w in ["error", "exception", "traceback", "failed"]):
                    tool_failure = True
                    break

        return RunResult(q.qid, "nestai-agent", bool(recs), recs,
                         steps=steps, tool_use_failure=tool_failure)
    except Exception as exc:
        tb = traceback.format_exc()
        tool_failure = "tool" in tb.lower() or "execute_tool" in tb.lower()
        return RunResult(q.qid, "nestai-agent", False, [],
                         steps=0, tool_use_failure=tool_failure, error=str(exc))


# ════════════════════════════════════════════════════════════════════════════
# 6.  REPORT PRINTER
# ════════════════════════════════════════════════════════════════════════════

def _pct(vals: list[float]) -> str:
    if not vals:
        return "  —"
    return f"{sum(vals)/len(vals)*100:5.1f}%"


def print_report(results: list[EvalResult]) -> None:
    systems = ["baseline-filter", "baseline-llm", "nestai-agent"]
    present = sorted(set(r.system for r in results), key=lambda s: systems.index(s) if s in systems else 99)

    C = 28   # label column width
    S = 16   # system column width
    W = C + S * len(present)

    def hdr(label: str) -> str:
        return f"  {label:<{C-2}}" + "".join(f"{s:>{S}}" for s in present)

    def row(label: str, fn) -> str:
        vals = [fn([r for r in results if r.system == s]) for s in present]
        return f"  {label:<{C-2}}" + "".join(f"{v:>{S}}" for v in vals)

    sep = "═" * W
    print(f"\n{sep}")
    print("  NestAI Offline Benchmark  ——  TravelPlanner-style evaluation")
    print(f"  {len(QUERIES)} fixed queries  ·  {len(present)} system(s)")
    print(sep)
    print(hdr(""))
    print("─" * W)

    def delivery(rs):   return _pct([float(r.delivered) for r in rs])
    def cs_micro(rs):   return _pct([r.cs_micro for r in rs if r.delivered])
    def cs_macro(rs):   return _pct([float(r.cs_macro) for r in rs if r.delivered])
    def hd_micro(rs):   return _pct([r.hard_micro for r in rs if r.delivered])
    def hd_macro(rs):   return _pct([float(r.hard_macro) for r in rs if r.delivered])
    def tooluse(rs):    return f"{sum(r.tool_use_failure for r in rs):>{S-1}}  "
    def final(rs):      return _pct([float(r.final_pass) for r in rs])

    print(row("Delivery Rate",              delivery))
    print(row("Commonsense Pass  (Micro)",  cs_micro))
    print(row("Commonsense Pass  (Macro)",  cs_macro))
    print(row("Hard Constraint   (Micro)",  hd_micro))
    print(row("Hard Constraint   (Macro)",  hd_macro))
    print(row("Tool-use Failures  (#)",     tooluse))
    print(row("Final Pass Rate  ★",         final))

    # ── Difficulty breakdown ──────────────────────────────────────────────
    print()
    print("  Final Pass Rate by difficulty:")
    print("─" * W)
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in results if r.difficulty == diff]
        def fp_diff(rs): return _pct([float(r.final_pass) for r in rs])
        vals = [fp_diff([r for r in sub if r.system == s]) for s in present]
        label = f"  {diff.capitalize():<{C-2}}"
        print(label + "".join(f"{v:>{S}}" for v in vals))

    # ── Per-query detail ──────────────────────────────────────────────────
    print()
    print("  Per-query breakdown  (D=delivered  H=hard✓  C=commonsense✓  F=final✓  T=tool-fail)")
    print("─" * W)
    qids = [q.qid for q in QUERIES]
    for qid in qids:
        q_obj = next(q for q in QUERIES if q.qid == qid)
        row_parts = f"  {qid} ({q_obj.difficulty[0].upper()})  {q_obj.text[:38]:<38}"
        for s in present:
            r = next((x for x in results if x.qid == qid and x.system == s), None)
            if r is None:
                row_parts += f"{'—':>{S}}"
            else:
                flags = (
                    ("D" if r.delivered else "·") +
                    ("H" if r.hard_macro else "·") +
                    ("C" if r.cs_macro   else "·") +
                    ("F" if r.final_pass else "·") +
                    ("T" if r.tool_use_failure else "·")
                )
                row_parts += f"{flags:>{S}}"
        print(row_parts)

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="NestAI offline benchmark runner")
    parser.add_argument("--api-key",   default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--dataset",   default="matched_subset_dataset.csv")
    parser.add_argument("--skip-llm",  action="store_true",
                        help="Skip baseline-llm and nestai-agent (no API key needed)")
    parser.add_argument("--skip-agent", action="store_true",
                        help="Skip nestai-agent only")
    parser.add_argument("--output",    default="benchmark/results.json")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key and not args.skip_llm:
        print("⚠  No OPENAI_API_KEY found — skipping LLM systems (use --skip-llm to suppress this warning).")
        args.skip_llm = True

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    from agent.services.dataset import load_listings
    listings = load_listings(args.dataset)
    dataset_ids = {str(l.get("id", "")) for l in listings}
    print(f"  {len(listings)} listings, {len(dataset_ids)} unique IDs.\n")

    systems_active = ["baseline-filter"]
    if not args.skip_llm:
        systems_active.append("baseline-llm")
    if not args.skip_llm and not args.skip_agent:
        systems_active.append("nestai-agent")

    print(f"Systems : {', '.join(systems_active)}")
    print(f"Queries : {len(QUERIES)}  (Easy×5 / Medium×5 / Hard×5)")
    print("─" * 70)

    all_eval: list[EvalResult] = []
    raw_log:  list[dict]      = []

    for q in QUERIES:
        print(f"\n[{q.qid}] ({q.difficulty.upper()})  {q.text[:65]}")

        for system in systems_active:
            t0 = time.time()

            if system == "baseline-filter":
                run = run_filter(listings, q)
            elif system == "baseline-llm":
                run = run_llm(listings, q, api_key)
            else:
                run = run_agent(q, args.dataset)

            elapsed = round(time.time() - t0, 1)
            ev = evaluate(q, run, dataset_ids)
            all_eval.append(ev)

            status  = "✓" if ev.final_pass else "✗"
            deliver = f"{len(run.listings)} results" if run.delivered else "NOT delivered"
            tf      = " [TOOL-FAIL]" if run.tool_use_failure else ""
            print(
                f"  {status} [{system:<16}]  {deliver:<14}  "
                f"hard={str(ev.hard_macro):<5}  cs={str(ev.cs_macro):<5}  "
                f"final={str(ev.final_pass):<5}  {elapsed}s{tf}"
            )
            if run.error:
                print(f"    ⚠  {run.error[:90]}")

            raw_log.append({
                "qid": q.qid, "system": system, "difficulty": q.difficulty,
                "delivered": run.delivered, "n_listings": len(run.listings),
                "hard_micro": round(ev.hard_micro, 3), "hard_macro": ev.hard_macro,
                "cs_micro": round(ev.cs_micro, 3),     "cs_macro": ev.cs_macro,
                "tool_use_failure": run.tool_use_failure,
                "final_pass": ev.final_pass,
                "steps": run.steps, "elapsed_s": elapsed,
                "hard_detail": ev.hard, "cs_detail": ev.cs,
            })

    # ── Print report ──────────────────────────────────────────────────────
    print_report(all_eval)

    # ── Save JSON ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(raw_log, f, indent=2)
    print(f"Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
