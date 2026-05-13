"""LangGraph node for dataset loading."""

from __future__ import annotations

from agent.config import DEFAULT_CONFIG
from agent.services.dataset import load_listings
from agent.state import AgentState


def load_data_node(state: AgentState) -> AgentState:
    """Load the listing dataset and hard-reset all search-specific state.

    Every field that must NOT carry over from a previous query is explicitly
    set here.  This ensures a fresh graph.invoke() is always a clean slate,
    regardless of what LangGraph left in memory from a prior invocation.
    """

    dataset_path = state.get("dataset_path") or str(DEFAULT_CONFIG.dataset_path)
    listings = load_listings(dataset_path)
    return {
        # ── Data ───────────────────────────────────────────────────────────
        "dataset_path": str(dataset_path),
        "listings": listings,
        # ── Preferences (overwritten by parse_preferences_node next) ───────
        "raw_preferences": {},
        "hard_constraints": {},
        "soft_preferences": {},
        "relaxable_constraints": {},
        # ── Search intermediates ────────────────────────────────────────────
        "filtered_listings": [],
        "scored_listings": [],
        "shortlisted_listings": [],
        "enriched_listings": [],
        # ── Results ────────────────────────────────────────────────────────
        "final_recommendations": [],
        "final_explanations": [],
        # ── Agent bookkeeping ───────────────────────────────────────────────
        "relaxation_history": [],
        "questions_asked": [],
        "need_user_input": False,
        "user_question": None,
        "question_key": None,
        "question_proposed_value": None,
        "sufficient_results": False,
        "results_diagnostics": {},
        "google_enrichment_diagnostics": {},
        "orchestrator_messages": [],
    }
