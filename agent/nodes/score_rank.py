"""LangGraph node for deterministic scoring and ranking."""

from __future__ import annotations

from agent.config import DEFAULT_CONFIG
from agent.services.scoring import rank_listings
from agent.state import AgentState


def score_rank_node(state: AgentState) -> AgentState:
    """Score filtered listings and keep a shortlist."""

    ranked = rank_listings(
        listings=state.get("filtered_listings", []),
        soft_preferences=state.get("soft_preferences", {}),
        hard_constraints=state.get("hard_constraints", {}),
        shortlist_size=DEFAULT_CONFIG.shortlist_size,
        weights=DEFAULT_CONFIG.scoring_weights,
    )
    return {
        "scored_listings": ranked,
        "shortlisted_listings": ranked[: DEFAULT_CONFIG.top_k_recommendations],
    }
