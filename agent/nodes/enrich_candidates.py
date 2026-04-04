"""LangGraph node for stage-two Google Maps enrichment and reranking."""

from __future__ import annotations

from agent.services.google_maps import enrich_and_rerank_listings
from agent.state import AgentState


def enrich_candidates_node(state: AgentState) -> AgentState:
    """Enrich shortlisted listings with live neighborhood context."""

    reranked, diagnostics = enrich_and_rerank_listings(
        listings=state.get("shortlisted_listings", []),
        soft_preferences=state.get("soft_preferences", {}),
        hard_constraints=state.get("hard_constraints", {}),
    )
    return {
        "enriched_listings": reranked,
        "scored_listings": reranked,
        "shortlisted_listings": reranked,
        "google_enrichment_diagnostics": diagnostics,
    }
