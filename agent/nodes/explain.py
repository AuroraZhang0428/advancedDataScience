"""LangGraph node for final recommendation explanations."""

from __future__ import annotations

from agent.config import DEFAULT_CONFIG
from agent.services.explanation import generate_final_output
from agent.state import AgentState


def explain_node(state: AgentState) -> AgentState:
    """Finalize top recommendations and human-readable explanations."""

    recommendations, explanations = generate_final_output(
        scored_listings=state.get("scored_listings", []),
        hard_constraints=state.get("hard_constraints", {}),
        soft_preferences=state.get("soft_preferences", {}),
        relaxation_history=state.get("relaxation_history", []),
        top_k=DEFAULT_CONFIG.top_k_recommendations,
    )
    return {
        "final_recommendations": recommendations,
        "final_explanations": explanations,
    }
