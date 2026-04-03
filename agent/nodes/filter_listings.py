"""LangGraph node for hard-constraint filtering."""

from __future__ import annotations

from agent.services.scoring import filter_hard_constraints
from agent.state import AgentState


def filter_listings_node(state: AgentState) -> AgentState:
    """Apply hard constraints to the loaded listings."""

    filtered = filter_hard_constraints(
        listings=state.get("listings", []),
        hard_constraints=state.get("hard_constraints", {}),
    )
    return {"filtered_listings": filtered}
