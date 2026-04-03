"""LangGraph node for dataset loading."""

from __future__ import annotations

from agent.config import DEFAULT_CONFIG
from agent.services.dataset import load_listings
from agent.state import AgentState


def load_data_node(state: AgentState) -> AgentState:
    """Load the listing dataset and initialize tracking fields."""

    dataset_path = state.get("dataset_path") or str(DEFAULT_CONFIG.dataset_path)
    listings = load_listings(dataset_path)
    return {
        "dataset_path": str(dataset_path),
        "listings": listings,
        "attempt_count": int(state.get("attempt_count", 0)),
        "relaxation_history": list(state.get("relaxation_history", [])),
        "questions_asked": list(state.get("questions_asked", [])),
        "need_user_input": False,
        "user_question": None,
    }
