"""LangGraph node for deciding whether current results are sufficient."""

from __future__ import annotations

from agent.config import DEFAULT_CONFIG
from agent.services.scoring import results_are_sufficient
from agent.state import AgentState


def evaluate_results_node(state: AgentState) -> AgentState:
    """Evaluate whether the ranked results are good enough to stop."""

    sufficient, diagnostics = results_are_sufficient(
        scored_listings=state.get("scored_listings", []),
        minimum_good_results=DEFAULT_CONFIG.minimum_good_results,
        good_score_threshold=DEFAULT_CONFIG.good_score_threshold,
    )
    return {
        "sufficient_results": sufficient,
        "results_diagnostics": diagnostics,
    }


def evaluate_results_route(state: AgentState) -> str:
    """Route either to explanation or adaptive relaxation."""

    return "sufficient" if state.get("sufficient_results", False) else "insufficient"
