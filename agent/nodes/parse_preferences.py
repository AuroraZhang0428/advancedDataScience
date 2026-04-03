"""LangGraph node for parsing user preferences."""

from __future__ import annotations

from agent.services.parser import parse_preferences
from agent.state import AgentState


def parse_preferences_node(state: AgentState) -> AgentState:
    """Parse the natural-language query into structured preferences."""

    parsed = parse_preferences(state["user_query"])
    return {
        "raw_preferences": parsed["raw_preferences"],
        "hard_constraints": parsed["hard_constraints"],
        "soft_preferences": parsed["soft_preferences"],
        "relaxable_constraints": parsed["relaxable_constraints"],
    }
