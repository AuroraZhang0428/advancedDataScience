"""LangGraph node that runs the ReAct orchestrator."""

from __future__ import annotations

from agent.orchestrator import run_orchestrator
from agent.state import AgentState


def orchestrate_node(state: AgentState) -> AgentState:
    """Hand control to the ReAct orchestrator for the full search loop."""
    return run_orchestrator(state)
