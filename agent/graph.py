"""LangGraph assembly for the NestAI apartment leasing agent.

The graph now has three nodes:
  load_data        — read and normalise the CSV dataset
  parse_preferences — extract structured preferences from the free-text query
  orchestrate      — ReAct loop: the LLM decides which tools to call and adapts
                     the search strategy autonomously until it finalises results
                     or asks the user a clarifying question.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.load_data import load_data_node
from agent.nodes.orchestrate import orchestrate_node
from agent.nodes.parse_preferences import parse_preferences_node
from agent.state import AgentState


def build_graph():
    """Build and compile the LangGraph workflow."""

    builder = StateGraph(AgentState)

    builder.add_node("load_data", load_data_node)
    builder.add_node("parse_preferences", parse_preferences_node)
    builder.add_node("orchestrate", orchestrate_node)

    builder.add_edge(START, "load_data")
    builder.add_edge("load_data", "parse_preferences")
    builder.add_edge("parse_preferences", "orchestrate")
    builder.add_edge("orchestrate", END)

    return builder.compile()
