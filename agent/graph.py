"""LangGraph assembly for the apartment leasing agent."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.enrich_candidates import enrich_candidates_node
from agent.nodes.evaluate_results import evaluate_results_node, evaluate_results_route
from agent.nodes.explain import explain_node
from agent.nodes.filter_listings import filter_listings_node
from agent.nodes.load_data import load_data_node
from agent.nodes.parse_preferences import parse_preferences_node
from agent.nodes.relax_or_ask import relax_or_ask_node, relax_or_ask_route
from agent.nodes.score_rank import score_rank_node
from agent.state import AgentState


def build_graph():
    """Build and compile the LangGraph workflow."""

    builder = StateGraph(AgentState)

    builder.add_node("load_data", load_data_node)
    builder.add_node("parse_preferences", parse_preferences_node)
    builder.add_node("filter_listings", filter_listings_node)
    builder.add_node("score_rank", score_rank_node)
    builder.add_node("enrich_candidates", enrich_candidates_node)
    builder.add_node("evaluate_results", evaluate_results_node)
    builder.add_node("relax_or_ask", relax_or_ask_node)
    builder.add_node("explain", explain_node)

    builder.add_edge(START, "load_data")
    builder.add_edge("load_data", "parse_preferences")
    builder.add_edge("parse_preferences", "filter_listings")
    builder.add_edge("filter_listings", "score_rank")
    builder.add_edge("score_rank", "enrich_candidates")
    builder.add_edge("enrich_candidates", "evaluate_results")

    builder.add_conditional_edges(
        "evaluate_results",
        evaluate_results_route,
        {
            "sufficient": "explain",
            "insufficient": "relax_or_ask",
        },
    )
    builder.add_conditional_edges(
        "relax_or_ask",
        relax_or_ask_route,
        {
            "retry": "filter_listings",
            "wait_user": END,
            "explain": "explain",
        },
    )
    builder.add_edge("explain", END)

    return builder.compile()
