"""Typed LangGraph state for the apartment leasing workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Central mutable state shared across LangGraph nodes."""

    user_query: str
    dataset_path: str
    raw_preferences: dict[str, Any]
    hard_constraints: dict[str, Any]
    soft_preferences: dict[str, Any]
    relaxable_constraints: dict[str, dict[str, Any]]
    listings: list[dict[str, Any]]
    filtered_listings: list[dict[str, Any]]
    scored_listings: list[dict[str, Any]]
    shortlisted_listings: list[dict[str, Any]]
    enriched_listings: list[dict[str, Any]]
    google_enrichment_diagnostics: dict[str, Any]
    sufficient_results: bool
    results_diagnostics: dict[str, Any]
    attempt_count: int
    relaxation_history: list[dict[str, Any]]
    latest_decision: dict[str, Any]
    need_user_input: bool
    user_question: str | None
    questions_asked: list[str]
    final_recommendations: list[dict[str, Any]]
    final_explanations: list[str]
