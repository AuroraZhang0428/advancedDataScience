"""Tool schemas and executors for the ReAct orchestrator.

Each tool wraps an existing service function and returns (observation_text, state_updates).
The orchestrator calls execute_tool() and merges state_updates into its working state.
"""

from __future__ import annotations

from typing import Any

from agent.config import DEFAULT_CONFIG
from agent.services.explanation import generate_final_output
from agent.services.scoring import (
    filter_hard_constraints,
    rank_listings,
    resolve_scoring_weights,
    results_are_sufficient,
)


# ---------------------------------------------------------------------------
# OpenAI function-calling schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "filter_listings",
            "description": (
                "Apply the current hard constraints to the full dataset and report how many "
                "listings match. Call this first, and again after any adjust_constraint call."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_and_rank",
            "description": (
                "Score and rank the currently filtered listings against the user's soft "
                "preferences. Returns quality assessment and a summary of the top results. "
                "Call this after filter_listings or after adjust_preference."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_price_range",
            "description": (
                "Inspect the price distribution in the full dataset for a given bedroom count. "
                "Use this to understand what budgets are realistic before adjusting max_price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_bedrooms": {
                        "type": "integer",
                        "description": "Only consider listings with at least this many bedrooms.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_constraint",
            "description": (
                "Change a hard constraint to widen the search. Supported constraints: "
                "max_price (number), min_bedrooms (integer), min_bathrooms (number). "
                "After calling this, call filter_listings to apply it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "constraint": {
                        "type": "string",
                        "enum": ["max_price", "min_bedrooms", "min_bathrooms"],
                        "description": "The hard constraint to update.",
                    },
                    "value": {
                        "description": "New value for the constraint.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One-sentence reason shown to the user.",
                    },
                },
                "required": ["constraint", "value", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_preference",
            "description": (
                "Change a soft preference to shift or widen the search. Supported: "
                "preferred_neighborhoods (list[str]), desired_amenities (list[str]), "
                "review_min_rating (float 0-5), amenity_strictness (float 0-1). "
                "After calling this, call score_and_rank to apply it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "preference": {
                        "type": "string",
                        "enum": [
                            "preferred_neighborhoods",
                            "desired_amenities",
                            "review_min_rating",
                            "amenity_strictness",
                        ],
                        "description": "The soft preference to update.",
                    },
                    "value": {
                        "description": "New value for the preference.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One-sentence reason shown to the user.",
                    },
                },
                "required": ["preference", "value", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enrich_with_location",
            "description": (
                "Add live neighborhood context (transit access, food scene, commute times) "
                "to the top shortlisted listings via Google Maps. Only works when "
                "GOOGLE_MAPS_API_KEY is set. Call after score_and_rank if location context matters."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Pause execution and ask the user a clarifying question. Use sparingly — "
                "only when the user's explicit decision is needed (e.g., confirming a large "
                "budget increase or a major trade-off). This ends the current search turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to present to the user.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_recommendations",
            "description": (
                "Generate the final recommendations with polished explanations and end the "
                "search. ALWAYS call this eventually — even if quality is imperfect. "
                "It is better to return the best available results with caveats than to "
                "return nothing. Call this after scoring, or after exhausting reasonable "
                "adaptations."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Tools that terminate the orchestrator loop
TERMINAL_TOOLS: frozenset[str] = frozenset({"ask_user", "finalize_recommendations"})


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def execute_tool(
    name: str,
    args: dict[str, Any],
    working_state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Execute a named tool and return (observation_text, state_updates)."""
    dispatch = {
        "filter_listings": _filter_listings,
        "score_and_rank": _score_and_rank,
        "check_price_range": _check_price_range,
        "adjust_constraint": _adjust_constraint,
        "adjust_preference": _adjust_preference,
        "enrich_with_location": _enrich_with_location,
        "ask_user": _ask_user,
        "finalize_recommendations": _finalize_recommendations,
    }
    fn = dispatch.get(name)
    if fn is None:
        return f"Unknown tool: {name!r}", {}
    return fn(args, working_state)


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def _filter_listings(args: dict, state: dict) -> tuple[str, dict]:
    filtered = filter_hard_constraints(
        listings=state.get("listings", []),
        hard_constraints=state.get("hard_constraints", {}),
    )
    n = len(filtered)
    if n == 0:
        prices = [l["price"] for l in state.get("listings", []) if l.get("price")]
        cheapest = f" Cheapest listing in dataset: ${min(prices):.0f}/night." if prices else ""
        return (
            f"0 listings match current constraints.{cheapest} "
            "Consider using check_price_range or adjust_constraint."
        ), {"filtered_listings": filtered}

    prices = [l["price"] for l in filtered if l.get("price")]
    price_info = (
        f" Price range: ${min(prices):.0f}–${max(prices):.0f}/night."
        if prices else ""
    )
    return f"{n} listings match current constraints.{price_info}", {"filtered_listings": filtered}


def _score_and_rank(args: dict, state: dict) -> tuple[str, dict]:
    filtered = state.get("filtered_listings", [])
    if not filtered:
        return "No filtered listings to score. Call filter_listings first.", {}

    weights = resolve_scoring_weights(
        state.get("soft_preferences", {}),
        fallback=DEFAULT_CONFIG.scoring_weights,
    )
    ranked = rank_listings(
        listings=filtered,
        soft_preferences=state.get("soft_preferences", {}),
        hard_constraints=state.get("hard_constraints", {}),
        shortlist_size=DEFAULT_CONFIG.shortlist_size,
        weights=weights,
    )

    sufficient, diagnostics = results_are_sufficient(
        scored_listings=ranked,
        hard_constraints=state.get("hard_constraints", {}),
        soft_preferences=state.get("soft_preferences", {}),
        minimum_good_results=DEFAULT_CONFIG.minimum_good_results,
        good_score_threshold=DEFAULT_CONFIG.good_score_threshold,
    )

    good_count = diagnostics.get("good_count", 0)
    quality_label = "SUFFICIENT" if sufficient else "INSUFFICIENT"
    top5 = ranked[:5]
    lines = [
        f"  {i}. {l.get('title', '?')} — score={l.get('score', 0):.2f}, "
        f"${l.get('price', 0):.0f}/night, {l.get('neighborhood', '?')}"
        for i, l in enumerate(top5, 1)
    ]
    obs = (
        f"Scored {len(ranked)} listings. Quality: {quality_label} "
        f"({good_count}/{DEFAULT_CONFIG.minimum_good_results} needed with "
        f"score ≥ {DEFAULT_CONFIG.good_score_threshold}).\n"
        "Top 5 results:\n" + "\n".join(lines)
    )

    return obs, {
        "scored_listings": ranked,
        "shortlisted_listings": ranked[: DEFAULT_CONFIG.shortlist_size],
        "sufficient_results": sufficient,
        "results_diagnostics": diagnostics,
    }


def _check_price_range(args: dict, state: dict) -> tuple[str, dict]:
    min_beds = int(args.get("min_bedrooms", 0))
    all_listings = state.get("listings", [])

    subset = [
        l for l in all_listings
        if l.get("bedrooms", 0) >= min_beds and l.get("price") is not None
    ]
    if not subset:
        return f"No listings found with ≥{min_beds} bedrooms.", {}

    prices = sorted(float(l["price"]) for l in subset)
    n = len(prices)
    p10 = prices[max(0, int(n * 0.10))]
    p25 = prices[max(0, int(n * 0.25))]
    p50 = prices[n // 2]
    p75 = prices[min(n - 1, int(n * 0.75))]

    return (
        f"Price distribution for ≥{min_beds}BR listings ({n} total):\n"
        f"  Min=${prices[0]:.0f}  P10=${p10:.0f}  P25=${p25:.0f}  "
        f"Median=${p50:.0f}  P75=${p75:.0f}  Max=${prices[-1]:.0f}  (per night)"
    ), {}


def _adjust_constraint(args: dict, state: dict) -> tuple[str, dict]:
    name = args["constraint"]
    value = args["value"]
    reason = args.get("reason", "")

    hard = dict(state.get("hard_constraints", {}))
    old_value = hard.get(name)
    hard[name] = value

    history = list(state.get("relaxation_history", []))
    history.append({
        "attempt": len(history) + 1,
        "action": "relax_hard",
        "change": f"{name}: {old_value} → {value}",
        "reason": reason,
    })

    return (
        f"Updated hard constraint '{name}': {old_value} → {value}. "
        f"Reason: {reason}. Call filter_listings to apply."
    ), {"hard_constraints": hard, "relaxation_history": history}


def _adjust_preference(args: dict, state: dict) -> tuple[str, dict]:
    name = args["preference"]
    value = args["value"]
    reason = args.get("reason", "")

    soft = dict(state.get("soft_preferences", {}))
    old_value = soft.get(name)
    soft[name] = value

    history = list(state.get("relaxation_history", []))
    history.append({
        "attempt": len(history) + 1,
        "action": "relax_soft",
        "change": f"{name}: {old_value!r} → {value!r}",
        "reason": reason,
    })

    return (
        f"Updated soft preference '{name}': {old_value!r} → {value!r}. "
        f"Reason: {reason}. Call score_and_rank to apply."
    ), {"soft_preferences": soft, "relaxation_history": history}


def _enrich_with_location(args: dict, state: dict) -> tuple[str, dict]:
    try:
        from agent.services.google_maps import enrich_and_rerank_listings

        reranked, diagnostics = enrich_and_rerank_listings(
            listings=state.get("shortlisted_listings", []),
            soft_preferences=state.get("soft_preferences", {}),
            hard_constraints=state.get("hard_constraints", {}),
        )
        if reranked:
            return (
                f"Enriched {len(reranked)} listings with live neighborhood data.",
                {
                    "enriched_listings": reranked,
                    "scored_listings": reranked,
                    "shortlisted_listings": reranked,
                    "google_enrichment_diagnostics": diagnostics,
                },
            )
        return "Location enrichment returned no data (Google Maps key may be missing).", {}
    except Exception as exc:
        return f"Location enrichment unavailable: {exc}", {}


def _ask_user(args: dict, state: dict) -> tuple[str, dict]:
    question = args.get("question", "")
    questions_asked = list(state.get("questions_asked", []))
    questions_asked.append(question)
    return (
        f"Pausing to ask user: {question}",
        {
            "need_user_input": True,
            "user_question": question,
            "questions_asked": questions_asked,
        },
    )


def _finalize_recommendations(args: dict, state: dict) -> tuple[str, dict]:
    scored = state.get("scored_listings", [])

    # Safety net: if score_and_rank was never called, use filtered listings directly
    if not scored:
        filtered = state.get("filtered_listings", [])
        if filtered:
            scored = sorted(filtered, key=lambda l: float(l.get("review_rating") or 0), reverse=True)
        else:
            return "No listings available to finalize. The search found no matching results.", {}

    recommendations, explanations = generate_final_output(
        scored_listings=scored,
        hard_constraints=state.get("hard_constraints", {}),
        soft_preferences=state.get("soft_preferences", {}),
        relaxation_history=state.get("relaxation_history", []),
        top_k=DEFAULT_CONFIG.top_k_recommendations,
    )
    return (
        f"Generated {len(recommendations)} final recommendations with explanations.",
        {
            "final_recommendations": recommendations,
            "final_explanations": explanations,
        },
    )
