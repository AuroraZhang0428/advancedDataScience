"""Rule-based relaxation policy for weak search results."""

from __future__ import annotations

import os
from typing import Any

from agent.config import DEFAULT_CONFIG
from agent.models import RelaxationDecision

try:
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI

    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    BaseModel = None
    Field = None
    ChatOpenAI = None


if HAS_LLM:
    class AutonomousRelaxationChoice(BaseModel):
        action: str = Field(description="One of 'relax_soft', 'relax_hard', 'ask_user', or 'stop'.")
        relaxed_key: str | None = Field(
            default=None,
            description="The logical constraint to relax, or null when action is stop.",
        )
        reason: str = Field(description="Short explanation for the chosen action.")


def _history_contains(relaxation_history: list[dict[str, Any]], key: str) -> bool:
    """Check whether a logical relaxation key has already been used."""

    for entry in relaxation_history:
        change = entry.get("change", {})
        if change.get("relaxed_key") == key:
            return True
    return False


def _llm_is_available() -> bool:
    """Return whether the autonomous relaxation policy can call the OpenAI LLM."""

    return HAS_LLM and ChatOpenAI is not None and bool(os.environ.get("OPENAI_API_KEY"))


def _require_llm_relaxation() -> None:
    """Ensure the OpenAI-backed relaxation policy is available."""

    if not _llm_is_available():
        raise RuntimeError("OPENAI_API_KEY is required because rule-based relaxation fallback has been removed.")


def _effective_budget_value(hard_constraints: dict[str, Any], relaxable_constraints: dict[str, Any]) -> float | None:
    """Compute the next autonomous budget target when budget relaxation is chosen."""

    current_budget = hard_constraints.get("max_price")
    if current_budget is None:
        return None
    suggested_increase_pct = float(relaxable_constraints.get("max_price", {}).get("suggested_increase_pct", 0.1))
    return round(float(current_budget) * (1 + suggested_increase_pct), 2)


def _effective_target_value(soft_preferences: dict[str, Any], relaxable_constraints: dict[str, Any], listings: list[dict[str, Any]]) -> float | None:
    """Compute a more realistic target price suggestion when the user's target is too low."""

    current_target = soft_preferences.get("target_price")
    if current_target is None:
        return None

    available_prices = sorted(
        float(listing["price"])
        for listing in listings
        if listing.get("price") is not None
    )
    if available_prices:
        reference_price = available_prices[0]
        return round(max(float(current_target) * 1.5, reference_price), 2)

    suggested_increase_pct = float(relaxable_constraints.get("target_price", {}).get("suggested_increase_pct", 0.5))
    return round(float(current_target) * (1 + suggested_increase_pct), 2)


def _available_relaxation_options(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the currently valid autonomous relaxation options."""

    relaxation_history = state.get("relaxation_history", [])
    soft_preferences = state.get("soft_preferences", {})
    hard_constraints = state.get("hard_constraints", {})
    relaxable_constraints = state.get("relaxable_constraints", {})
    results_diagnostics = state.get("results_diagnostics", {})
    filtered_listings = state.get("filtered_listings", [])
    questions_asked = set(state.get("questions_asked", []))
    options: list[dict[str, Any]] = []

    if (
        results_diagnostics.get("target_price_fit_poor")
        and relaxable_constraints.get("target_price", {}).get("can_relax")
        and "target_price" not in questions_asked
    ):
        current_target = float(soft_preferences.get("target_price", 0.0))
        suggested_target = _effective_target_value(soft_preferences, relaxable_constraints, filtered_listings)
        if suggested_target is not None:
            options.append(
                {
                    "relaxed_key": "target_price",
                    "action": "ask_user",
                    "question_key": "target_price",
                    "user_question": (
                        f"I could not find robust matches near your target of ${current_target:,.0f} per night. "
                        f"Would you like me to search closer to about ${suggested_target:,.0f} per night instead?"
                    ),
                    "change": {},
                }
            )

    if (
        relaxable_constraints.get("preferred_neighborhoods", {}).get("can_relax")
        and not _history_contains(relaxation_history, "preferred_neighborhoods")
        and soft_preferences.get("preferred_neighborhoods")
    ):
        options.append(
            {
                "relaxed_key": "preferred_neighborhoods",
                "action": "relax_soft",
                "change": {"soft_preferences": {"expanded_neighborhood_search": True}},
            }
        )

    review_rule = relaxable_constraints.get("review_min_rating", {})
    current_review_min = soft_preferences.get("review_min_rating")
    minimum_review_min = review_rule.get("minimum")
    if (
        review_rule.get("can_relax")
        and current_review_min is not None
        and minimum_review_min is not None
        and float(current_review_min) > float(minimum_review_min)
        and not _history_contains(relaxation_history, "review_min_rating")
    ):
        next_review_min = max(
            float(minimum_review_min),
            float(current_review_min) - float(review_rule.get("step", 0.2)),
        )
        options.append(
            {
                "relaxed_key": "review_min_rating",
                "action": "relax_soft",
                "change": {"soft_preferences": {"review_min_rating": round(next_review_min, 2)}},
            }
        )

    amenity_rule = relaxable_constraints.get("desired_amenities", {})
    amenity_strictness = float(soft_preferences.get("amenity_strictness", 1.0))
    if (
        amenity_rule.get("can_relax")
        and amenity_strictness > 0.6
        and not _history_contains(relaxation_history, "desired_amenities")
    ):
        options.append(
            {
                "relaxed_key": "desired_amenities",
                "action": "relax_soft",
                "change": {
                    "soft_preferences": {"amenity_strictness": round(max(0.6, amenity_strictness - 0.25), 2)}
                },
            }
        )

    if (
        relaxable_constraints.get("min_bedrooms", {}).get("can_relax")
        and hard_constraints.get("min_bedrooms") is not None
        and "min_bedrooms" not in questions_asked
    ):
        current_bedrooms = int(hard_constraints["min_bedrooms"])
        options.append(
            {
                "relaxed_key": "min_bedrooms",
                "action": "ask_user",
                "question_key": "min_bedrooms",
                "user_question": (
                    f"Would you consider {max(current_bedrooms - 1, 0)} bedroom(s) instead of {current_bedrooms} "
                    "if that unlocks stronger matches?"
                ),
                "change": {},
            }
        )

    if (
        relaxable_constraints.get("max_price", {}).get("can_relax")
        and hard_constraints.get("max_price") is not None
        and "max_price" not in questions_asked
    ):
        next_budget = _effective_budget_value(hard_constraints, relaxable_constraints)
        if next_budget is not None:
            options.append(
                {
                    "relaxed_key": "max_price",
                    "action": "ask_user",
                    "question_key": "max_price",
                    "user_question": (
                        f"Would you be open to increasing the budget from ${float(hard_constraints['max_price']):,.0f} "
                        f"to about ${next_budget:,.0f} for better matches?"
                    ),
                    "change": {},
                }
            )

    return options


def _top_candidate_summary(state: dict[str, Any], limit: int = 3) -> str:
    """Summarize the current top candidates for the autonomous decision prompt."""

    candidates = state.get("scored_listings", [])[:limit]
    if not candidates:
        return "No scored candidates available."

    lines: list[str] = []
    for candidate in candidates:
        lines.append(
            f"id={candidate.get('id')} | title={candidate.get('title')} | score={float(candidate.get('score', 0.0)):.2f} | "
            f"price={candidate.get('price')} | neighborhood={candidate.get('neighborhood') or candidate.get('neighborhood_group')} | "
            f"llm_rank_reason={candidate.get('llm_rank_reason', '')}"
        )
    return "\n".join(lines)


def _choose_relaxation_action_llm(state: dict[str, Any]) -> RelaxationDecision:
    """Use the LLM to choose the next autonomous relaxation step."""

    _require_llm_relaxation()

    attempt_count = int(state.get("attempt_count", 0))
    if attempt_count >= DEFAULT_CONFIG.max_attempts:
        return RelaxationDecision(
            action="stop",
            reason="Maximum search attempts reached, so the agent will explain the best available options.",
        )

    options = _available_relaxation_options(state)
    if not options:
        return RelaxationDecision(
            action="stop",
            reason="No autonomous relaxation options remain, so the agent will summarize the best available listings.",
        )

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(AutonomousRelaxationChoice)
        prompt = (
            "You are controlling an autonomous apartment recommendation agent.\n"
            "Choose the single best next action to improve result quality.\n"
            "You may only choose one of the supplied options or stop.\n"
            "Prefer the least destructive change, and ask the user before weakening semi-hard constraints or unrealistic price targets.\n\n"
            f"Attempt count: {attempt_count}\n"
            f"Results diagnostics: {state.get('results_diagnostics', {})}\n"
            f"Hard constraints: {state.get('hard_constraints', {})}\n"
            f"Soft preferences: {state.get('soft_preferences', {})}\n"
            f"Top candidates:\n{_top_candidate_summary(state)}\n\n"
            f"Allowed relaxation options:\n{options}\n"
        )
        choice = structured_llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(f"OpenAI-backed relaxation decision failed: {exc}") from exc

    if str(choice.action).lower() == "stop":
        return RelaxationDecision(action="stop", reason=choice.reason)

    selected_key = str(choice.relaxed_key or "").strip()
    selected_option = next((option for option in options if option["relaxed_key"] == selected_key), None)
    if selected_option is None:
        return RelaxationDecision(action="stop", reason="LLM selected an unsupported relaxation, so the agent will stop safely.")

    if str(choice.action).lower() == "ask_user":
        return RelaxationDecision(
            action="ask_user",
            reason=choice.reason,
            user_question=selected_option.get("user_question"),
            change={"question_key": selected_option.get("question_key")},
        )

    return RelaxationDecision(
        action=selected_option["action"],
        reason=choice.reason,
        change={
            "relaxed_key": selected_option["relaxed_key"],
            **selected_option["change"],
        },
    )


def _choose_relaxation_action_rule_based(state: dict[str, Any]) -> RelaxationDecision:
    """Choose the next relaxation or clarification action."""

    attempt_count = int(state.get("attempt_count", 0))
    relaxation_history = state.get("relaxation_history", [])
    soft_preferences = state.get("soft_preferences", {})
    hard_constraints = state.get("hard_constraints", {})
    relaxable_constraints = state.get("relaxable_constraints", {})
    questions_asked = set(state.get("questions_asked", []))
    filtered_listings = state.get("filtered_listings", [])
    results_diagnostics = state.get("results_diagnostics", {})

    if attempt_count >= DEFAULT_CONFIG.max_attempts:
        return RelaxationDecision(
            action="stop",
            reason="Maximum search attempts reached, so the agent will explain the best available options.",
        )

    if (
        results_diagnostics.get("target_price_fit_poor")
        and relaxable_constraints.get("target_price", {}).get("can_relax")
        and "target_price" not in questions_asked
        and soft_preferences.get("target_price") is not None
    ):
        current_target = float(soft_preferences["target_price"])
        suggested_target = _effective_target_value(soft_preferences, relaxable_constraints, filtered_listings)
        return RelaxationDecision(
            action="ask_user",
            reason="The current target price appears too far from the available listings, so the agent should confirm whether to search at a more realistic nightly price.",
            user_question=(
                f"I could not find robust matches near your target of ${current_target:,.0f} per night. "
                f"Would you like me to search closer to about ${float(suggested_target or current_target):,.0f} per night instead?"
            ),
            change={"question_key": "target_price"},
        )

    if not filtered_listings:
        if (
            relaxable_constraints.get("min_bedrooms", {}).get("can_relax")
            and "min_bedrooms" not in questions_asked
            and hard_constraints.get("min_bedrooms") is not None
        ):
            current = int(hard_constraints["min_bedrooms"])
            fallback = max(current - 1, 0)
            return RelaxationDecision(
                action="ask_user",
                reason="No listings satisfy the current hard filters, and the bedroom requirement is the clearest semi-hard constraint to revisit.",
                user_question=(
                    f"No current listings satisfy every hard constraint. Would you consider {fallback} bedroom(s) "
                    f"instead of {current} to broaden the search?"
                ),
                change={"question_key": "min_bedrooms"},
            )

        if (
            relaxable_constraints.get("max_price", {}).get("can_relax")
            and "max_price" not in questions_asked
            and hard_constraints.get("max_price") is not None
        ):
            current_budget = float(hard_constraints["max_price"])
            suggested = current_budget * (
                1 + float(relaxable_constraints["max_price"].get("suggested_increase_pct", 0.1))
            )
            return RelaxationDecision(
                action="ask_user",
                reason="No listings survive the current hard filters, so budget flexibility is the next trade-off worth clarifying.",
                user_question=(
                    f"I could broaden the search if you are open to increasing the budget from ${current_budget:,.0f} "
                    f"to about ${suggested:,.0f}. Would you like to do that?"
                ),
                change={"question_key": "max_price"},
            )

    preferred_area_rule = relaxable_constraints.get("preferred_neighborhoods", {})
    if (
        preferred_area_rule.get("can_relax")
        and not _history_contains(relaxation_history, "preferred_neighborhoods")
        and soft_preferences.get("preferred_neighborhoods")
    ):
        return RelaxationDecision(
            action="relax_soft",
            reason="Too few strong matches; expanding beyond the preferred neighborhoods is the least costly trade-off.",
            change={
                "relaxed_key": "preferred_neighborhoods",
                "soft_preferences": {
                    "expanded_neighborhood_search": True,
                },
            },
        )

    review_rule = relaxable_constraints.get("review_min_rating", {})
    current_review_min = soft_preferences.get("review_min_rating")
    minimum_review_min = review_rule.get("minimum")
    if (
        review_rule.get("can_relax")
        and current_review_min is not None
        and minimum_review_min is not None
        and float(current_review_min) > float(minimum_review_min)
        and not _history_contains(relaxation_history, "review_min_rating")
    ):
        next_review_min = max(
            float(minimum_review_min),
            float(current_review_min) - float(review_rule.get("step", 0.2)),
        )
        return RelaxationDecision(
            action="relax_soft",
            reason="Results are still thin, so slightly lowering the preferred review threshold is the next safest relaxation.",
            change={
                "relaxed_key": "review_min_rating",
                "soft_preferences": {"review_min_rating": round(next_review_min, 2)},
            },
        )

    amenity_rule = relaxable_constraints.get("desired_amenities", {})
    amenity_strictness = float(soft_preferences.get("amenity_strictness", 1.0))
    if (
        amenity_rule.get("can_relax")
        and amenity_strictness > 0.6
        and not _history_contains(relaxation_history, "desired_amenities")
    ):
        return RelaxationDecision(
            action="relax_soft",
            reason="Relaxing amenity strictness can surface more balanced listings without weakening core space requirements.",
            change={
                "relaxed_key": "desired_amenities",
                "soft_preferences": {"amenity_strictness": round(max(0.6, amenity_strictness - 0.25), 2)},
            },
        )

    if (
        relaxable_constraints.get("min_bedrooms", {}).get("can_relax")
        and "min_bedrooms" not in questions_asked
        and hard_constraints.get("min_bedrooms") is not None
    ):
        current = int(hard_constraints["min_bedrooms"])
        fallback = max(current - 1, 0)
        return RelaxationDecision(
            action="ask_user",
            reason="The next meaningful improvement would require weakening the bedroom requirement, which should be user-approved.",
            user_question=(
                f"I only found limited strong matches. Would you consider {fallback} bedroom(s) instead of {current} "
                "if that unlocks better location or quality?"
            ),
            change={"question_key": "min_bedrooms"},
        )

    if (
        relaxable_constraints.get("max_price", {}).get("can_relax")
        and "max_price" not in questions_asked
        and hard_constraints.get("max_price") is not None
    ):
        current_budget = float(hard_constraints["max_price"])
        suggested = current_budget * (1 + float(relaxable_constraints["max_price"].get("suggested_increase_pct", 0.1)))
        return RelaxationDecision(
            action="ask_user",
            reason="Budget flexibility is now the most likely way to unlock stronger results, so the agent should ask before changing it.",
            user_question=(
                f"Would you be open to increasing the budget from ${current_budget:,.0f} to about ${suggested:,.0f} "
                "for better overall matches?"
            ),
            change={"question_key": "max_price"},
        )

    return RelaxationDecision(
        action="stop",
        reason="No safe automatic relaxations remain, so the agent will summarize the best available listings.",
    )


def choose_relaxation_action(state: dict[str, Any]) -> RelaxationDecision:
    """Choose the next relaxation step using only the LLM policy."""

    return _choose_relaxation_action_llm(state)
