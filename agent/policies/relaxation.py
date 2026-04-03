"""Rule-based relaxation policy for weak search results."""

from __future__ import annotations

from typing import Any

from agent.config import DEFAULT_CONFIG
from agent.models import RelaxationDecision


def _history_contains(relaxation_history: list[dict[str, Any]], key: str) -> bool:
    """Check whether a logical relaxation key has already been used."""

    for entry in relaxation_history:
        change = entry.get("change", {})
        if change.get("relaxed_key") == key:
            return True
    return False


def choose_relaxation_action(state: dict[str, Any]) -> RelaxationDecision:
    """Choose the next relaxation or clarification action."""

    attempt_count = int(state.get("attempt_count", 0))
    relaxation_history = state.get("relaxation_history", [])
    soft_preferences = state.get("soft_preferences", {})
    hard_constraints = state.get("hard_constraints", {})
    relaxable_constraints = state.get("relaxable_constraints", {})
    questions_asked = set(state.get("questions_asked", []))
    filtered_listings = state.get("filtered_listings", [])

    if attempt_count >= DEFAULT_CONFIG.max_attempts:
        return RelaxationDecision(
            action="stop",
            reason="Maximum search attempts reached, so the agent will explain the best available options.",
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
