"""Explanation helpers for deterministic recommendation summaries."""

from __future__ import annotations

from typing import Any


def _describe_top_strengths(score_breakdown: dict[str, float]) -> list[str]:
    """Turn the strongest score components into user-facing phrases."""

    labels = {
        "review_rating": "strong review quality",
        "amenity_match": "amenities that line up well with the request",
        "purpose_alignment": "a good fit for how the user wants to use the space",
        "neighborhood_fit": "location alignment with the preferred area",
    }
    ranked = sorted(score_breakdown.items(), key=lambda item: item[1], reverse=True)
    return [labels[key] for key, value in ranked[:2] if value >= 0.65 and key in labels]


def _describe_tradeoffs(score_breakdown: dict[str, float]) -> list[str]:
    """Turn weaker score components into trade-off phrases."""

    labels = {
        "review_rating": "review quality is acceptable rather than exceptional",
        "amenity_match": "some requested amenities may be missing",
        "purpose_alignment": "remote-work or quiet-work support is not perfect",
        "neighborhood_fit": "the listing is a weaker location match than the top neighborhood choices",
    }
    ranked = sorted(score_breakdown.items(), key=lambda item: item[1])
    return [labels[key] for key, value in ranked[:2] if value < 0.60 and key in labels]


def generate_listing_explanation(
    listing: dict[str, Any],
    hard_constraints: dict[str, Any],
    soft_preferences: dict[str, Any],
    relaxation_history: list[dict[str, Any]],
) -> str:
    """Generate a deterministic explanation for a single recommendation."""

    title = listing.get("title") or "This listing"
    neighborhood = listing.get("neighborhood") or listing.get("neighborhood_group") or "the searched area"
    score = float(listing.get("score", 0.0))
    breakdown = {
        key: float(value) for key, value in (listing.get("score_breakdown") or {}).items()
    }

    strengths = _describe_top_strengths(breakdown)
    tradeoffs = _describe_tradeoffs(breakdown)
    price = listing.get("price")
    bedrooms = listing.get("bedrooms")
    bathrooms = listing.get("bathrooms")

    facts: list[str] = [f"{title} stands out with an overall score of {score:.2f}."]
    facts.append(f"It is located in {neighborhood}.")

    detail_bits: list[str] = []
    if bedrooms is not None:
        detail_bits.append(f"{bedrooms:g} bedroom")
    if bathrooms is not None:
        detail_bits.append(f"{bathrooms:g} bathroom")
    if price is not None:
        detail_bits.append(f"priced at ${float(price):,.0f}")
    if detail_bits:
        facts.append("Core facts: " + ", ".join(detail_bits) + ".")

    if strengths:
        facts.append("Why it matches well: " + "; ".join(strengths) + ".")
    if tradeoffs:
        facts.append("Main trade-offs: " + "; ".join(tradeoffs) + ".")

    requested_bedrooms = hard_constraints.get("min_bedrooms")
    requested_bathrooms = hard_constraints.get("min_bathrooms")
    if requested_bedrooms is not None or requested_bathrooms is not None:
        constraint_bits: list[str] = []
        if requested_bedrooms is not None and bedrooms is not None:
            constraint_bits.append(
                f"bedroom target {'met' if float(bedrooms) >= float(requested_bedrooms) else 'missed'}"
            )
        if requested_bathrooms is not None and bathrooms is not None:
            constraint_bits.append(
                f"bathroom target {'met' if float(bathrooms) >= float(requested_bathrooms) else 'missed'}"
            )
        if constraint_bits:
            facts.append("Constraint check: " + ", ".join(constraint_bits) + ".")

    preferred_neighborhoods = soft_preferences.get("preferred_neighborhoods") or []
    if preferred_neighborhoods:
        preferred_text = ", ".join(str(area) for area in preferred_neighborhoods)
        facts.append(f"Preferred areas considered: {preferred_text}.")

    if relaxation_history:
        latest = relaxation_history[-1]
        facts.append(
            "Search context: the agent had to adjust preferences by "
            f"{latest.get('action', 'making a trade-off')} because {latest.get('reason', 'results were limited')}."
        )

    if breakdown:
        facts.append(
            "Score breakdown: "
            + ", ".join(f"{key}={value:.2f}" for key, value in breakdown.items())
            + "."
        )

    # TODO: Add an optional LLM rewrite step here to turn the deterministic draft
    # into a more polished narrative while preserving the same factual content.
    return " ".join(facts)


def generate_final_output(
    scored_listings: list[dict[str, Any]],
    hard_constraints: dict[str, Any],
    soft_preferences: dict[str, Any],
    relaxation_history: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build final recommendation payloads and explanations."""

    recommendations = scored_listings[:top_k]
    explanations = [
        generate_listing_explanation(
            listing,
            hard_constraints=hard_constraints,
            soft_preferences=soft_preferences,
            relaxation_history=relaxation_history,
        )
        for listing in recommendations
    ]
    return recommendations, explanations
