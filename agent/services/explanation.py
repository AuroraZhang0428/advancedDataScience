"""Explanation helpers for deterministic recommendation summaries."""

from __future__ import annotations

from typing import Any

import os
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
except ImportError:
    ChatOpenAI = None
    PromptTemplate = None

def _rewrite_with_llm(draft: str) -> str:
    """Use an LLM to rewrite the deterministic draft."""
    if ChatOpenAI is None or PromptTemplate is None or not os.environ.get("OPENAI_API_KEY"):
        return draft
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = PromptTemplate.from_template(
            "You are an expert real estate and leasing agent.\n"
            "Rewrite the following deterministic listing description into a polished, natural, "
            "and persuasive short paragraph.\n"
            "CRITICAL: You MUST include the exact numerical listing price (e.g., $1,500) and explicit trade-offs.\n"
            "Maintain all factual information, score breakdowns, and make it sound conversational.\n"
            "Ensure you clearly emphasize whether the price matches the user's budget and qualitative preference (cheap/expensive).\n\n"
            "Draft:\n{draft}\n\nPolished Version:"
        )
        chain = prompt | llm
        result = chain.invoke({"draft": draft})
        return result.content.strip()
    except Exception as e:
        print(f"Warning: LLM rewrite failed: {e}")
        return draft


def _describe_top_strengths(score_breakdown: dict[str, float]) -> list[str]:
    """Turn the strongest score components into user-facing phrases."""

    labels = {
        "review_rating": "strong review quality",
        "amenity_match": "amenities that line up well with the request",
        "purpose_alignment": "a good fit for how the user wants to use the space",
        "neighborhood_fit": "location alignment with the preferred area",
        "price_score": "excellent price value relative to the budget",
    }
    ranked = sorted(score_breakdown.items(), key=lambda item: item[1], reverse=True)
    return [labels[key] for key, value in ranked if value >= 0.65 and key in labels]


def _describe_tradeoffs(score_breakdown: dict[str, float]) -> list[str]:
    """Turn weaker score components into trade-off phrases."""

    labels = {
        "review_rating": "review quality is acceptable rather than exceptional",
        "amenity_match": "some requested amenities may be missing",
        "purpose_alignment": "remote-work or quiet-work support is not perfect",
        "neighborhood_fit": "the listing is a weaker location match than the top neighborhood choices",
        "price_score": "the price is less ideal compared to budget or price preference",
    }
    ranked = sorted(score_breakdown.items(), key=lambda item: item[1])
    tradeoffs = [labels[key] for key, value in ranked if value < 0.60 and key in labels]
    
    if not tradeoffs and ranked:
        lowest_key, lowest_val = ranked[0]
        if lowest_val < 0.95 and lowest_key in labels:
            tradeoffs.append(f"relatively speaking, {labels[lowest_key]}")
            
    return tradeoffs


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
    if price is not None and str(price).strip():
        facts.append(f"Price: ${float(price):,.0f}")

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

    # Rewrite the deterministic draft into a more polished narrative using an LLM.
    draft_explanation = " ".join(facts)
    return _rewrite_with_llm(draft_explanation)


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
