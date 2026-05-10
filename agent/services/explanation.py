"""Explanation helpers for deterministic recommendation summaries."""

from __future__ import annotations

from typing import Any

import json as _json
import os
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


# ── Review extraction ────────────────────────────────────────────────────────

def _extract_review_snippets(listing: dict[str, Any], max_recent: int = 2) -> str:
    """Return a compact block of guest review text for the LLM to read.

    Mirrors the user-facing display: up to max_recent non-critical reviews
    and 1 critical review (if any exist), taken directly from the JSON
    sample_reviews column so the LLM reads the same reviews the user sees.
    """
    raw = (listing.get("raw") or {}).get("sample_reviews", "")
    if not raw:
        return ""
    try:
        records = _json.loads(str(raw))
    except (ValueError, TypeError):
        return ""

    recent = [r for r in records if not r.get("critical")]
    critical = [r for r in records if r.get("critical")]

    shown = recent[:max_recent]
    if critical:
        shown.append(critical[0])

    if not shown:
        return ""

    lines = []
    for r in shown:
        tag = "[critical]" if r.get("critical") else "[recent]"
        name = r.get("name", "Guest")
        text = r.get("text", "").replace("\n", " ").strip()
        if text:
            lines.append(f"{tag} {name}: {text}")

    return "\n".join(lines)


# ── Score → plain-language quality label ─────────────────────────────────────

def _quality(score: float) -> str:
    if score >= 0.80:
        return "excellent"
    if score >= 0.65:
        return "good"
    if score >= 0.50:
        return "acceptable"
    return "weak"


# ── Section builders ──────────────────────────────────────────────────────────

def _build_fit_section(
    listing: dict[str, Any],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
    breakdown: dict[str, float],
) -> str:
    """Describe how the listing fits the user's request in plain language."""
    parts: list[str] = []

    neighborhood = listing.get("neighborhood") or listing.get("neighborhood_group") or "the area"
    price = listing.get("price")
    bedrooms = listing.get("bedrooms")
    bathrooms = listing.get("bathrooms")

    # Location
    parts.append(f"Located in {neighborhood}.")

    # Price — interpret relative to budget, not as a raw score
    if price is not None:
        budget = hard_constraints.get("max_price")
        target = soft_preferences.get("target_price")
        price_val = float(price)
        if target:
            diff_pct = (price_val - float(target)) / float(target) * 100
            if abs(diff_pct) <= 5:
                parts.append(f"Priced at ${price_val:,.0f}/night, right on target.")
            elif diff_pct < 0:
                parts.append(f"Priced at ${price_val:,.0f}/night, {abs(diff_pct):.0f}% under the target.")
            else:
                parts.append(f"Priced at ${price_val:,.0f}/night, {diff_pct:.0f}% above the target.")
        elif budget:
            diff_pct = (price_val - float(budget)) / float(budget) * 100
            if price_val <= float(budget):
                parts.append(f"Priced at ${price_val:,.0f}/night, within the ${float(budget):,.0f} budget.")
            else:
                parts.append(f"Priced at ${price_val:,.0f}/night, {diff_pct:.0f}% over the stated budget.")
        else:
            parts.append(f"Priced at ${price_val:,.0f}/night.")

    # Bedrooms / bathrooms
    room_bits = []
    if bedrooms is not None:
        room_bits.append(f"{bedrooms:g} bedroom{'s' if float(bedrooms) != 1 else ''}")
    if bathrooms is not None:
        room_bits.append(f"{bathrooms:g} bathroom{'s' if float(bathrooms) != 1 else ''}")
    if room_bits:
        parts.append(", ".join(room_bits).capitalize() + ".")

    # Review rating — plain language, no numbers
    review_q = breakdown.get("review_rating")
    review_rating = listing.get("review_rating")
    if review_q is not None:
        label = _quality(review_q)
        if review_rating is not None:
            num = float(review_rating)
            if num >= 4.8:
                parts.append("Guests consistently rate this place very highly.")
            elif num >= 4.5:
                parts.append("Well-reviewed by past guests.")
            elif num >= 4.0:
                parts.append("Generally positive reviews from past guests.")
            else:
                parts.append("Reviews are mixed — worth reading them before booking.")
        else:
            parts.append(f"Review quality is {label}.")

    # Remote work / quiet
    if soft_preferences.get("remote_work"):
        wifi = listing.get("wifi")
        workspace = listing.get("workspace")
        signals = []
        if wifi:
            signals.append("WiFi confirmed")
        if workspace:
            signals.append("dedicated workspace available")
        if signals:
            parts.append("Work-friendly: " + ", ".join(signals) + ".")
        else:
            parts.append("Remote-work amenities are not explicitly listed for this place.")

    if soft_preferences.get("quiet_preference"):
        quiet = listing.get("quiet_score")
        if quiet is not None:
            if float(quiet) >= 0.7:
                parts.append("The area is generally quiet.")
            elif float(quiet) >= 0.5:
                parts.append("The area has moderate noise levels.")
            else:
                parts.append("This is a lively area — may not be the quietest option.")

    # Amenity match
    amenity_q = breakdown.get("amenity_match")
    desired = soft_preferences.get("desired_amenities") or []
    if amenity_q is not None and desired:
        listing_amenities = {str(a).lower() for a in listing.get("amenities", [])}
        matched = [a for a in desired if str(a).lower() in listing_amenities]
        missing = [a for a in desired if str(a).lower() not in listing_amenities]
        if matched:
            parts.append("Includes: " + ", ".join(matched) + ".")
        if missing:
            parts.append("Missing: " + ", ".join(missing) + ".")

    # Location context (Google Maps)
    loc = dict(listing.get("location_context") or {})
    if loc.get("google_maps_enriched"):
        loc_bits = []
        subway = loc.get("nearby_subway_count")
        train = loc.get("nearby_train_count")
        bus = loc.get("nearby_bus_count")
        food = loc.get("nearby_food_count")
        grocery = loc.get("nearby_grocery_count")
        commute = loc.get("average_commute_minutes")
        if subway:
            loc_bits.append(f"{int(subway)} subway stop{'s' if subway != 1 else ''} nearby")
        if train:
            loc_bits.append(f"{int(train)} train stop{'s' if train != 1 else ''} nearby")
        if bus:
            loc_bits.append(f"{int(bus)} bus stop{'s' if bus != 1 else ''} nearby")
        if food is not None:
            loc_bits.append(f"{food} dining options nearby")
        if grocery is not None:
            loc_bits.append(f"{grocery} grocery option{'s' if grocery != 1 else ''} nearby")
        if commute is not None:
            loc_bits.append(f"average commute around {float(commute):.0f} minutes")
        if loc_bits:
            parts.append("Neighborhood snapshot: " + ", ".join(loc_bits) + ".")

        commute_summaries = loc.get("commute_summaries") or []
        if commute_summaries:
            parts.append("Commute detail: " + "; ".join(str(s) for s in commute_summaries[:2]) + ".")

    return " ".join(parts)


def _build_tradeoff_section(breakdown: dict[str, float]) -> str:
    """Identify genuine weaknesses and phrase them honestly."""
    labels = {
        "review_rating": "reviews are not the strongest among the options shown",
        "amenity_match": "some requested amenities may be missing",
        "purpose_alignment": "it may not be perfectly set up for remote work or quiet focus",
        "neighborhood_fit": "the location or commute is a weaker point compared to other recommendations",
        "price_score": "the price is not the best fit relative to the stated budget",
        "google_maps_fit": "transit, food access, or commute came back weaker from live data",
        "stage_two_llm_fit": "in overall balance it ranked behind the top choices",
    }
    weak = [labels[k] for k, v in breakdown.items() if v < 0.55 and k in labels]
    if not weak:
        return ""
    return "One thing to keep in mind: " + "; ".join(weak) + "."


def _build_relaxation_section(relaxation_history: list[dict[str, Any]]) -> str:
    """Surface any search adjustments in plain language."""
    if not relaxation_history:
        return ""
    lines = []
    for item in relaxation_history:
        action = str(item.get("action") or "").replace("_", " ")
        change = item.get("change") or ""
        reason = item.get("reason") or "to find better results"
        if change:
            lines.append(f"The search {action}: {change} — {reason}.")
        else:
            lines.append(f"The search was adjusted ({action}) because {reason}.")
    return " ".join(lines)


# ── LLM rewrite ───────────────────────────────────────────────────────────────

def _rewrite_with_llm(
    fit_section: str,
    tradeoff_section: str,
    relaxation_section: str,
    user_query: str,
    review_snippets: str = "",
) -> str:
    """Use an LLM to turn structured sections into a clean, readable explanation."""
    if ChatOpenAI is None or not os.environ.get("OPENAI_API_KEY"):
        parts = [fit_section]
        if tradeoff_section:
            parts.append(tradeoff_section)
        if relaxation_section:
            parts.append(relaxation_section)
        return "\n\n".join(parts)

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        prompt = (
            "You are a straightforward apartment leasing assistant. "
            "Write a recommendation explanation for an average user — no jargon, no flowery language, no scores or numbers except the price.\n\n"
            f"The user asked for: {user_query}\n\n"
            "Use the structured notes below to write 2–4 short paragraphs in this order:\n\n"
            "  Paragraph 1 (always) — A quick baseline snapshot of the listing. "
            "Briefly cover: the neighborhood feel, overall guest experience quality (are reviews strong? is it clean and well-kept?), "
            "and transport access (subway, bus, commute) if that data is available. "
            "This paragraph should give the user a general sense of the place even if they didn't ask about these things.\n\n"
            "  Paragraph 2 (always) — How this listing specifically fits what the user asked for. "
            "Focus on the aspects explicitly mentioned in their query. Be direct and concrete.\n\n"
            "  Paragraph 3 (only if there are real tradeoffs) — Honest things to be aware of. "
            "Skip entirely if there are no meaningful concerns.\n\n"
            "  Paragraph 4 (only if the search was adjusted) — What changed during the search and why, in plain terms. "
            "Skip entirely if no adjustments were made.\n\n"
            "Rules:\n"
            "- Do not mention any scores, percentages, or numeric ratings\n"
            "- Do not use words like 'holistic', 'curated', 'seamlessly', 'boasts', 'nestled', 'vibrant'\n"
            "- Keep each paragraph to 2–4 sentences\n"
            "- The price is the only number you may include\n\n"
            f"Listing notes:\n{fit_section}\n\n"
            f"Tradeoff notes:\n{tradeoff_section or 'No significant tradeoffs.'}\n\n"
            f"Search adjustment notes:\n{relaxation_section or 'No adjustments were made.'}\n\n"
            + (f"Guest reviews (use these to extract real impressions about cleanliness, host quality, noise, etc.):\n{review_snippets}\n" if review_snippets else "")
        )
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception:
        parts = [fit_section]
        if tradeoff_section:
            parts.append(tradeoff_section)
        if relaxation_section:
            parts.append(relaxation_section)
        return "\n\n".join(parts)





# ── Public API ────────────────────────────────────────────────────────────────

def generate_listing_explanation(
    listing: dict[str, Any],
    hard_constraints: dict[str, Any],
    soft_preferences: dict[str, Any],
    relaxation_history: list[dict[str, Any]],
    user_query: str = "",
) -> str:
    """Generate a plain-language explanation for a single recommendation."""
    breakdown = {
        key: float(value) for key, value in (listing.get("score_breakdown") or {}).items()
    }

    fit = _build_fit_section(listing, soft_preferences, hard_constraints, breakdown)
    tradeoff = _build_tradeoff_section(breakdown)
    relaxation = _build_relaxation_section(relaxation_history)
    reviews = _extract_review_snippets(listing)

    return _rewrite_with_llm(fit, tradeoff, relaxation, user_query, review_snippets=reviews)


def generate_final_output(
    scored_listings: list[dict[str, Any]],
    hard_constraints: dict[str, Any],
    soft_preferences: dict[str, Any],
    relaxation_history: list[dict[str, Any]],
    top_k: int,
    user_query: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build final recommendation payloads and explanations."""
    recommendations = scored_listings[:top_k]
    explanations = [
        generate_listing_explanation(
            listing,
            hard_constraints=hard_constraints,
            soft_preferences=soft_preferences,
            relaxation_history=relaxation_history,
            user_query=user_query,
        )
        for listing in recommendations
    ]
    return recommendations, explanations
