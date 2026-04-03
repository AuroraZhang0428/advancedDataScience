"""Deterministic filtering and scoring logic for apartment recommendations."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from agent.config import DEFAULT_CONFIG, ScoringWeights
from agent.models import ScoreBreakdown


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numeric value into the closed interval [lower, upper]."""

    return max(lower, min(value, upper))


def _safe_float(value: Any) -> float | None:
    """Best-effort float conversion used by scoring helpers."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_hard_constraints(
    listings: list[dict[str, Any]],
    hard_constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply hard filtering rules to normalized listings."""

    filtered: list[dict[str, Any]] = []
    min_bedrooms = hard_constraints.get("min_bedrooms")
    min_bathrooms = hard_constraints.get("min_bathrooms")
    max_price = hard_constraints.get("max_price")
    room_type = hard_constraints.get("room_type")

    for listing in listings:
        bedrooms = _safe_float(listing.get("bedrooms"))
        bathrooms = _safe_float(listing.get("bathrooms"))
        price = _safe_float(listing.get("price"))
        listing_room_type = str(listing.get("raw", {}).get("room_type") or "")

        if min_bedrooms is not None and (bedrooms is None or bedrooms < float(min_bedrooms)):
            continue
        if min_bathrooms is not None and (bathrooms is None or bathrooms < float(min_bathrooms)):
            continue
        if max_price is not None and (price is None or price > float(max_price)):
            continue
        if room_type is not None and listing_room_type.lower() != str(room_type).lower():
            continue
        filtered.append(listing)

    return filtered


def compute_review_score(listing: dict[str, Any], soft_preferences: dict[str, Any]) -> float:
    """Score review quality, optionally honoring a desired threshold."""

    rating = _safe_float(listing.get("review_rating"))
    if rating is None:
        return 0.50

    base = _clip(rating / 5.0)
    desired_min = _safe_float(soft_preferences.get("review_min_rating"))
    if desired_min is None:
        return base
    if rating >= desired_min:
        return base
    return _clip(base * 0.75)


def compute_amenity_match(listing: dict[str, Any], soft_preferences: dict[str, Any]) -> float:
    """Score how well listing amenities align with desired amenities."""

    desired = [str(item).lower() for item in soft_preferences.get("desired_amenities", [])]
    if not desired:
        return 1.0

    listing_amenities = {str(item).lower() for item in listing.get("amenities", [])}
    matched = sum(1 for amenity in desired if amenity in listing_amenities)
    base = matched / max(len(desired), 1)

    strictness = _safe_float(soft_preferences.get("amenity_strictness")) or 1.0
    relaxed_floor = 1.0 - strictness
    return _clip((base * strictness) + relaxed_floor)


def compute_purpose_alignment(listing: dict[str, Any], soft_preferences: dict[str, Any]) -> float:
    """Score fit for higher-level user goals such as remote work."""

    if not soft_preferences.get("remote_work") and not soft_preferences.get("quiet_preference"):
        return 1.0

    signals: list[float] = []
    if soft_preferences.get("remote_work"):
        wifi = listing.get("wifi")
        workspace = listing.get("workspace")
        purpose_tags = {str(item).lower() for item in listing.get("purpose_tags", [])}
        signals.append(1.0 if wifi is True else 0.45 if wifi is None else 0.10)
        signals.append(1.0 if workspace is True else 0.50 if workspace is None else 0.20)
        signals.append(1.0 if "remote_work" in purpose_tags else 0.60)

    quiet_score = _safe_float(listing.get("quiet_score"))
    if soft_preferences.get("quiet_preference"):
        signals.append(quiet_score if quiet_score is not None else 0.55)

    if not signals:
        return 1.0
    return _clip(sum(signals) / len(signals))


def compute_neighborhood_score(listing: dict[str, Any], soft_preferences: dict[str, Any]) -> float:
    """Score location fit against preferred neighborhoods or areas."""

    preferences = [str(item).lower() for item in soft_preferences.get("preferred_neighborhoods", [])]
    if not preferences:
        return 1.0

    neighborhood = str(listing.get("neighborhood") or "").lower()
    neighborhood_group = str(listing.get("neighborhood_group") or "").lower()
    best = 0.20

    for preferred in preferences:
        if preferred == neighborhood:
            best = max(best, 1.0)
            continue
        if preferred in neighborhood or neighborhood in preferred:
            best = max(best, 0.90)
            continue
        if preferred == neighborhood_group or preferred in neighborhood_group:
            best = max(best, 0.75)
            continue

        similarity = SequenceMatcher(None, preferred, neighborhood).ratio()
        best = max(best, similarity * 0.70)

    if soft_preferences.get("expanded_neighborhood_search"):
        best = max(best, 0.65)

    return _clip(best)


def score_listing(
    listing: dict[str, Any],
    soft_preferences: dict[str, Any],
    weights: ScoringWeights | None = None,
) -> dict[str, Any]:
    """Compute a transparent weighted score and score breakdown for one listing."""

    effective_weights = weights or DEFAULT_CONFIG.scoring_weights
    breakdown = ScoreBreakdown(
        review_rating=compute_review_score(listing, soft_preferences),
        amenity_match=compute_amenity_match(listing, soft_preferences),
        purpose_alignment=compute_purpose_alignment(listing, soft_preferences),
        neighborhood_fit=compute_neighborhood_score(listing, soft_preferences),
    )
    breakdown_dict = breakdown.as_dict()
    weighted_sum = 0.0
    for component, weight in effective_weights.as_dict().items():
        weighted_sum += breakdown_dict[component] * weight

    scored_listing = dict(listing)
    scored_listing["score"] = round(_clip(weighted_sum), 4)
    scored_listing["score_breakdown"] = {
        key: round(value, 4) for key, value in breakdown_dict.items()
    }
    return scored_listing


def rank_listings(
    listings: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
    shortlist_size: int | None = None,
    weights: ScoringWeights | None = None,
) -> list[dict[str, Any]]:
    """Score and sort listings from best to worst."""

    scored = [score_listing(listing, soft_preferences, weights=weights) for listing in listings]
    scored.sort(
        key=lambda item: (
            item.get("score", 0.0),
            item.get("score_breakdown", {}).get("review_rating", 0.0),
            item.get("review_rating") or 0.0,
        ),
        reverse=True,
    )
    if shortlist_size is None:
        return scored
    return scored[:shortlist_size]


def count_good_results(scored_listings: list[dict[str, Any]], threshold: float) -> int:
    """Count results whose overall score clears the threshold."""

    return sum(1 for listing in scored_listings if float(listing.get("score", 0.0)) >= threshold)


def results_are_sufficient(
    scored_listings: list[dict[str, Any]],
    minimum_good_results: int | None = None,
    good_score_threshold: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Assess whether current results are strong enough to stop searching."""

    min_good = minimum_good_results or DEFAULT_CONFIG.minimum_good_results
    threshold = good_score_threshold or DEFAULT_CONFIG.good_score_threshold
    good_count = count_good_results(scored_listings, threshold)
    top_candidates = scored_listings[: DEFAULT_CONFIG.top_k_recommendations]
    diversity = len(
        {
            str(item.get("neighborhood") or item.get("neighborhood_group") or "unknown").lower()
            for item in top_candidates
        }
    )

    sufficient = good_count >= min_good and len(top_candidates) > 0
    diagnostics = {
        "good_result_count": good_count,
        "good_score_threshold": threshold,
        "minimum_good_results": min_good,
        "top_candidate_count": len(top_candidates),
        "top_neighborhood_diversity": diversity,
    }
    return sufficient, diagnostics
