"""Deterministic filtering and scoring logic for apartment recommendations."""

from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import Any

from agent.config import DEFAULT_CONFIG, ScoringWeights
from agent.services.neighborhoods import (
    compute_commute_score,
    compute_food_score,
    compute_transit_score,
)

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
    class ComponentScores(BaseModel):
        review_rating: float = Field(description="Review-quality fit from 0.0 to 1.0.")
        amenity_match: float = Field(description="Amenity fit from 0.0 to 1.0.")
        purpose_alignment: float = Field(description="Usage or lifestyle fit from 0.0 to 1.0.")
        neighborhood_fit: float = Field(description="Neighborhood and commute fit from 0.0 to 1.0.")
        price_score: float = Field(description="Price fit from 0.0 to 1.0.")


    class RankedCandidate(BaseModel):
        id: str = Field(description="Listing id from the provided candidates.")
        fit_score: float = Field(description="Overall fit score from 0.0 to 1.0.")
        component_scores: ComponentScores = Field(
            description="Component scores reflecting how this listing fits the user's priorities.",
        )
        reason: str = Field(description="Short reason this listing was placed here.")


    class RankingResponse(BaseModel):
        ranked_candidates: list[RankedCandidate] = Field(
            default_factory=list,
            description="Candidates sorted best to worst.",
        )


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


def _effective_nightly_budget(hard_constraints: dict[str, Any]) -> float | None:
    """Normalize the user's stated budget into the dataset's nightly price units."""

    budget = _safe_float(hard_constraints.get("max_price"))
    if budget is None:
        return None

    price_period = str(hard_constraints.get("price_period") or "nightly").lower()
    if price_period == "monthly":
        return budget / 30.0
    return budget


def _effective_target_price(soft_preferences: dict[str, Any], hard_constraints: dict[str, Any]) -> float | None:
    """Normalize the user's desired target price into the dataset's nightly price units."""

    target = _safe_float(soft_preferences.get("target_price"))
    if target is None:
        return None

    price_period = str(hard_constraints.get("price_period") or "nightly").lower()
    if price_period == "monthly":
        return target / 30.0
    return target


def _effective_price_floor(soft_preferences: dict[str, Any], hard_constraints: dict[str, Any]) -> float | None:
    """Normalize the user's desired price floor into the dataset's nightly price units."""

    floor = _safe_float(soft_preferences.get("price_floor"))
    if floor is None:
        return None

    price_period = str(hard_constraints.get("price_period") or "nightly").lower()
    if price_period == "monthly":
        return floor / 30.0
    return floor


def resolve_scoring_weights(
    soft_preferences: dict[str, Any],
    fallback: ScoringWeights | None = None,
) -> ScoringWeights:
    """Resolve query-specific scoring weights inferred from the user's priorities."""

    fallback_weights = fallback or DEFAULT_CONFIG.scoring_weights
    raw_weights = dict(soft_preferences.get("priority_weights") or {})
    if not raw_weights:
        return fallback_weights

    keys = list(fallback_weights.as_dict().keys())
    cleaned: dict[str, float] = {}
    total = 0.0
    for key in keys:
        raw_value = raw_weights.get(key, fallback_weights.as_dict()[key])
        try:
            numeric = max(float(raw_value), 0.0)
        except (TypeError, ValueError):
            numeric = fallback_weights.as_dict()[key]
        cleaned[key] = numeric
        total += numeric

    if total <= 0:
        return fallback_weights

    normalized = {key: cleaned[key] / total for key in keys}
    return ScoringWeights(
        review_rating=normalized["review_rating"],
        amenity_match=normalized["amenity_match"],
        purpose_alignment=normalized["purpose_alignment"],
        neighborhood_fit=normalized["neighborhood_fit"],
        price_score=normalized["price_score"],
    )


def _normalize_active_weights(
    effective_weights: ScoringWeights,
    active_components: set[str],
) -> dict[str, float]:
    """Normalize query-specific weights over only the components that truly apply."""

    all_weights = effective_weights.as_dict()
    if not active_components:
        return {}

    active_total = sum(all_weights.get(component, 0.0) for component in active_components)
    if active_total <= 0:
        uniform_weight = 1.0 / len(active_components)
        return {component: uniform_weight for component in active_components}

    return {
        component: all_weights[component] / active_total
        for component in all_weights
        if component in active_components
    }


def _llm_is_available() -> bool:
    """Return whether the ranking pipeline can call the OpenAI LLM."""

    return HAS_LLM and ChatOpenAI is not None and bool(os.environ.get("OPENAI_API_KEY"))


def _require_llm_ranking() -> None:
    """Ensure the OpenAI-backed ranking path is available."""

    if not _llm_is_available():
        raise RuntimeError("OPENAI_API_KEY is required because deterministic LLM-ranking fallback has been removed.")


def _candidate_summary(listing: dict[str, Any]) -> str:
    """Create a compact candidate summary for LLM reranking."""

    amenities = ", ".join(str(item) for item in listing.get("amenities", [])[:6]) or "none"
    neighborhood = listing.get("neighborhood") or listing.get("neighborhood_group") or "Unknown area"
    price = listing.get("price")
    price_text = f"${float(price):,.0f} nightly" if price is not None else "price unavailable"
    bedrooms = listing.get("bedrooms")
    bathrooms = listing.get("bathrooms")
    review_rating = listing.get("review_rating")
    deterministic_score = float(listing.get("score", 0.0))
    scoring_weights_used = listing.get("scoring_weights_used", {})
    return (
        f"id={listing.get('id')} | title={listing.get('title', 'Untitled')} | neighborhood={neighborhood} | "
        f"price={price_text} | bedrooms={bedrooms} | bathrooms={bathrooms} | review_rating={review_rating} | "
        f"wifi={listing.get('wifi')} | workspace={listing.get('workspace')} | quiet_score={listing.get('quiet_score')} | "
        f"purpose_tags={listing.get('purpose_tags', [])} | amenities={amenities} | "
        f"coarse_retrieval_score={deterministic_score:.2f} | weights={scoring_weights_used}"
    )


def _rerank_with_llm(
    candidates: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rerank top candidates using an LLM when credentials are available."""

    if not candidates:
        return []

    _require_llm_ranking()

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(RankingResponse)
        candidate_block = "\n".join(_candidate_summary(candidate) for candidate in candidates)
        prompt = (
            "You are evaluating apartment listing candidates for the user's true intent.\n"
            "The coarse retrieval score was only used to gather a shortlist. Do not follow it mechanically.\n"
            "Directly judge each listing from the user preferences and listing facts.\n"
            "Treat any stated budget as a nightly target/ceiling unless explicitly monthly.\n"
            "Return all candidate ids sorted best to worst with fit_score values between 0.0 and 1.0.\n"
            "Also return component_scores for review_rating, amenity_match, purpose_alignment, neighborhood_fit, and price_score.\n"
            "Use the user's priority_weights as guidance for what matters most, but do not compute a rigid weighted average.\n\n"
            f"Hard constraints:\n{hard_constraints}\n\n"
            f"Soft preferences:\n{soft_preferences}\n\n"
            f"Candidates:\n{candidate_block}\n"
        )
        response = structured_llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(f"OpenAI-backed ranking failed: {exc}") from exc

    candidate_map = {str(candidate.get("id")): dict(candidate) for candidate in candidates}
    reranked: list[dict[str, Any]] = []

    for ranked_candidate in response.ranked_candidates:
        listing = candidate_map.pop(str(ranked_candidate.id), None)
        if listing is None:
            continue
        coarse_score = float(listing.get("score", 0.0))
        llm_fit_score = _clip(float(ranked_candidate.fit_score))
        listing["coarse_score"] = round(coarse_score, 4)
        listing["llm_fit_score"] = round(llm_fit_score, 4)
        listing["llm_rank_reason"] = ranked_candidate.reason.strip()
        listing["score"] = round(llm_fit_score, 4)
        listing["score_breakdown"] = {
            "review_rating": round(_clip(float(ranked_candidate.component_scores.review_rating)), 4),
            "amenity_match": round(_clip(float(ranked_candidate.component_scores.amenity_match)), 4),
            "purpose_alignment": round(_clip(float(ranked_candidate.component_scores.purpose_alignment)), 4),
            "neighborhood_fit": round(_clip(float(ranked_candidate.component_scores.neighborhood_fit)), 4),
            "price_score": round(_clip(float(ranked_candidate.component_scores.price_score)), 4),
            "llm_fit": round(llm_fit_score, 4),
        }
        reranked.append(listing)

    leftovers = sorted(
        candidate_map.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    reranked.extend(leftovers)
    return reranked


def filter_hard_constraints(
    listings: list[dict[str, Any]],
    hard_constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply hard filtering rules to normalized listings."""

    filtered: list[dict[str, Any]] = []
    min_bedrooms = hard_constraints.get("min_bedrooms")
    min_bathrooms = hard_constraints.get("min_bathrooms")
    max_price = _effective_nightly_budget(hard_constraints)
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


def compute_amenity_match(listing: dict[str, Any], soft_preferences: dict[str, Any]) -> float | None:
    """Score how well listing amenities align with desired amenities."""

    desired = [str(item).lower() for item in soft_preferences.get("desired_amenities", [])]
    if not desired:
        return None

    listing_amenities = {str(item).lower() for item in listing.get("amenities", [])}
    matched = sum(1 for amenity in desired if amenity in listing_amenities)
    base = matched / max(len(desired), 1)

    strictness = _safe_float(soft_preferences.get("amenity_strictness")) or 1.0
    relaxed_floor = 1.0 - strictness
    return _clip((base * strictness) + relaxed_floor)


def compute_purpose_alignment(listing: dict[str, Any], soft_preferences: dict[str, Any]) -> float | None:
    """Score fit for higher-level user goals such as remote work."""

    if not soft_preferences.get("remote_work") and not soft_preferences.get("quiet_preference"):
        return None

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


def compute_neighborhood_score(listing: dict[str, Any], soft_preferences: dict[str, Any]) -> float | None:
    """Score location fit against neighborhood, commute, transit, and food preferences."""

    preferences = [str(item).lower() for item in soft_preferences.get("preferred_neighborhoods", [])]
    explicit_neighborhood_score: float | None = None

    neighborhood = str(listing.get("neighborhood") or "").lower()
    neighborhood_group = str(listing.get("neighborhood_group") or "").lower()
    if preferences:
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
        explicit_neighborhood_score = _clip(best)

    component_scores: list[tuple[float, float]] = []
    if explicit_neighborhood_score is not None:
        component_scores.append((explicit_neighborhood_score, 0.40))

    commute_destinations = [str(item) for item in soft_preferences.get("commute_destinations", []) if str(item).strip()]
    commute_score = compute_commute_score(listing, commute_destinations)
    if commute_score is not None:
        component_scores.append((commute_score, 0.30))

    if soft_preferences.get("transit_priority"):
        component_scores.append((compute_transit_score(listing), 0.15))

    if soft_preferences.get("food_scene_priority"):
        component_scores.append((compute_food_score(listing), 0.15))

    if not component_scores:
        return None

    total_weight = sum(weight for _, weight in component_scores)
    weighted_score = sum(score * weight for score, weight in component_scores) / total_weight
    return _clip(weighted_score)


def compute_price_score(
    listing: dict[str, Any],
    hard_constraints: dict[str, Any],
    soft_preferences: dict[str, Any] | None = None,
) -> float | None:
    """
    Computes a price score (0 to 1) comparing price versus budget.
    We apply a simple heuristic, incorporating qualitative price preferences.
    """
    price = _safe_float(listing.get("price"))
    soft_preferences = soft_preferences or {}
    price_preference = hard_constraints.get("price_preference", "none")
    budget = _effective_nightly_budget(hard_constraints)
    target_price = _effective_target_price(soft_preferences, hard_constraints)
    price_floor = _effective_price_floor(soft_preferences, hard_constraints)
    price_is_applicable = (
        target_price is not None
        or price_floor is not None
        or budget is not None
        or str(price_preference).lower() in {"cheap", "expensive", "moderate"}
    )
    if not price_is_applicable:
        return None

    if price is None or price <= 0:
        return 0.5

    score: float = 0.5

    def apply_floor_softness(base_score: float) -> float:
        """Softly penalize prices that fall below the user's stated floor."""

        if price_floor is None or price_floor <= 0:
            return _clip(base_score)
        if price >= price_floor:
            return _clip(base_score)

        floor_ratio = price / price_floor
        softened_penalty = max(0.35, floor_ratio)
        return _clip(base_score * softened_penalty)

    if target_price is not None and target_price > 0:
        distance_ratio = abs(price - target_price) / target_price
        base_score = max(0.0, 1.0 - min(distance_ratio, 1.0))

        if price_preference == "cheap" and price <= target_price:
            base_score = min(1.0, base_score + 0.08)
        elif price_preference == "expensive" and price >= target_price:
            base_score = min(1.0, base_score + 0.08)

        return apply_floor_softness(base_score)
    
    if budget is not None and budget > 0:
        if price <= budget:
            # When the user gives a price, treat it as the nightly ceiling/target.
            # This avoids ultra-cheap outliers dominating results simply because they are lowest.
            ratio = price / budget
            base_score = 0.70 + (0.30 * ratio)
        else:
            ratio = price / budget
            base_score = max(0.0, 1.5 - ratio)  # drops to 0 at 1.5x budget
            
        if price_preference == "cheap":
            under_budget_bonus = max(0.0, (budget - price) / budget)
            base_score += under_budget_bonus * 0.10
        elif price_preference == "expensive" and price <= budget:
            ratio = price / budget
            base_score = min(1.0, 0.7 + ratio * 0.3)
            
        score = apply_floor_softness(base_score)
    elif price_floor is not None and price_floor > 0:
        if price >= price_floor:
            above_floor_ratio = min((price - price_floor) / price_floor, 1.0)
            score = 0.8 + (0.2 * above_floor_ratio)
        else:
            floor_ratio = price / price_floor
            score = 0.35 + (0.45 * floor_ratio)
    else:
        if price_preference == "cheap":
            # Heavily favor lower prices, drop quickly
            normalized = price / 300.0
            score = 1.0 - min(normalized, 1.0)
        elif price_preference == "expensive":
            # Favor luxury or higher prices. Map $0 to $1000 range positively
            normalized = price / 1000.0
            score = min(normalized, 1.0)
            # Give a slight bump so anything over $500 gets a good score
            if price > 500:
                score = max(score, 0.8)
        elif price_preference == "moderate":
            # Best score in the middle (e.g. $150-$300)
            diff = abs(price - 200.0)
            score = 1.0 - min(diff / 400.0, 1.0)
        else:
            # Default distribution for NYC, slight preference for cheaper
            normalized = price / 500.0  # arbitrary normalization factor
            score = 1.0 - min(normalized, 1.0)
        
    return _clip(score)


def score_listing(
    listing: dict[str, Any],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
    weights: ScoringWeights | None = None,
) -> dict[str, Any]:
    """Compute a transparent weighted score and score breakdown for one listing."""

    effective_weights = weights or resolve_scoring_weights(soft_preferences, fallback=DEFAULT_CONFIG.scoring_weights)
    raw_component_scores: dict[str, float | None] = {
        "review_rating": compute_review_score(listing, soft_preferences),
        "amenity_match": compute_amenity_match(listing, soft_preferences),
        "purpose_alignment": compute_purpose_alignment(listing, soft_preferences),
        "neighborhood_fit": compute_neighborhood_score(listing, soft_preferences),
        "price_score": compute_price_score(listing, hard_constraints, soft_preferences=soft_preferences),
    }
    active_components = {
        component for component, value in raw_component_scores.items() if value is not None
    }
    active_weights = _normalize_active_weights(effective_weights, active_components)
    weighted_sum = sum(
        float(raw_component_scores[component]) * active_weights[component]
        for component in active_weights
    )

    scored_listing = dict(listing)
    scored_listing["score"] = round(_clip(weighted_sum), 4)
    scored_listing["score_breakdown"] = {
        key: round(float(value), 4)
        for key, value in raw_component_scores.items()
        if value is not None
    }
    scored_listing["scoring_weights_used"] = {
        key: round(value, 4) for key, value in active_weights.items()
    }
    scored_listing["active_retrieval_components"] = sorted(active_components)
    return scored_listing


def rank_listings(
    listings: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
    shortlist_size: int | None = None,
    weights: ScoringWeights | None = None,
) -> list[dict[str, Any]]:
    """Score listings, then optionally rerank the top candidates with an LLM."""

    scored = [score_listing(listing, soft_preferences, hard_constraints, weights=weights) for listing in listings]
    scored.sort(
        key=lambda item: (
            item.get("score", 0.0),
            item.get("score_breakdown", {}).get("review_rating", 0.0),
            item.get("review_rating") or 0.0,
        ),
        reverse=True,
    )

    if shortlist_size is None:
        shortlist = list(scored)
    else:
        shortlist = scored[:shortlist_size]

    llm_reranked = _rerank_with_llm(
        candidates=shortlist,
        soft_preferences=soft_preferences,
        hard_constraints=hard_constraints,
    )
    llm_reranked.sort(
        key=lambda item: (
            item.get("score", 0.0),
            item.get("llm_fit_score", 0.0),
            item.get("coarse_score", 0.0),
        ),
        reverse=True,
    )
    return llm_reranked


def count_good_results(scored_listings: list[dict[str, Any]], threshold: float) -> int:
    """Count results whose overall score clears the threshold."""

    return sum(1 for listing in scored_listings if float(listing.get("score", 0.0)) >= threshold)


def results_are_sufficient(
    scored_listings: list[dict[str, Any]],
    hard_constraints: dict[str, Any] | None = None,
    soft_preferences: dict[str, Any] | None = None,
    minimum_good_results: int | None = None,
    good_score_threshold: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Assess whether current results are strong enough to stop searching."""

    hard_constraints = hard_constraints or {}
    soft_preferences = soft_preferences or {}
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

    top_price_scores = [
        float((item.get("score_breakdown") or {}).get("price_score", 0.0))
        for item in top_candidates
    ]
    best_price_score = max(top_price_scores) if top_price_scores else 0.0
    target_price = _effective_target_price(soft_preferences, hard_constraints)
    price_floor = _effective_price_floor(soft_preferences, hard_constraints)
    budget = _effective_nightly_budget(hard_constraints)

    target_price_fit_poor = target_price is not None and best_price_score < 0.35
    price_floor_fit_poor = price_floor is not None and best_price_score < 0.35
    budget_fit_poor = budget is not None and len(top_candidates) == 0

    sufficient = (
        good_count >= min_good
        and len(top_candidates) > 0
        and not target_price_fit_poor
        and not price_floor_fit_poor
        and not budget_fit_poor
    )
    diagnostics = {
        "good_result_count": good_count,
        "good_score_threshold": threshold,
        "minimum_good_results": min_good,
        "top_candidate_count": len(top_candidates),
        "top_neighborhood_diversity": diversity,
        "best_price_score": round(best_price_score, 4),
        "target_price_fit_poor": target_price_fit_poor,
        "price_floor_fit_poor": price_floor_fit_poor,
        "budget_fit_poor": budget_fit_poor,
        "price_floor": price_floor,
        "target_price": target_price,
        "max_price": budget,
    }
    return sufficient, diagnostics
