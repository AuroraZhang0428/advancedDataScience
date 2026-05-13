"""Deterministic filtering and scoring logic for apartment recommendations."""

from __future__ import annotations

import json as _json
import os
from difflib import SequenceMatcher
from typing import Any

from agent.config import DEFAULT_CONFIG, ScoringWeights
from agent.services.neighborhoods import (
    compute_commute_score,
    compute_food_score,
    compute_transit_score,
    haversine_km,
    resolve_neighborhood_alias,
    resolve_place_reference,
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


# ---------------------------------------------------------------------------
# Review-phrase keyword lists for purpose signal extraction
# ---------------------------------------------------------------------------

_WIFI_POSITIVE = [
    "fast wifi", "great wifi", "good wifi", "strong wifi", "reliable wifi",
    "solid wifi", "speedy wifi", "excellent wifi", "wifi worked",
    "wifi was great", "wifi was fast", "wifi was good", "wifi was excellent",
    "fast internet", "good internet", "great internet", "reliable internet",
    "internet was fast", "internet was great",
]
_WIFI_NEGATIVE = [
    "slow wifi", "bad wifi", "no wifi", "poor wifi", "weak wifi",
    "wifi issues", "wifi problem", "wifi didn't work", "wifi not working",
    "wifi was slow", "wifi was bad", "wifi was terrible", "wifi was weak",
    "no internet", "slow internet", "internet issues", "internet problem",
    "internet was slow", "internet didn't work",
]

_WORKSPACE_POSITIVE = [
    "great desk", "good desk", "nice desk", "dedicated desk", "large desk",
    "great workspace", "good workspace", "nice workspace", "comfortable working",
    "perfect for work", "great for work", "ideal for work", "work-friendly",
    "worked from home", "worked remotely", "great for remote",
]
_WORKSPACE_NEGATIVE = [
    "no desk", "no workspace", "nowhere to work", "hard to work",
    "difficult to work", "not suitable for work", "not good for working",
    "no place to work",
]

_QUIET_POSITIVE = [
    "very quiet", "so quiet", "nice and quiet", "really quiet", "quite quiet",
    "quiet street", "quiet neighborhood", "quiet area", "quiet building",
    "surprisingly quiet", "peaceful", "tranquil", "serene", "calm",
]
_QUIET_NEGATIVE = [
    "very noisy", "so noisy", "really noisy", "too noisy",
    "very loud", "so loud", "too loud", "really loud",
    "street noise", "loud neighbors", "loud noise", "noise from",
    "could hear everything", "couldn't sleep", "lot of noise", "lots of noise",
]


def _review_purpose_signals(listing: dict[str, Any]) -> dict[str, float | None]:
    """Scan guest reviews for wifi, workspace, and quiet quality mentions.

    Returns a float signal in [0, 1] per category (1 = all positive mentions,
    0 = all negative) or None when no relevant phrases are found.
    Reviews are the ground truth that overrides listing-level claims.
    """
    raw_sample = listing.get("raw", {}).get("sample_reviews", "")
    records: list[dict] = []
    try:
        records = _json.loads(raw_sample) if raw_sample else []
    except (ValueError, TypeError):
        pass

    if records and isinstance(records, list):
        all_text = " ".join(r.get("text", "") for r in records if isinstance(r, dict)).lower()
    elif raw_sample and isinstance(raw_sample, str):
        all_text = raw_sample.lower()
    else:
        return {"wifi_review": None, "workspace_review": None, "quiet_review": None}

    def _signal(pos_phrases: list[str], neg_phrases: list[str]) -> float | None:
        pos = sum(1 for p in pos_phrases if p in all_text)
        neg = sum(1 for p in neg_phrases if p in all_text)
        if pos == 0 and neg == 0:
            return None
        return pos / (pos + neg)  # 1.0 = all positive, 0.0 = all negative

    return {
        "wifi_review": _signal(_WIFI_POSITIVE, _WIFI_NEGATIVE),
        "workspace_review": _signal(_WORKSPACE_POSITIVE, _WORKSPACE_NEGATIVE),
        "quiet_review": _signal(_QUIET_POSITIVE, _QUIET_NEGATIVE),
    }


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


def _haversine_to_neighborhood_score(dist_km: float) -> float:
    """Convert haversine distance (km) to a neighbourhood proximity score in [0, 1].

    Used as the geographic floor for listings when the preferred neighbourhood
    can be resolved to coordinates but does not match by name.
    """
    if dist_km <= 1.0:
        return 0.85
    if dist_km <= 2.5:
        return 0.65
    if dist_km <= 5.0:
        return 0.45
    if dist_km <= 9.0:
        return 0.25
    return 0.12


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


def _candidate_summary(listing: dict[str, Any], topics: list[str] | None = None) -> str:
    """Create a compact candidate summary for LLM reranking.

    When topics are supplied the same topic-relevance ranking used for the
    user-facing display is applied, so the LLM always scores on the same
    reviews the user will see (top 10: 3 shown + 7 more).
    """
    from agent.services.reviews import _comment_topic_score  # local import avoids circular deps

    amenities = ", ".join(str(item) for item in listing.get("amenities", [])[:6]) or "none"
    neighborhood = listing.get("neighborhood") or listing.get("neighborhood_group") or "Unknown area"
    price = listing.get("price")
    price_text = f"${float(price):,.0f} nightly" if price is not None else "price unavailable"
    bedrooms = listing.get("bedrooms")
    bathrooms = listing.get("bathrooms")
    review_rating = listing.get("review_rating")
    review_count = listing.get("raw", {}).get("number_of_reviews", 0)
    raw_sample = listing.get("raw", {}).get("sample_reviews", "")
    deterministic_score = float(listing.get("score", 0.0))
    scoring_weights_used = listing.get("scoring_weights_used", {})

    # Parse JSON review array; fall back to plain string for legacy rows
    records: list[dict] = []
    try:
        records = _json.loads(raw_sample) if raw_sample else []
    except (ValueError, TypeError):
        pass

    # For scoring we prioritise completeness over display aesthetics:
    # always include ALL critical reviews (up to 5) so the LLM never misses a
    # negative signal, then fill remaining slots with topic-ranked recent ones.
    # This differs slightly from the 2-recent+1-critical display layout, which
    # is fine — scoring needs to be comprehensive, display needs to be readable.
    _SCORING_LIMIT = 10
    recent = [r for r in records if not r.get("critical")]
    critical_recs = [r for r in records if r.get("critical")]

    if topics and recent:
        recent = sorted(
            recent,
            key=lambda r: _comment_topic_score(r.get("text", ""), topics),
            reverse=True,
        )

    remaining_slots = max(0, _SCORING_LIMIT - len(critical_recs))
    display_slice = critical_recs + recent[:remaining_slots]

    review_text_parts: list[str] = []
    for r in display_slice:
        tag = "[critical]" if r.get("critical") else "[recent]"
        line = f"{tag} [{r.get('date','')}] {r.get('name','')}: {r.get('text','').replace(chr(10), ' ')}"
        review_text_parts.append(line)

    if not review_text_parts and raw_sample and isinstance(raw_sample, str):
        review_text_parts = [raw_sample.replace("\n", " ")[:600]]

    reviews_block = "\n".join(review_text_parts) if review_text_parts else "No reviews available."

    return (
        f"id={listing.get('id')} | title={listing.get('title', 'Untitled')} | neighborhood={neighborhood} | "
        f"price={price_text} | bedrooms={bedrooms} | bathrooms={bathrooms} | "
        f"review_rating={review_rating} ({review_count} reviews) | "
        f"wifi={listing.get('wifi')} | workspace={listing.get('workspace')} | quiet_score={listing.get('quiet_score')} | "
        f"purpose_tags={listing.get('purpose_tags', [])} | amenities={amenities} | "
        f"coarse_retrieval_score={deterministic_score:.2f} | weights={scoring_weights_used}\n"
        f"reviews:\n{reviews_block}"
    )


def _rerank_with_llm(
    candidates: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
    topics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Rerank top candidates using an LLM when credentials are available."""

    if not candidates:
        return []

    _require_llm_ranking()

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(RankingResponse)
        candidate_block = "\n".join(_candidate_summary(candidate, topics=topics) for candidate in candidates)
        prompt = (
            "You are evaluating apartment listing candidates for the user's true intent.\n"
            "The coarse retrieval score was only used to gather a shortlist. Do not follow it mechanically.\n"
            "Directly judge each listing from the user preferences and listing facts.\n"
            "Treat any stated budget as a nightly target/ceiling unless explicitly monthly.\n"
            "Return all candidate ids sorted best to worst with fit_score values between 0.0 and 1.0.\n"
            "Also return component_scores for review_rating, amenity_match, purpose_alignment, neighborhood_fit, and price_score.\n"
            "Use the user's priority_weights as guidance for what matters most, but do not compute a rigid weighted average.\n\n"
            "For purpose_alignment specifically: read the guest reviews carefully for mentions of wifi speed/reliability, "
            "desk or workspace quality, and noise/quiet levels. Reviews are ground truth — a listing that claims wifi "
            "but has reviews saying 'wifi was terrible' or 'couldn't work here' should score low on purpose_alignment "
            "for remote-work queries. Conversely, positive review mentions of fast wifi, great workspace, or peaceful "
            "quiet should boost the score even if the listing title doesn't emphasize it.\n\n"
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
    min_guests = hard_constraints.get("min_guests")
    min_bedrooms = hard_constraints.get("min_bedrooms")
    min_bathrooms = hard_constraints.get("min_bathrooms")
    max_price = _effective_nightly_budget(hard_constraints)
    room_type = hard_constraints.get("room_type")

    for listing in listings:
        accommodates = _safe_float(listing.get("accommodates"))
        bedrooms = _safe_float(listing.get("bedrooms"))
        bathrooms = _safe_float(listing.get("bathrooms"))
        price = _safe_float(listing.get("price"))
        listing_room_type = str(listing.get("raw", {}).get("room_type") or "")

        # Only apply min_guests when the listing has guest-capacity data.
        # If accommodates is unknown we cannot say it fails — skip the check.
        if min_guests is not None and accommodates is not None and accommodates < float(min_guests):
            continue
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
    
    # Factor in review volume for confidence
    num_reviews = _safe_float(listing.get("raw", {}).get("number_of_reviews")) or 0.0
    if num_reviews > 50:
        confidence = 1.0
    elif num_reviews > 10:
        confidence = 0.95
    elif num_reviews > 0:
        confidence = 0.90
    else:
        confidence = 0.85
        
    base = _clip(base * confidence)

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
    """Score fit for higher-level user goals such as remote work.

    Review-derived signals are the primary source (70%) when guest reviews
    mention wifi, workspace, or quiet quality. Structural signals (listing
    title, amenities, explicit fields) contribute 30% — acting as a baseline
    when reviews are silent on a topic.
    """
    if not soft_preferences.get("remote_work") and not soft_preferences.get("quiet_preference"):
        return None

    # Extract review-based signals once; used for both remote_work and quiet branches.
    review_signals = _review_purpose_signals(listing)

    signals: list[float] = []
    if soft_preferences.get("remote_work"):
        wifi = listing.get("wifi")
        workspace = listing.get("workspace")
        purpose_tags = {str(item).lower() for item in listing.get("purpose_tags", [])}

        # Structural baseline (listing title / amenities / explicit column)
        wifi_struct = 1.0 if wifi is True else 0.45 if wifi is None else 0.10
        workspace_struct = 1.0 if workspace is True else 0.50 if workspace is None else 0.20

        # Blend with review signal when available (30% structural, 70% review).
        # Reviews are the primary source — guest experience outweighs listing claims.
        wifi_review = review_signals.get("wifi_review")
        workspace_review = review_signals.get("workspace_review")
        wifi_signal = (0.3 * wifi_struct + 0.7 * wifi_review) if wifi_review is not None else wifi_struct
        workspace_signal = (0.3 * workspace_struct + 0.7 * workspace_review) if workspace_review is not None else workspace_struct

        signals.append(wifi_signal)
        signals.append(workspace_signal)
        signals.append(1.0 if "remote_work" in purpose_tags else 0.60)

    if soft_preferences.get("quiet_preference"):
        quiet_score = _safe_float(listing.get("quiet_score"))
        quiet_review = review_signals.get("quiet_review")
        if quiet_score is not None and quiet_review is not None:
            signals.append(0.3 * quiet_score + 0.7 * quiet_review)
        elif quiet_review is not None:
            signals.append(quiet_review)
        else:
            signals.append(quiet_score if quiet_score is not None else 0.55)

    if not signals:
        return 1.0
    return _clip(sum(signals) / len(signals))


def compute_neighborhood_score(listing: dict[str, Any], soft_preferences: dict[str, Any]) -> float | None:
    """Score location fit against neighborhood, commute, transit, and food preferences.

    Three layers of intelligence are applied:

    Fix 1 – Live transit/food scores: when the listing has been enriched by Google
    Maps (``location_context["google_maps_enriched"] == True``), the real scores
    computed from nearby Places API results are used instead of the static lookup
    tables in ``neighborhoods.py``.

    Fix 2 – Alias-resolved neighborhood matching: before running SequenceMatcher,
    the preferred neighbourhood string is normalised through ``NEIGHBORHOOD_ALIASES``
    (e.g. "central park" → "upper west side", "les" → "lower east side") so that
    colloquial names and landmarks map to the right canonical neighbourhood.

    Fix 3 – Haversine distance floor: when listing coordinates are available and the
    preferred neighbourhood can be resolved to a centroid via
    ``resolve_place_reference()``, the flat 0.20 floor is replaced by a proximity
    score derived from the straight-line distance (km). A listing two blocks away
    earns a high floor; one across the borough earns a low one.
    """

    preferences = [str(item).lower() for item in soft_preferences.get("preferred_neighborhoods", [])]
    explicit_neighborhood_score: float | None = None

    neighborhood = str(listing.get("neighborhood") or "").lower()
    neighborhood_group = str(listing.get("neighborhood_group") or "").lower()

    if preferences:
        listing_lat = listing.get("latitude")
        listing_lon = listing.get("longitude")
        best = 0.20  # default floor; overridden by haversine when coordinates available

        for preferred in preferences:
            # Fix 2: Resolve colloquial name / landmark to canonical neighbourhood.
            canonical = resolve_neighborhood_alias(preferred)

            # Fix 3: Geographic distance floor.
            if listing_lat is not None and listing_lon is not None:
                centroid = resolve_place_reference(preferred)
                if centroid and centroid.get("latitude") and centroid.get("longitude"):
                    dist_km = haversine_km(
                        float(listing_lat), float(listing_lon),
                        float(centroid["latitude"]), float(centroid["longitude"]),
                    )
                    best = max(best, _haversine_to_neighborhood_score(dist_km))

            # Name-level matching: check both the original label and the canonical alias.
            for check in ([preferred] if canonical == preferred else [preferred, canonical]):
                if check == neighborhood:
                    best = max(best, 1.0)
                elif check in neighborhood or neighborhood in check:
                    best = max(best, 0.90)
                elif check == neighborhood_group or check in neighborhood_group:
                    best = max(best, 0.75)

            # Fuzzy fallback on the canonical form (character similarity is more meaningful
            # when both sides are proper neighbourhood names rather than landmarks).
            similarity = SequenceMatcher(None, canonical, neighborhood).ratio()
            best = max(best, similarity * 0.70)

        if soft_preferences.get("expanded_neighborhood_search"):
            best = max(best, 0.65)
        explicit_neighborhood_score = _clip(best)

    # Read LLM-assigned sub-weights; fall back to defaults if not present.
    sub_w = soft_preferences.get("location_sub_weights") or {}
    w_neighborhood = float(sub_w.get("neighborhood_match", 0.40))
    w_commute = float(sub_w.get("commute", 0.30))
    w_transit = float(sub_w.get("transit", 0.15))
    w_food = float(sub_w.get("food_scene", 0.15))

    component_scores: list[tuple[float, float]] = []
    if explicit_neighborhood_score is not None:
        component_scores.append((explicit_neighborhood_score, w_neighborhood))

    commute_destinations = [str(item) for item in soft_preferences.get("commute_destinations", []) if str(item).strip()]
    commute_score = compute_commute_score(listing, commute_destinations)
    if commute_score is not None:
        component_scores.append((commute_score, w_commute))

    # Fix 1: Prefer live Google Maps scores over static lookup tables when available.
    location_ctx = listing.get("location_context") or {}
    gm_enriched = bool(location_ctx.get("google_maps_enriched"))

    if soft_preferences.get("transit_priority"):
        if gm_enriched and location_ctx.get("transit_score") is not None:
            transit_s = float(location_ctx["transit_score"])
        else:
            transit_s = compute_transit_score(listing)
        component_scores.append((transit_s, w_transit))

    if soft_preferences.get("food_scene_priority"):
        if gm_enriched and location_ctx.get("food_score") is not None:
            food_s = float(location_ctx["food_score"])
        else:
            food_s = compute_food_score(listing)
        component_scores.append((food_s, w_food))

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
    user_query: str = "",
) -> list[dict[str, Any]]:
    """Score listings, then optionally rerank the top candidates with an LLM.

    ``user_query`` is used to detect review topics so the LLM reranker sees
    the same topic-ranked review slice (top 10) that the user-facing display
    will show — keeping the two views consistent.
    """
    from agent.services.reviews import detect_topics  # local import avoids circular deps

    topics = detect_topics(user_query, soft_preferences)

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

    # ── Optional LLM reranking ───────────────────────────────────────────────
    if _llm_is_available():
        try:
            llm_reranked = _rerank_with_llm(
                candidates=shortlist,
                soft_preferences=soft_preferences,
                hard_constraints=hard_constraints,
                topics=topics,
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
        except Exception:
            pass  # fall through to deterministic sort below

    return shortlist



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
