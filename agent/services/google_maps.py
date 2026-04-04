"""Google Maps enrichment for shortlisted apartment candidates."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

try:
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI

    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    BaseModel = None
    Field = None
    ChatOpenAI = None


PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
ROUTES_COMPUTE_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

TRANSIT_TYPES = ["subway_station", "train_station", "transit_station", "bus_station"]
FOOD_TYPES = ["restaurant", "cafe", "bakery", "meal_takeaway"]
GROCERY_TYPES = ["supermarket", "grocery_store", "convenience_store"]
TRANSIT_TYPE_TO_MODE = {
    "subway_station": "subway",
    "train_station": "train",
    "bus_station": "bus",
    "transit_station": "transit_hub",
}


if HAS_LLM:
    class EnrichedRankedCandidate(BaseModel):
        """Structured ranking output for enriched listing reranking."""

        id: str = Field(description="Listing id from the candidate set.")
        fit_score: float = Field(description="Holistic fit score from 0.0 to 1.0.")
        reason: str = Field(description="Short ranking justification grounded in the provided facts.")


    class EnrichedRankingResponse(BaseModel):
        """Structured response containing all enriched candidate rankings."""

        ranked_candidates: list[EnrichedRankedCandidate] = Field(
            default_factory=list,
            description="Candidates sorted best to worst.",
        )


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a value into the closed interval [lower, upper]."""

    return max(lower, min(value, upper))


def _safe_float(value: Any) -> float | None:
    """Best-effort float conversion."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_duration_minutes(duration_text: str | None) -> float | None:
    """Convert Google route duration text like '1840s' into minutes."""

    if not duration_text or not duration_text.endswith("s"):
        return None
    try:
        seconds = float(duration_text[:-1])
    except ValueError:
        return None
    return round(seconds / 60.0, 1)


def google_maps_available() -> bool:
    """Return whether Google Maps enrichment can run."""

    return bool(os.environ.get("GOOGLE_MAPS_API_KEY"))


def _post_json(url: str, payload: dict[str, Any], field_mask: str) -> dict[str, Any]:
    """Send a JSON POST request to a Google Maps Platform endpoint."""

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set.")

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": field_mask,
        },
    )
    with request.urlopen(req, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _search_text_place(query: str) -> dict[str, Any] | None:
    """Resolve a commute anchor or place reference into a single place result."""

    payload = {
        "textQuery": query,
        "pageSize": 1,
        "languageCode": "en-US",
        "regionCode": "US",
    }
    response = _post_json(
        PLACES_TEXT_SEARCH_URL,
        payload,
        "places.id,places.displayName,places.formattedAddress,places.location",
    )
    places = response.get("places") or []
    return places[0] if places else None


def _search_nearby(
    latitude: float,
    longitude: float,
    included_types: list[str],
    radius_meters: float,
    max_result_count: int,
) -> list[dict[str, Any]]:
    """Find nearby POIs of a certain type around a listing."""

    payload = {
        "includedTypes": included_types,
        "maxResultCount": max_result_count,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius_meters,
            }
        },
    }
    response = _post_json(
        PLACES_NEARBY_SEARCH_URL,
        payload,
        "places.displayName,places.primaryType,places.formattedAddress,places.location",
    )
    return list(response.get("places") or [])


def _compute_commute_minutes(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    travel_mode: str,
) -> float | None:
    """Compute point-to-point route duration in minutes."""

    payload = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": origin_latitude,
                    "longitude": origin_longitude,
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": destination_latitude,
                    "longitude": destination_longitude,
                }
            }
        },
        "travelMode": travel_mode,
        "computeAlternativeRoutes": False,
        "languageCode": "en-US",
        "units": "IMPERIAL",
    }
    response = _post_json(
        ROUTES_COMPUTE_URL,
        payload,
        "routes.duration,routes.distanceMeters",
    )
    routes = response.get("routes") or []
    if not routes:
        return None
    return _parse_duration_minutes(routes[0].get("duration"))


def _collect_place_names(places: list[dict[str, Any]], limit: int = 3) -> list[str]:
    """Collect readable place names from Google responses."""

    names: list[str] = []
    for place in places[:limit]:
        display_name = place.get("displayName") or {}
        text = display_name.get("text") if isinstance(display_name, dict) else None
        if text:
            names.append(str(text))
    return names


def _classify_transit_places(places: list[dict[str, Any]]) -> dict[str, Any]:
    """Break nearby transit results into mode-specific counts and examples."""

    counts = {"subway": 0, "train": 0, "bus": 0, "transit_hub": 0}
    examples = {"subway": [], "train": [], "bus": [], "transit_hub": []}

    for place in places:
        primary_type = str(place.get("primaryType") or "").strip().lower()
        mode = TRANSIT_TYPE_TO_MODE.get(primary_type, "transit_hub")
        counts[mode] += 1

        display_name = place.get("displayName") or {}
        text = display_name.get("text") if isinstance(display_name, dict) else None
        if text and len(examples[mode]) < 3:
            examples[mode].append(str(text))

    return {
        "counts": counts,
        "examples": examples,
    }


def _count_to_score(count: int, saturation: int) -> float:
    """Convert a raw nearby-place count into a normalized score."""

    if saturation <= 0:
        return 0.0
    return _clip(count / float(saturation))


def _commute_minutes_to_score(minutes: float | None) -> float | None:
    """Map commute duration into a simple normalized preference score."""

    if minutes is None:
        return None
    if minutes <= 20:
        return 1.0
    if minutes <= 35:
        return 0.82
    if minutes <= 50:
        return 0.62
    if minutes <= 70:
        return 0.38
    return 0.18


def _location_context_summary(listing: dict[str, Any]) -> str:
    """Create a compact candidate summary for stage-two LLM reranking."""

    context = dict(listing.get("location_context") or {})
    neighborhood = listing.get("neighborhood") or listing.get("neighborhood_group") or "Unknown area"
    price = listing.get("price") or 0.0
    commute_text = context.get("commute_summary") or "no commute data"
    preferred_transit_modes = ", ".join(context.get("preferred_transit_modes", [])[:3]) or "any"
    subway_count = context.get("nearby_subway_count", 0)
    train_count = context.get("nearby_train_count", 0)
    bus_count = context.get("nearby_bus_count", 0)
    transit_hub_count = context.get("nearby_transit_hub_count", 0)
    subway_examples = ", ".join(context.get("nearby_subway_examples", [])[:3]) or "none"
    train_examples = ", ".join(context.get("nearby_train_examples", [])[:3]) or "none"
    bus_examples = ", ".join(context.get("nearby_bus_examples", [])[:3]) or "none"
    food_examples = ", ".join(context.get("nearby_food_examples", [])[:3]) or "none"
    grocery_examples = ", ".join(context.get("nearby_grocery_examples", [])[:3]) or "none"
    return (
        f"id={listing.get('id')} | title={listing.get('title', 'Untitled')} | neighborhood={neighborhood} | "
        f"price=${float(price):,.0f} nightly | score={float(listing.get('score', 0.0)):.2f} | "
        f"review_rating={listing.get('review_rating')} | detailed_location_score={float(listing.get('detailed_location_score', 0.0)):.2f} | "
        f"preferred_transit_modes={preferred_transit_modes} | "
        f"subway_count={subway_count} ({subway_examples}) | "
        f"train_count={train_count} ({train_examples}) | "
        f"bus_count={bus_count} ({bus_examples}) | "
        f"transit_hub_count={transit_hub_count} | "
        f"food_count={context.get('nearby_food_count', 0)} ({food_examples}) | "
        f"grocery_count={context.get('nearby_grocery_count', 0)} ({grocery_examples}) | "
        f"commute={commute_text}"
    )


def _llm_is_available() -> bool:
    """Return whether stage-two LLM reranking can run."""

    return HAS_LLM and ChatOpenAI is not None and bool(os.environ.get("OPENAI_API_KEY"))


def _rerank_enriched_with_llm(
    listings: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Use the LLM to balance the enriched neighborhood facts holistically."""

    if not _llm_is_available() or not listings:
        return None

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(EnrichedRankingResponse)
        prompt = (
            "You are balancing apartment recommendations after live Google Maps enrichment.\n"
            "Use the retrieved transit, food, grocery, and commute facts as primary evidence.\n"
            "Do not invent neighborhood facts beyond what is provided.\n"
            "Treat the current numeric score as a strong prior, but rerank candidates based on the user's likely lived experience.\n"
            "Return every candidate id sorted best to worst with fit_score values between 0.0 and 1.0.\n\n"
            f"Hard constraints:\n{hard_constraints}\n\n"
            f"Soft preferences:\n{soft_preferences}\n\n"
            "Candidates:\n"
            + "\n".join(_location_context_summary(listing) for listing in listings)
        )
        response = structured_llm.invoke(prompt)
    except Exception as exc:
        print(f"Warning: enriched LLM reranking failed: {exc}")
        return None

    candidate_map = {str(listing.get("id")): dict(listing) for listing in listings}
    reranked: list[dict[str, Any]] = []

    for candidate in response.ranked_candidates:
        listing = candidate_map.pop(str(candidate.id), None)
        if listing is None:
            continue
        prior_score = float(listing.get("score", 0.0))
        llm_fit_score = _clip(float(candidate.fit_score))
        listing["pre_enrichment_llm_score"] = round(prior_score, 4)
        listing["stage_two_llm_fit_score"] = round(llm_fit_score, 4)
        listing["llm_rank_reason"] = candidate.reason.strip()
        listing["score"] = round((0.35 * prior_score) + (0.65 * llm_fit_score), 4)
        score_breakdown = dict(listing.get("score_breakdown", {}))
        score_breakdown["google_maps_fit"] = round(float(listing.get("detailed_location_score", 0.0)), 4)
        score_breakdown["stage_two_llm_fit"] = round(llm_fit_score, 4)
        listing["score_breakdown"] = score_breakdown
        reranked.append(listing)

    leftovers = sorted(
        candidate_map.values(),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    reranked.extend(leftovers)
    return reranked


def _resolve_commute_destinations(destinations: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve named commute anchors into coordinates for routing."""

    resolved: list[dict[str, Any]] = []
    failures: list[str] = []

    for destination in destinations[:3]:
        try:
            place = _search_text_place(destination)
        except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
            failures.append(f"{destination}: {exc}")
            continue

        if not place:
            failures.append(f"{destination}: no match")
            continue

        location = place.get("location") or {}
        latitude = _safe_float(location.get("latitude"))
        longitude = _safe_float(location.get("longitude"))
        if latitude is None or longitude is None:
            failures.append(f"{destination}: no coordinates")
            continue

        display_name = place.get("displayName") or {}
        name_text = display_name.get("text") if isinstance(display_name, dict) else destination
        resolved.append(
            {
                "query": destination,
                "name": str(name_text or destination),
                "latitude": latitude,
                "longitude": longitude,
                "formatted_address": place.get("formattedAddress"),
            }
        )

    return resolved, failures


def _enrich_listing(
    listing: dict[str, Any],
    resolved_destinations: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Attach Google Maps neighborhood context to a single listing."""

    enriched = dict(listing)
    latitude = _safe_float(listing.get("latitude"))
    longitude = _safe_float(listing.get("longitude"))
    warnings: list[str] = []

    if latitude is None or longitude is None:
        enriched["location_context"] = {
            "google_maps_enriched": False,
            "reason": "missing_listing_coordinates",
        }
        enriched["detailed_location_score"] = float(listing.get("score", 0.0))
        return enriched, warnings

    nearby_transit: list[dict[str, Any]] = []
    nearby_food: list[dict[str, Any]] = []
    nearby_grocery: list[dict[str, Any]] = []

    try:
        nearby_transit = _search_nearby(latitude, longitude, TRANSIT_TYPES, radius_meters=1200, max_result_count=6)
    except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        warnings.append(f"transit lookup failed for {listing.get('id')}: {exc}")

    try:
        nearby_food = _search_nearby(latitude, longitude, FOOD_TYPES, radius_meters=1600, max_result_count=8)
    except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        warnings.append(f"food lookup failed for {listing.get('id')}: {exc}")

    try:
        nearby_grocery = _search_nearby(latitude, longitude, GROCERY_TYPES, radius_meters=1400, max_result_count=5)
    except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        warnings.append(f"grocery lookup failed for {listing.get('id')}: {exc}")

    commute_summaries: list[str] = []
    commute_minutes: list[float] = []
    travel_mode = "TRANSIT" if soft_preferences.get("transit_priority") else "DRIVE"
    for destination in resolved_destinations:
        try:
            minutes = _compute_commute_minutes(
                origin_latitude=latitude,
                origin_longitude=longitude,
                destination_latitude=float(destination["latitude"]),
                destination_longitude=float(destination["longitude"]),
                travel_mode=travel_mode,
            )
        except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
            warnings.append(f"route lookup failed for {listing.get('id')} -> {destination['query']}: {exc}")
            continue

        if minutes is None:
            continue
        commute_minutes.append(minutes)
        commute_summaries.append(f"{destination['name']}: {minutes:.0f} min")

    transit_score = _count_to_score(len(nearby_transit), saturation=5)
    transit_breakdown = _classify_transit_places(nearby_transit)
    transit_counts = transit_breakdown["counts"]
    transit_examples_by_mode = transit_breakdown["examples"]
    preferred_transit_modes = [
        str(mode).strip().lower()
        for mode in soft_preferences.get("preferred_transit_modes", [])
        if str(mode).strip()
    ]

    if preferred_transit_modes:
        preferred_transit_hits = sum(transit_counts.get(mode, 0) for mode in preferred_transit_modes)
        transit_score = _count_to_score(preferred_transit_hits, saturation=max(2, len(preferred_transit_modes) * 2))
        if transit_counts.get("transit_hub", 0):
            transit_score = _clip(transit_score + min(0.1, 0.05 * transit_counts["transit_hub"]))

    food_score = _count_to_score(len(nearby_food), saturation=7)
    grocery_score = _count_to_score(len(nearby_grocery), saturation=4)
    commute_score_values = [
        score
        for score in (_commute_minutes_to_score(minutes) for minutes in commute_minutes)
        if score is not None
    ]
    avg_commute_score = (
        sum(commute_score_values) / len(commute_score_values)
        if commute_score_values
        else None
    )

    weighted_components: list[tuple[float, float]] = [(grocery_score, 0.15)]
    weighted_components.append((transit_score, 0.25 if soft_preferences.get("transit_priority") else 0.15))
    weighted_components.append((food_score, 0.25 if soft_preferences.get("food_scene_priority") else 0.15))
    if avg_commute_score is not None:
        weighted_components.append((avg_commute_score, 0.40))

    detailed_location_score = sum(score * weight for score, weight in weighted_components) / sum(
        weight for _, weight in weighted_components
    )

    enriched["location_context"] = {
        "google_maps_enriched": True,
        "travel_mode": travel_mode,
        "nearby_transit_count": len(nearby_transit),
        "nearby_transit_examples": _collect_place_names(nearby_transit),
        "preferred_transit_modes": preferred_transit_modes,
        "nearby_subway_count": transit_counts["subway"],
        "nearby_subway_examples": transit_examples_by_mode["subway"],
        "nearby_train_count": transit_counts["train"],
        "nearby_train_examples": transit_examples_by_mode["train"],
        "nearby_bus_count": transit_counts["bus"],
        "nearby_bus_examples": transit_examples_by_mode["bus"],
        "nearby_transit_hub_count": transit_counts["transit_hub"],
        "nearby_transit_hub_examples": transit_examples_by_mode["transit_hub"],
        "nearby_food_count": len(nearby_food),
        "nearby_food_examples": _collect_place_names(nearby_food),
        "nearby_grocery_count": len(nearby_grocery),
        "nearby_grocery_examples": _collect_place_names(nearby_grocery),
        "commute_destinations": [destination["name"] for destination in resolved_destinations],
        "commute_summaries": commute_summaries,
        "commute_summary": "; ".join(commute_summaries) if commute_summaries else "no live commute data",
        "average_commute_minutes": round(sum(commute_minutes) / len(commute_minutes), 1) if commute_minutes else None,
    }
    enriched["detailed_location_score"] = round(_clip(detailed_location_score), 4)
    enriched["score_breakdown"] = dict(enriched.get("score_breakdown", {}))
    enriched["score_breakdown"]["google_maps_fit"] = enriched["detailed_location_score"]
    enriched["score"] = round(
        (0.65 * float(enriched.get("score", 0.0))) + (0.35 * enriched["detailed_location_score"]),
        4,
    )
    return enriched, warnings


def enrich_and_rerank_listings(
    listings: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Google Maps enrichment followed by stage-two LLM balancing."""

    if not listings:
        return [], {"google_maps_used": False, "reason": "no_shortlisted_listings"}

    if not google_maps_available():
        return list(listings), {"google_maps_used": False, "reason": "missing_google_maps_api_key"}

    commute_destinations = [
        str(item).strip()
        for item in soft_preferences.get("commute_destinations", [])
        if str(item).strip()
    ]
    resolved_destinations, destination_warnings = _resolve_commute_destinations(commute_destinations)

    enriched_listings: list[dict[str, Any]] = []
    warnings: list[str] = list(destination_warnings)
    for listing in listings:
        enriched, listing_warnings = _enrich_listing(
            listing,
            resolved_destinations=resolved_destinations,
            soft_preferences=soft_preferences,
        )
        warnings.extend(listing_warnings)
        enriched_listings.append(enriched)

    enriched_listings.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    llm_reranked = _rerank_enriched_with_llm(
        listings=enriched_listings,
        soft_preferences=soft_preferences,
        hard_constraints=hard_constraints,
    )
    final_ranked = llm_reranked if llm_reranked is not None else enriched_listings
    final_ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    diagnostics = {
        "google_maps_used": True,
        "resolved_commute_destinations": [item["name"] for item in resolved_destinations],
        "warnings": warnings,
        "listing_count_enriched": len(enriched_listings),
        "stage_two_llm_used": llm_reranked is not None,
    }
    return final_ranked, diagnostics
