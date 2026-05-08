"""Google Maps enrichment for shortlisted apartment candidates."""

from __future__ import annotations

import json
import os
import re
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

TRANSIT_RADIUS_METERS = 1000
FOOD_RADIUS_METERS = 1400
GROCERY_RADIUS_METERS = 1400

TRANSIT_TYPES = ["subway_station", "train_station", "transit_station", "bus_station"]
FOOD_TYPES = ["restaurant", "cafe", "bakery", "meal_takeaway"]
GROCERY_TYPES = ["supermarket", "grocery_store", "convenience_store"]
TRANSIT_TYPE_TO_MODE = {
    "subway_station": "subway",
    "train_station": "train",
    "bus_station": "bus",
    "transit_station": "transit_hub",
}

# Maps user-supplied cuisine labels to substrings found in Google place names/types
CUISINE_TYPE_MAP: dict[str, list[str]] = {
    "italian": ["italian", "pizza", "pasta", "trattoria", "osteria"],
    "pizza": ["pizza"],
    "japanese": ["japanese", "sushi", "ramen", "izakaya", "tempura"],
    "sushi": ["sushi"],
    "chinese": ["chinese", "dim sum", "cantonese", "szechuan", "peking"],
    "mexican": ["mexican", "taco", "burrito", "tex-mex"],
    "indian": ["indian", "curry", "tandoor", "biryani"],
    "thai": ["thai"],
    "french": ["french", "brasserie", "bistro", "crepe"],
    "american": ["american", "burger", "bbq", "barbecue", "diner", "steakhouse"],
    "fast food": ["mcdonald", "burger king", "subway", "kfc", "wendy", "chipotle", "five guys", "shake shack"],
    "cafe": ["cafe", "coffee", "espresso", "starbucks", "dunkin"],
    "bakery": ["bakery", "boulangerie", "patisserie", "pastry"],
    "vegan": ["vegan", "plant-based", "plant based"],
    "vegetarian": ["vegetarian", "vegan", "veggie"],
}

# Regex matching a single subway/train line token like "A", "1", "Q", "L", "NJ", "PATH"
_LINE_TOKEN_RE = re.compile(r'^[A-Z0-9]{1,4}$')


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


def _require_google_maps() -> None:
    """Ensure Google Maps enrichment is available."""

    if not google_maps_available():
        raise RuntimeError("GOOGLE_MAPS_API_KEY is required because Google Maps enrichment fallback has been removed.")


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


def _is_path_station(place: dict[str, Any]) -> bool:
    """Detect a PATH station by display name since Google Places has no dedicated type."""
    display_name = place.get("displayName") or {}
    name = (display_name.get("text") if isinstance(display_name, dict) else "") or ""
    name_upper = name.upper()
    return any(kw in name_upper for kw in ("PATH STATION", "PATH TRAIN", "PATH TERMINAL", "PATH - "))


def _classify_transit_places(places: list[dict[str, Any]]) -> dict[str, Any]:
    """Break nearby transit results into mode-specific counts and examples."""

    counts = {"subway": 0, "train": 0, "bus": 0, "transit_hub": 0, "path": 0}
    examples: dict[str, list[str]] = {"subway": [], "train": [], "bus": [], "transit_hub": [], "path": []}

    for place in places:
        display_name = place.get("displayName") or {}
        text = display_name.get("text") if isinstance(display_name, dict) else None
        name_str = str(text or "")

        if _is_path_station(place):
            mode = "path"
        else:
            primary_type = str(place.get("primaryType") or "").strip().lower()
            mode = TRANSIT_TYPE_TO_MODE.get(primary_type, "transit_hub")

        counts[mode] += 1
        if text and len(examples[mode]) < 3:
            examples[mode].append(name_str)

    return {
        "counts": counts,
        "examples": examples,
    }


def _cuisine_matches(place_name: str, cuisine_labels: list[str]) -> bool:
    """Return True if the place name contains a substring for any of the cuisine labels."""
    name_lower = place_name.lower()
    for label in cuisine_labels:
        for substring in CUISINE_TYPE_MAP.get(label, [label]):
            if substring in name_lower:
                return True
    return False


def _score_food_places(
    food_places: list[dict[str, Any]],
    preferred_cuisines: list[str],
    avoided_cuisines: list[str],
) -> float:
    """Score food places with cuisine-awareness."""
    if not food_places:
        return 0.0

    total = 0.0
    for place in food_places:
        display_name = place.get("displayName") or {}
        name = (display_name.get("text") if isinstance(display_name, dict) else "") or ""
        if avoided_cuisines and _cuisine_matches(name, avoided_cuisines):
            continue
        if preferred_cuisines and _cuisine_matches(name, preferred_cuisines):
            total += 0.20
        else:
            total += 0.10

    # Normalize: 7 generic places ≈ 0.70, saturate at 1.0
    return _clip(total / 0.70)


def _transit_diversity_score(transit_places: list[dict[str, Any]]) -> float:
    """Score transit by mode variety and distinct subway/train line count."""
    if not transit_places:
        return 0.0

    # Mode diversity
    modes_present: set[str] = set()
    for place in transit_places:
        if _is_path_station(place):
            modes_present.add("path")
        else:
            primary_type = str(place.get("primaryType") or "").strip().lower()
            modes_present.add(TRANSIT_TYPE_TO_MODE.get(primary_type, "transit_hub"))
    mode_score = _clip(len(modes_present) * 0.25)

    # Line diversity: extract tokens from parenthetical suffixes like "(4,5,6,L)"
    unique_lines: set[str] = set()
    for place in transit_places:
        display_name = place.get("displayName") or {}
        name = (display_name.get("text") if isinstance(display_name, dict) else "") or ""
        for paren in re.findall(r'\(([^)]+)\)', name):
            for token in re.split(r'[,/\s]+', paren):
                token = token.strip().upper()
                if token and _LINE_TOKEN_RE.match(token):
                    unique_lines.add(token)

    if len(unique_lines) >= 2:
        line_score = _clip(len(unique_lines) * 0.12)
        return _clip(mode_score * 0.3 + line_score * 0.7)
    return mode_score


def _score_transit_places(
    transit_places: list[dict[str, Any]],
    preferred_transit_modes: list[str],
    transit_breakdown: dict[str, Any],
) -> float:
    """Compute a transit score that considers diversity and user mode preferences."""
    transit_counts = transit_breakdown["counts"]

    if preferred_transit_modes:
        preferred_hits = sum(transit_counts.get(mode, 0) for mode in preferred_transit_modes)
        base = _clip(preferred_hits / max(2, len(preferred_transit_modes) * 2))
        # Small bonus for transit hubs and PATH when user wants PATH
        hub_bonus = min(0.1, 0.05 * transit_counts.get("transit_hub", 0))
        path_bonus = 0.1 if "path" in preferred_transit_modes and transit_counts.get("path", 0) > 0 else 0.0
        return _clip(base + hub_bonus + path_bonus)

    return _transit_diversity_score(transit_places)


def _resolve_travel_mode(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    transit_priority: bool,
) -> str:
    """Pick WALK for very short distances, TRANSIT or DRIVE otherwise."""
    import math
    dlat = (dest_lat - origin_lat) * 111_000
    dlon = (dest_lon - origin_lon) * 111_000 * math.cos(math.radians(origin_lat))
    distance_m = math.sqrt(dlat ** 2 + dlon ** 2)
    if distance_m < 1000:
        return "WALK"
    return "TRANSIT" if transit_priority else "DRIVE"


def _compute_dynamic_weights(soft_preferences: dict[str, Any], has_commute: bool) -> dict[str, float]:
    """Compute location score component weights mirroring resolve_scoring_weights logic."""
    transit_priority = bool(soft_preferences.get("transit_priority"))
    food_priority = bool(soft_preferences.get("food_scene_priority"))
    preferred_modes = [str(m).lower() for m in soft_preferences.get("preferred_transit_modes", [])]
    preferred_cuisines = soft_preferences.get("preferred_cuisines", [])

    w_transit = 0.20
    w_food = 0.15
    w_grocery = 0.15
    w_commute = 0.50 if has_commute else 0.0

    if transit_priority or preferred_modes:
        w_transit += 0.10
    if food_priority or preferred_cuisines:
        w_food += 0.10

    if not has_commute:
        # Redistribute commute weight proportionally
        total_non_commute = w_transit + w_food + w_grocery
        if total_non_commute > 0:
            scale = 1.0 / total_non_commute
            w_transit *= scale
            w_food *= scale
            w_grocery *= scale
        return {"transit": w_transit, "food": w_food, "grocery": w_grocery, "commute": 0.0}

    # Normalize so all weights sum to 1
    total = w_transit + w_food + w_grocery + w_commute
    return {
        "transit": w_transit / total,
        "food": w_food / total,
        "grocery": w_grocery / total,
        "commute": w_commute / total,
    }


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
    path_count = context.get("nearby_path_count", 0)
    subway_examples = ", ".join(context.get("nearby_subway_examples", [])[:3]) or "none"
    train_examples = ", ".join(context.get("nearby_train_examples", [])[:3]) or "none"
    bus_examples = ", ".join(context.get("nearby_bus_examples", [])[:3]) or "none"
    path_examples = ", ".join(context.get("nearby_path_examples", [])[:3]) or "none"
    food_examples = ", ".join(context.get("nearby_food_examples", [])[:3]) or "none"
    grocery_examples = ", ".join(context.get("nearby_grocery_examples", [])[:3]) or "none"
    return (
        f"id={listing.get('id')} | title={listing.get('title', 'Untitled')} | neighborhood={neighborhood} | "
        f"price=${float(price):,.0f} nightly | stage_one_fit={float(listing.get('score', 0.0)):.2f} | "
        f"review_rating={listing.get('review_rating')} | detailed_location_score={float(listing.get('detailed_location_score', 0.0)):.2f} | "
        f"preferred_transit_modes={preferred_transit_modes} | "
        f"subway_count={subway_count} ({subway_examples}) | "
        f"train_count={train_count} ({train_examples}) | "
        f"bus_count={bus_count} ({bus_examples}) | "
        f"path_count={path_count} ({path_examples}) | "
        f"transit_hub_count={transit_hub_count} | "
        f"food_count={context.get('nearby_food_count', 0)} ({food_examples}) | "
        f"grocery_count={context.get('nearby_grocery_count', 0)} ({grocery_examples}) | "
        f"commute={commute_text}"
    )


def _llm_is_available() -> bool:
    """Return whether stage-two LLM reranking can run."""

    return HAS_LLM and ChatOpenAI is not None and bool(os.environ.get("OPENAI_API_KEY"))


def _require_llm_reranking() -> None:
    """Ensure the OpenAI-backed stage-two reranking path is available."""

    if not _llm_is_available():
        raise RuntimeError("OPENAI_API_KEY is required because non-LLM stage-two reranking fallback has been removed.")


def _rerank_enriched_with_llm(
    listings: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """Use the LLM to balance the enriched neighborhood facts holistically."""

    if not listings:
        return []

    _require_llm_reranking()

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(EnrichedRankingResponse)
        prompt = (
            "You are balancing apartment recommendations after live Google Maps enrichment.\n"
            "Use the retrieved transit, food, grocery, and commute facts as primary evidence.\n"
            "Do not invent neighborhood facts beyond what is provided.\n"
            "Use the earlier stage_one_fit only as coarse context from retrieval, not as a prior you must follow.\n"
            "If the live neighborhood evidence points somewhere else, trust the live evidence.\n"
            "Return every candidate id sorted best to worst with fit_score values between 0.0 and 1.0.\n\n"
            f"Hard constraints:\n{hard_constraints}\n\n"
            f"Soft preferences:\n{soft_preferences}\n\n"
            "Candidates:\n"
            + "\n".join(_location_context_summary(listing) for listing in listings)
        )
        response = structured_llm.invoke(prompt)
    except Exception as exc:
        raise RuntimeError(f"OpenAI-backed stage-two reranking failed: {exc}") from exc

    candidate_map = {str(listing.get("id")): dict(listing) for listing in listings}
    reranked: list[dict[str, Any]] = []

    for candidate in response.ranked_candidates:
        listing = candidate_map.pop(str(candidate.id), None)
        if listing is None:
            continue
        prior_score = float(listing.get("score", 0.0))
        llm_fit_score = _clip(float(candidate.fit_score))
        listing["pre_enrichment_score"] = round(prior_score, 4)
        listing["stage_two_llm_fit_score"] = round(llm_fit_score, 4)
        listing["llm_rank_reason"] = candidate.reason.strip()
        listing["score"] = round(llm_fit_score, 4)
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
        nearby_transit = _search_nearby(latitude, longitude, TRANSIT_TYPES, radius_meters=TRANSIT_RADIUS_METERS, max_result_count=8)
    except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        warnings.append(f"transit lookup failed for {listing.get('id')}: {exc}")

    try:
        nearby_food = _search_nearby(latitude, longitude, FOOD_TYPES, radius_meters=FOOD_RADIUS_METERS, max_result_count=10)
    except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        warnings.append(f"food lookup failed for {listing.get('id')}: {exc}")

    try:
        nearby_grocery = _search_nearby(latitude, longitude, GROCERY_TYPES, radius_meters=GROCERY_RADIUS_METERS, max_result_count=5)
    except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
        warnings.append(f"grocery lookup failed for {listing.get('id')}: {exc}")

    transit_priority = bool(soft_preferences.get("transit_priority"))
    preferred_transit_modes = [
        str(mode).strip().lower()
        for mode in soft_preferences.get("preferred_transit_modes", [])
        if str(mode).strip()
    ]
    preferred_cuisines = soft_preferences.get("preferred_cuisines", [])
    avoided_cuisines = soft_preferences.get("avoided_cuisines", [])

    transit_breakdown = _classify_transit_places(nearby_transit)
    transit_counts = transit_breakdown["counts"]
    transit_examples_by_mode = transit_breakdown["examples"]

    transit_score = _score_transit_places(nearby_transit, preferred_transit_modes, transit_breakdown)
    food_score = _score_food_places(nearby_food, preferred_cuisines, avoided_cuisines)
    grocery_score = _clip(len(nearby_grocery) / 4.0)

    commute_summaries: list[str] = []
    commute_minutes_list: list[float] = []
    for destination in resolved_destinations:
        dest_lat = float(destination["latitude"])
        dest_lon = float(destination["longitude"])
        travel_mode = _resolve_travel_mode(latitude, longitude, dest_lat, dest_lon, transit_priority)
        try:
            minutes = _compute_commute_minutes(
                origin_latitude=latitude,
                origin_longitude=longitude,
                destination_latitude=dest_lat,
                destination_longitude=dest_lon,
                travel_mode=travel_mode,
            )
        except (RuntimeError, error.URLError, error.HTTPError, TimeoutError, OSError) as exc:
            warnings.append(f"route lookup failed for {listing.get('id')} -> {destination['query']}: {exc}")
            continue

        if minutes is None:
            continue
        commute_minutes_list.append(minutes)
        commute_summaries.append(f"{destination['name']}: {minutes:.0f} min ({travel_mode.lower()})")

    commute_score_values = [
        s for s in (_commute_minutes_to_score(m) for m in commute_minutes_list) if s is not None
    ]
    avg_commute_score = (
        sum(commute_score_values) / len(commute_score_values) if commute_score_values else None
    )

    has_commute = avg_commute_score is not None
    weights = _compute_dynamic_weights(soft_preferences, has_commute)

    detailed_location_score = (
        transit_score * weights["transit"]
        + food_score * weights["food"]
        + grocery_score * weights["grocery"]
        + (avg_commute_score * weights["commute"] if has_commute else 0.0)
    )

    enriched["location_context"] = {
        "google_maps_enriched": True,
        "nearby_transit_count": len(nearby_transit),
        "nearby_transit_examples": _collect_place_names(nearby_transit),
        "preferred_transit_modes": preferred_transit_modes,
        "nearby_subway_count": transit_counts["subway"],
        "nearby_subway_examples": transit_examples_by_mode["subway"],
        "nearby_train_count": transit_counts["train"],
        "nearby_train_examples": transit_examples_by_mode["train"],
        "nearby_bus_count": transit_counts["bus"],
        "nearby_bus_examples": transit_examples_by_mode["bus"],
        "nearby_path_count": transit_counts["path"],
        "nearby_path_examples": transit_examples_by_mode["path"],
        "nearby_transit_hub_count": transit_counts["transit_hub"],
        "nearby_transit_hub_examples": transit_examples_by_mode["transit_hub"],
        "nearby_food_count": len(nearby_food),
        "nearby_food_examples": _collect_place_names(nearby_food),
        "nearby_grocery_count": len(nearby_grocery),
        "nearby_grocery_examples": _collect_place_names(nearby_grocery),
        "commute_destinations": [destination["name"] for destination in resolved_destinations],
        "commute_summaries": commute_summaries,
        "commute_summary": "; ".join(commute_summaries) if commute_summaries else "no live commute data",
        "average_commute_minutes": round(sum(commute_minutes_list) / len(commute_minutes_list), 1) if commute_minutes_list else None,
    }
    enriched["detailed_location_score"] = round(_clip(detailed_location_score), 4)
    enriched["score_breakdown"] = dict(enriched.get("score_breakdown", {}))
    enriched["score_breakdown"]["google_maps_fit"] = enriched["detailed_location_score"]
    return enriched, warnings


def enrich_and_rerank_listings(
    listings: list[dict[str, Any]],
    soft_preferences: dict[str, Any],
    hard_constraints: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run Google Maps enrichment followed by stage-two LLM balancing.

    When neither GOOGLE_MAPS_API_KEY nor OPENAI_API_KEY are available the
    function skips all external calls and returns the listings sorted by their
    existing deterministic score so the pipeline always produces results.
    """

    if not listings:
        return [], {"google_maps_used": False, "reason": "no_shortlisted_listings"}

    # ── Skip enrichment when Google Maps key is absent ──────────────────────
    if not google_maps_available():
        sorted_listings = sorted(
            listings,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        return sorted_listings, {
            "google_maps_used": False,
            "reason": "GOOGLE_MAPS_API_KEY not set — using deterministic score order",
        }

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

    # ── Stage-two LLM reranking (optional) ──────────────────────────────────
    if _llm_is_available():
        try:
            llm_reranked = _rerank_enriched_with_llm(
                listings=enriched_listings,
                soft_preferences=soft_preferences,
                hard_constraints=hard_constraints,
            )
            enriched_listings = llm_reranked
        except Exception:
            pass  # fall back to deterministic sort already applied above

    enriched_listings.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    diagnostics = {
        "google_maps_used": True,
        "resolved_commute_destinations": [item["name"] for item in resolved_destinations],
        "warnings": warnings,
        "listing_count_enriched": len(enriched_listings),
        "stage_two_llm_used": _llm_is_available(),
    }
    return enriched_listings, diagnostics
