"""Neighborhood context helpers for commute, transit, and food scoring."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.config import PROJECT_ROOT


BOROUGH_BASE_SCORES: dict[str, dict[str, float]] = {
    "manhattan": {"transit": 0.90, "food": 0.84},
    "brooklyn": {"transit": 0.74, "food": 0.80},
    "queens": {"transit": 0.66, "food": 0.78},
    "bronx": {"transit": 0.58, "food": 0.62},
    "staten island": {"transit": 0.38, "food": 0.52},
}

NEIGHBORHOOD_OVERRIDES: dict[str, dict[str, float]] = {
    "chelsea": {"transit": 0.98, "food": 0.90},
    "east village": {"transit": 0.92, "food": 0.97},
    "financial district": {"transit": 0.98, "food": 0.82},
    "flatiron district": {"transit": 0.99, "food": 0.90},
    "flushing": {"transit": 0.75, "food": 0.99},
    "greenwich village": {"transit": 0.95, "food": 0.95},
    "hell's kitchen": {"transit": 0.96, "food": 0.89},
    "jackson heights": {"transit": 0.76, "food": 0.96},
    "long island city": {"transit": 0.86, "food": 0.76},
    "lower east side": {"transit": 0.91, "food": 0.96},
    "midtown": {"transit": 1.00, "food": 0.86},
    "morningside heights": {"transit": 0.88, "food": 0.74},
    "park slope": {"transit": 0.82, "food": 0.86},
    "prospect heights": {"transit": 0.80, "food": 0.89},
    "soho": {"transit": 0.95, "food": 0.92},
    "tribeca": {"transit": 0.91, "food": 0.87},
    "upper east side": {"transit": 0.91, "food": 0.80},
    "upper west side": {"transit": 0.92, "food": 0.82},
    "west village": {"transit": 0.92, "food": 0.98},
    "williamsburg": {"transit": 0.86, "food": 0.95},
    "downtown brooklyn": {"transit": 0.90, "food": 0.82},
    "astoria": {"transit": 0.76, "food": 0.93},
    "bushwick": {"transit": 0.70, "food": 0.90},
}

DESTINATION_ALIASES: dict[str, str] = {
    "baruch": "gramercy",
    "brooklyn heights": "downtown brooklyn",
    "columbia": "morningside heights",
    "cuny graduate center": "midtown",
    "downtown manhattan": "financial district",
    "fidi": "financial district",
    "fit": "chelsea",
    "financial district": "financial district",
    "fordham lincoln center": "upper west side",
    "fordham rose hill": "belmont",
    "midtown east": "midtown",
    "midtown west": "midtown",
    "nyu": "greenwich village",
    "pace": "financial district",
    "times square": "midtown",
}

TRANSIT_TITLE_KEYWORDS = ["subway", "train", "metro", "station", "commute", "transit"]
FOOD_TITLE_KEYWORDS = ["restaurant", "restaurants", "dining", "cafes", "coffee", "eatery", "food"]


def _normalize(text: str | None) -> str:
    """Lowercase and normalize spacing for place names."""

    return " ".join(str(text or "").lower().replace("-", " ").split())


def _iter_points(coordinates: Any) -> list[tuple[float, float]]:
    """Recursively collect longitude/latitude pairs from GeoJSON coordinates."""

    if not isinstance(coordinates, list) or not coordinates:
        return []
    if len(coordinates) == 2 and all(isinstance(value, (int, float)) for value in coordinates):
        return [(float(coordinates[0]), float(coordinates[1]))]

    points: list[tuple[float, float]] = []
    for item in coordinates:
        points.extend(_iter_points(item))
    return points


@lru_cache(maxsize=1)
def load_neighborhood_centers() -> dict[str, dict[str, Any]]:
    """Load neighborhood centroids from the GeoJSON shipped with the repo."""

    path = Path(PROJECT_ROOT) / "neighbourhoods.geojson"
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    centers: dict[str, dict[str, Any]] = {}

    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        neighborhood = str(properties.get("neighbourhood") or "").strip()
        if not neighborhood:
            continue

        points = _iter_points(feature.get("geometry", {}).get("coordinates", []))
        if not points:
            continue

        avg_lon = sum(point[0] for point in points) / len(points)
        avg_lat = sum(point[1] for point in points) / len(points)
        centers[_normalize(neighborhood)] = {
            "name": neighborhood,
            "borough": str(properties.get("neighbourhood_group") or "").strip() or None,
            "latitude": avg_lat,
            "longitude": avg_lon,
        }

    return centers


def resolve_place_reference(place: str) -> dict[str, Any] | None:
    """Resolve a commute or school reference to a neighborhood centroid when possible."""

    normalized = _normalize(place)
    if not normalized:
        return None

    centers = load_neighborhood_centers()
    alias_target = DESTINATION_ALIASES.get(normalized, normalized)
    if alias_target in centers:
        return centers[alias_target]

    for key, value in centers.items():
        if alias_target == key or alias_target in key or key in alias_target:
            return value
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the haversine distance in kilometers between two coordinates."""

    earth_radius_km = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def compute_commute_score(listing: dict[str, Any], commute_destinations: list[str]) -> float | None:
    """Score commute fit based on distance to work or school anchors."""

    if not commute_destinations:
        return None

    listing_lat = listing.get("latitude")
    listing_lon = listing.get("longitude")
    listing_neighborhood = _normalize(listing.get("neighborhood"))
    listing_borough = _normalize(listing.get("neighborhood_group"))

    scores: list[float] = []
    for destination in commute_destinations:
        resolved = resolve_place_reference(destination)
        if resolved is None:
            continue

        if (
            listing_lat is not None
            and listing_lon is not None
            and resolved.get("latitude") is not None
            and resolved.get("longitude") is not None
        ):
            distance_km = haversine_km(
                float(listing_lat),
                float(listing_lon),
                float(resolved["latitude"]),
                float(resolved["longitude"]),
            )
            if distance_km <= 1.5:
                score = 1.0
            elif distance_km <= 3.0:
                score = 0.86
            elif distance_km <= 5.0:
                score = 0.72
            elif distance_km <= 8.0:
                score = 0.55
            elif distance_km <= 12.0:
                score = 0.35
            else:
                score = 0.12
            scores.append(score)
            continue

        destination_name = _normalize(resolved.get("name"))
        destination_borough = _normalize(resolved.get("borough"))
        if destination_name and destination_name == listing_neighborhood:
            scores.append(1.0)
        elif destination_borough and destination_borough == listing_borough:
            scores.append(0.65)
        else:
            scores.append(0.30)

    if not scores:
        return None
    return sum(scores) / len(scores)


def _score_from_profile(listing: dict[str, Any], dimension: str) -> float:
    """Return a transit or food score for a listing's neighborhood."""

    borough = _normalize(listing.get("neighborhood_group"))
    neighborhood = _normalize(listing.get("neighborhood"))
    base = BOROUGH_BASE_SCORES.get(borough, {}).get(dimension, 0.55)
    score = NEIGHBORHOOD_OVERRIDES.get(neighborhood, {}).get(dimension, base)

    title = _normalize(listing.get("title"))
    keywords = TRANSIT_TITLE_KEYWORDS if dimension == "transit" else FOOD_TITLE_KEYWORDS
    if any(keyword in title for keyword in keywords):
        score = min(1.0, score + 0.08)
    return score


def compute_transit_score(listing: dict[str, Any]) -> float:
    """Estimate how friendly the location is for frequent transit use."""

    return _score_from_profile(listing, "transit")


def compute_food_score(listing: dict[str, Any]) -> float:
    """Estimate how strong the neighborhood is for dining and food access."""

    return _score_from_profile(listing, "food")
