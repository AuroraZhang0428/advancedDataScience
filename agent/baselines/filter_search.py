"""Baseline 1 — Filter-Based Search.

Simulates a traditional rule-based apartment search engine:
  1. Parse query with regex/keyword heuristics (no LLM).
  2. Apply hard filters (price, bedrooms, neighborhood keyword).
  3. Sort by price ascending (cheapest first) and return top-N.

No semantic understanding, no soft scoring, no adaptive relaxation.
This represents what a basic property-search website would do.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Regex-based preference extractor
# ---------------------------------------------------------------------------

_PRICE_PATTERN = re.compile(
    r"\$\s*(\d[\d,]*)"          # $200 or $1,500
    r"(?:\s*[-–]\s*\$\s*(\d[\d,]*))?"  # optional range end
    r"(?:\s*/\s*(?:night|nightly|mo(?:nth)?|month))?",
    re.IGNORECASE,
)
_BEDROOM_PATTERN = re.compile(
    r"(\d+)\s*(?:-\s*)?(?:bed(?:room)?s?|BR|br)\b", re.IGNORECASE
)
_BATHROOM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*bath(?:room)?s?\b", re.IGNORECASE
)
_GUEST_PATTERN = re.compile(
    r"(\d+)\s*(?:guest|person|people|pax)\b", re.IGNORECASE
)

# Common NYC neighborhoods for keyword match
_NEIGHBORHOODS = [
    "chelsea", "harlem", "williamsburg", "brooklyn", "queens",
    "bronx", "manhattan", "lower east side", "upper east side",
    "upper west side", "midtown", "downtown", "soho", "tribeca",
    "astoria", "greenwich village", "west village", "hell's kitchen",
    "hell kitchen", "park slope", "bushwick", "bed-stuy",
    "bedford-stuyvesant", "long island city", "lic", "flushing",
    "financial district", "fidi", "murray hill", "gramercy",
    "east village", "noho", "nolita", "little italy", "chinatown",
    "battery park", "inwood", "washington heights", "morningside",
    "hamilton heights", "crown heights", "prospect heights",
    "cobble hill", "boerum hill", "fort greene", "dumbo",
    "red hook", "bay ridge", "sunset park", "flatbush",
    "east new york", "jamaica", "jackson heights", "ridgewood",
    "maspeth", "glendale", "forest hills", "rego park",
    "sunnyside", "woodside", "elmhurst", "corona",
]

# Amenity keywords → normalized canonical name
_AMENITY_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwifi\b|\bwi-fi\b|\bwireless\b|\binternet\b", re.IGNORECASE), "wifi"),
    (re.compile(r"\bworkspace\b|\bdesk\b|\bwork from home\b|\bwfh\b|\bremote work\b", re.IGNORECASE), "workspace"),
    (re.compile(r"\bgym\b|\bfitness\b|\bexercise\b", re.IGNORECASE), "gym"),
    (re.compile(r"\bparking\b|\bgarage\b", re.IGNORECASE), "parking"),
    (re.compile(r"\bpool\b|\bswimming\b", re.IGNORECASE), "pool"),
    (re.compile(r"\bpet\b|\bdog\b|\bcat\b", re.IGNORECASE), "pets_allowed"),
    (re.compile(r"\blaundry\b|\bwasher\b|\bdryer\b", re.IGNORECASE), "laundry"),
    (re.compile(r"\bkitchen\b|\bcooking\b", re.IGNORECASE), "kitchen"),
    (re.compile(r"\bdoorman\b", re.IGNORECASE), "doorman"),
    (re.compile(r"\bair\s*condition\b|\bac\b|\ba/c\b", re.IGNORECASE), "air_conditioning"),
]

_BOROUGH_ALIASES: dict[str, list[str]] = {
    "brooklyn": ["brooklyn"],
    "manhattan": ["manhattan"],
    "queens": ["queens"],
    "bronx": ["bronx"],
    "staten island": ["staten island"],
}

_ROOM_TYPE_KEYWORDS = {
    "entire": "Entire home/apt",
    "whole": "Entire home/apt",
    "full apartment": "Entire home/apt",
    "private room": "Private room",
    "shared room": "Shared room",
}

_BOROUGH_NAMES = {"brooklyn", "manhattan", "queens", "bronx", "staten island"}

_MONTHLY_KEYWORDS = re.compile(
    r"\bper\s+month\b|\bmonthly\b|\b/mo\b|\b/month\b", re.IGNORECASE
)


def _parse_price(query: str) -> dict[str, Any]:

    """Extract price ceiling and whether it's monthly."""
    monthly = bool(_MONTHLY_KEYWORDS.search(query))
    matches = _PRICE_PATTERN.findall(query)
    if not matches:
        return {}

    amounts: list[float] = []
    for m in matches:
        for part in m:
            if part:
                try:
                    amounts.append(float(part.replace(",", "")))
                except ValueError:
                    pass

    if not amounts:
        return {}

    max_price = max(amounts)
    if monthly:
        max_price = max_price / 30.0  # convert to nightly for dataset comparison

    return {"max_price": max_price, "price_period": "monthly" if monthly else "nightly"}


def parse_query(query: str) -> dict[str, Any]:
    """
    Heuristic rule-based parser. Returns structured constraints.
    Intentionally simple: no LLM, no context, no disambiguation.
    """
    constraints: dict[str, Any] = {
        "max_price": None,
        "min_bedrooms": None,
        "min_bathrooms": None,
        "min_guests": None,
        "neighborhoods": [],
        "amenities": [],
        "room_type": None,
        "price_period": "nightly",
    }

    # Price
    price_info = _parse_price(query)
    constraints.update(price_info)

    # Bedrooms
    br = _BEDROOM_PATTERN.search(query)
    if br:
        constraints["min_bedrooms"] = int(br.group(1))

    # Bathrooms
    ba = _BATHROOM_PATTERN.search(query)
    if ba:
        constraints["min_bathrooms"] = float(ba.group(1))

    # Guests
    gu = _GUEST_PATTERN.search(query)
    if gu:
        constraints["min_guests"] = int(gu.group(1))

    # Neighborhoods — simple substring match
    ql = query.lower()
    found_hoods = [n for n in _NEIGHBORHOODS if n in ql]
    constraints["neighborhoods"] = found_hoods

    # Amenities
    found_amenities: list[str] = []
    for pattern, canonical in _AMENITY_KEYWORDS:
        if pattern.search(query):
            found_amenities.append(canonical)
    constraints["amenities"] = found_amenities

    # Room type
    for kw, rt in _ROOM_TYPE_KEYWORDS.items():
        if kw in ql:
            constraints["room_type"] = rt
            break

    return constraints


# ---------------------------------------------------------------------------
# Filter + sort pipeline
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _listing_matches_neighborhood(
    listing: dict[str, Any],
    neighborhoods: list[str],
) -> bool:
    """Return True if the listing matches any of the requested neighborhood keywords."""
    hood = str(listing.get("neighborhood") or "").lower()
    hood_group = str(listing.get("neighborhood_group") or "").lower()
    for n in neighborhoods:
        if n in hood or hood in n:
            return True
        # Borough-level match: 'brooklyn' matches neighborhood_group='Brooklyn'
        if n in hood_group or hood_group in n:
            return True
        # Fuzzy: check if target neighborhood name contains the search term
        if n in _BOROUGH_NAMES and n in hood_group:
            return True
    return False


def _listing_has_amenity(listing: dict[str, Any], amenity: str) -> bool:
    """Check amenity presence via both the amenities list and boolean fields."""
    listing_amenities = {str(a).lower() for a in listing.get("amenities", [])}
    if amenity in listing_amenities:
        return True
    # Boolean field fallbacks
    if amenity == "wifi" and listing.get("wifi"):
        return True
    if amenity == "workspace" and listing.get("workspace"):
        return True
    return False


def filter_and_sort(
    listings: list[dict[str, Any]],
    constraints: dict[str, Any],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Apply hard filters then sort by price ascending.
    Mimics a simple property portal: exact match filters, cheapest first.
    """
    max_price = _safe_float(constraints.get("max_price"))
    min_bedrooms = _safe_float(constraints.get("min_bedrooms"))
    min_bathrooms = _safe_float(constraints.get("min_bathrooms"))
    min_guests = _safe_float(constraints.get("min_guests"))
    neighborhoods = [n.lower() for n in (constraints.get("neighborhoods") or [])]
    amenities = [a.lower() for a in (constraints.get("amenities") or [])]
    room_type = constraints.get("room_type")

    filtered: list[dict[str, Any]] = []

    for listing in listings:
        price = _safe_float(listing.get("price"))
        bedrooms = _safe_float(listing.get("bedrooms"))
        bathrooms = _safe_float(listing.get("bathrooms"))
        accommodates = _safe_float(listing.get("accommodates"))
        hood = str(listing.get("neighborhood") or listing.get("neighborhood_group") or "").lower()
        listing_rt = str(listing.get("raw", {}).get("room_type") or "").lower()

        # Hard price filter
        if max_price is not None and (price is None or price > max_price):
            continue
        # Hard bedroom filter
        if min_bedrooms is not None and (bedrooms is None or bedrooms < min_bedrooms):
            continue
        # Hard bathroom filter
        if min_bathrooms is not None and (bathrooms is None or bathrooms < min_bathrooms):
            continue
        # Hard guest filter
        if min_guests is not None and (accommodates is None or accommodates < min_guests):
            continue
        # Room type filter (exact keyword match)
        if room_type and listing_rt and room_type.lower() not in listing_rt:
            continue
        # Neighborhood filter — keyword substring match + borough-level match
        if neighborhoods and not _listing_matches_neighborhood(listing, neighborhoods):
            continue
        # Amenity filter — must have ALL specified amenities (with boolean-field fallback)
        if amenities and not all(_listing_has_amenity(listing, a) for a in amenities):
            continue

        filtered.append(listing)

    # Sort: price ascending (cheapest first) — this is the typical portal behavior
    filtered.sort(key=lambda l: (_safe_float(l.get("price")) or 9999.0))

    return filtered[:top_n]


def run_filter_baseline(
    listings: list[dict[str, Any]],
    query: str,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    End-to-end filter baseline: parse → filter → sort → return results.

    Returns a dict with keys matching the NestAI API response shape so
    the frontend comparison view can render it identically.
    """
    constraints = parse_query(query)
    results = filter_and_sort(listings, constraints, top_n=top_n)

    # Build a simple trace to explain what happened
    trace: list[dict[str, str]] = [
        {
            "step": "Parsed query with regex rules",
            "detail": (
                f"Extracted constraints via keyword/regex matching — "
                f"no LLM or semantic understanding used."
            ),
        },
        {
            "step": "Applied hard filters",
            "detail": (
                f"Filtered {len(listings):,} listings with exact rules: "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in constraints.items()
                    if v not in (None, [], "")
                )
                or "no constraints extracted"
            ),
        },
        {
            "step": "Sorted by price (ascending)",
            "detail": (
                f"No semantic scoring. Returned top {len(results)} listings "
                f"sorted cheapest-first from {len(listings):,} candidates."
            ),
        },
    ]

    # Serialize results in the same shape as the NestAI agent
    serialized = []
    for r in results:
        price = r.get("price")
        serialized.append({
            "id": r.get("id", ""),
            "title": r.get("title", "Untitled"),
            "neighborhood": r.get("neighborhood") or r.get("neighborhood_group") or "Unknown",
            "neighborhood_group": r.get("neighborhood_group", ""),
            "price": float(price) if price is not None else None,
            "bedrooms": r.get("bedrooms"),
            "bathrooms": r.get("bathrooms"),
            "review_rating": r.get("review_rating"),
            "amenities": r.get("amenities", []),
            "wifi": r.get("wifi"),
            "workspace": r.get("workspace"),
            "quiet_score": r.get("quiet_score"),
            "purpose_tags": r.get("purpose_tags", []),
            "score": 0.0,                      # no composite score computed
            "score_breakdown": {},
            "llm_fit_score": None,
            "llm_rank_reason": None,
            "deterministic_score": None,
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
        })

    return {
        "baseline": "filter",
        "baseline_label": "Filter-Based Search",
        "baseline_description": (
            "Traditional rule-based search: regex/keyword parsing, "
            "exact hard filters, sorted by price ascending. "
            "No AI, no soft scoring, no adaptive relaxation."
        ),
        "parsed_constraints": constraints,
        "recommendations": serialized,
        "explanations": [],
        "agent_trace": trace,
        "relaxation_history": [],
        "need_user_input": False,
        "user_question": None,
    }
