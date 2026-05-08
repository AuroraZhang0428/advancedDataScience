"""Baseline 1 — Filter-Based Search.

Parse the user query with simple regex / keyword matching (no LLM).
Apply hard filters, sort by price ascending, return top-N results.
No semantic understanding, no scoring pipeline.
"""

from __future__ import annotations

import re
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

_TOP_N = 10

# Amenity keywords mapped to normalised amenity names
_AMENITY_KEYWORDS: dict[str, list[str]] = {
    "wifi":      ["wifi", "wi-fi", "wireless", "internet"],
    "workspace": ["workspace", "work space", "desk", "office", "remote work", "work from home", "wfh"],
    "gym":       ["gym", "fitness", "workout"],
    "laundry":   ["laundry", "washer", "dryer", "washing machine"],
    "parking":   ["parking", "garage"],
    "kitchen":   ["kitchen", "cook"],
    "tv":        ["tv", "television"],
    "ac":        ["ac", "air conditioning", "air conditioner"],
    "pool":      ["pool", "swimming"],
    "elevator":  ["elevator", "lift"],
}

# Neighbourhood aliases for popular NYC areas
_NEIGHBOURHOOD_ALIASES: dict[str, list[str]] = {
    "manhattan":    ["manhattan", "midtown", "downtown", "uptown", "upper east", "upper west",
                     "harlem", "chelsea", "soho", "tribeca", "fidi", "financial district",
                     "hell's kitchen", "hells kitchen", "east village", "west village",
                     "lower east side", "les", "gramercy", "kips bay", "murray hill",
                     "inwood", "washington heights"],
    "brooklyn":     ["brooklyn", "williamsburg", "bushwick", "bedford", "park slope",
                     "crown heights", "flatbush", "bay ridge", "cobble hill", "carroll gardens",
                     "boerum hill", "dumbo", "greenpoint", "sunset park", "prospect"],
    "queens":       ["queens", "astoria", "long island city", "lic", "flushing",
                     "jackson heights", "jamaica", "forest hills", "ridgewood"],
    "bronx":        ["bronx", "the bronx"],
    "staten island":["staten island", "si"],
}

# Room type keywords
_ROOM_TYPE_KEYWORDS: dict[str, list[str]] = {
    "private room": ["private room", "private bedroom"],
    "entire home":  ["entire", "whole apartment", "whole place", "whole home", "full apartment"],
    "shared room":  ["shared room", "shared space", "hostel", "dorm"],
}


# ── Regex helpers ─────────────────────────────────────────────────────────────

def _extract_price(query: str) -> float | None:
    """Return max nightly price from the query, or None if not mentioned."""
    patterns = [
        r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:/\s*night|per\s*night|a\s*night)?",
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|bucks?)\s*(?:a|per)?\s*night",
        r"under\s+\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"below\s+\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"less\s+than\s+\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"max(?:imum)?\s+(?:of\s+)?\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"budget(?:\s+of)?\s+\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:no more|not more) than\s+\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def _extract_bedrooms(query: str) -> int | None:
    """Return minimum bedrooms requested, or None."""
    patterns = [
        r"(\d+)\s*[-–]?\s*bed(?:room)?s?",
        r"(\d+)\s*br\b",
        r"(\d+)\s*bedroom",
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            return int(m.group(1))
    # Keywords
    if re.search(r"\bstudio\b", query, re.IGNORECASE):
        return 0
    return None


def _extract_min_rating(query: str) -> float | None:
    """Return minimum review rating, or None."""
    patterns = [
        r"(\d(?:\.\d+)?)\s*\+?\s*(?:stars?|rating|rated|score)",
        r"rated?\s+(\d(?:\.\d+)?)\s*\+",
        r"above\s+(\d(?:\.\d+)?)\s*(?:stars?)?",
        r"at\s+least\s+(\d(?:\.\d+)?)\s*(?:stars?)?",
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            # Normalise: if on a 5-star scale keep as-is, else assume 100-scale
            if val <= 5:
                return val
            return val / 20.0  # e.g. "90 rating" → 4.5 stars
    # "good reviews" / "highly rated" → implicit ≥ 4.0
    if re.search(r"\b(good|great|excellent|high(?:ly)?)\s*(?:reviews?|rated?|rating)\b", query, re.IGNORECASE):
        return 4.0
    return None


def _extract_amenities(query: str) -> list[str]:
    """Return list of amenity keys that appear in the query."""
    found = []
    q = query.lower()
    for amenity, keywords in _AMENITY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            found.append(amenity)
    return found


def _extract_neighborhoods(query: str) -> list[str]:
    """Return list of canonical neighbourhood names (borough or specific) from query."""
    found = []
    q = query.lower()
    for canonical, aliases in _NEIGHBOURHOOD_ALIASES.items():
        if any(alias in q for alias in aliases):
            found.append(canonical)
    # Also catch direct neighbourhood name fragments not in the alias table
    # (just re-search for capitalised words near location indicators)
    loc_m = re.findall(
        r"(?:in|near|around|at)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\s+area|,|\band\b|$)",
        query, re.IGNORECASE
    )
    for raw in loc_m:
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in found and len(cleaned) > 2:
            found.append(cleaned)
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def _extract_room_type(query: str) -> str | None:
    """Return 'private room', 'entire home', or 'shared room' if mentioned."""
    q = query.lower()
    for canonical, keywords in _ROOM_TYPE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return canonical
    return None


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_query(query: str) -> dict[str, Any]:
    """Parse a natural-language query into a flat dict of hard constraints."""
    return {
        "max_price":      _extract_price(query),
        "min_bedrooms":   _extract_bedrooms(query),
        "min_rating":     _extract_min_rating(query),
        "amenities":      _extract_amenities(query),
        "neighborhoods":  _extract_neighborhoods(query),
        "room_type":      _extract_room_type(query),
    }


# ── Filtering ─────────────────────────────────────────────────────────────────

def _listing_matches(listing: dict[str, Any], constraints: dict[str, Any]) -> bool:
    """Return True if the listing satisfies all extracted hard constraints."""

    # Price
    max_price = constraints["max_price"]
    if max_price is not None:
        price = listing.get("price")
        if price is None or float(price) > max_price:
            return False

    # Bedrooms
    min_beds = constraints["min_bedrooms"]
    if min_beds is not None:
        beds = listing.get("bedrooms")
        if beds is None or int(beds) < min_beds:
            return False

    # Rating
    min_rating = constraints["min_rating"]
    if min_rating is not None:
        rating = listing.get("review_rating") or listing.get("review_scores_rating")
        if rating is None or float(rating) < min_rating:
            return False

    # Amenities (all must be present)
    for amenity in constraints["amenities"]:
        # Check the amenities list field
        amenities_list = [a.lower() for a in (listing.get("amenities") or [])]
        # Also check boolean shortcut fields (wifi, workspace)
        bool_val = listing.get(amenity)
        if amenity not in amenities_list and not bool_val:
            return False

    # Neighbourhoods (at least one must match)
    neighbourhoods = constraints["neighborhoods"]
    if neighbourhoods:
        nbhd = (listing.get("neighborhood") or listing.get("neighbourhood") or
                listing.get("neighborhood_group") or "").lower()
        nbhd_group = (listing.get("neighborhood_group") or "").lower()
        matched = False
        for n in neighbourhoods:
            if n in nbhd or n in nbhd_group:
                matched = True
                break
        if not matched:
            return False

    # Room type
    room_type = constraints["room_type"]
    if room_type is not None:
        rt = (listing.get("room_type") or "").lower()
        if room_type not in rt:
            return False

    return True


# ── Serialisation ─────────────────────────────────────────────────────────────

def _serialise_listing(listing: dict[str, Any]) -> dict[str, Any]:
    price = listing.get("price")
    return {
        "id":               listing.get("id", ""),
        "title":            listing.get("title", "Untitled"),
        "neighborhood":     listing.get("neighborhood") or listing.get("neighbourhood") or
                            listing.get("neighborhood_group") or "Unknown area",
        "neighborhood_group": listing.get("neighborhood_group", ""),
        "price":            float(price) if price is not None else None,
        "bedrooms":         listing.get("bedrooms"),
        "bathrooms":        listing.get("bathrooms"),
        "review_rating":    listing.get("review_rating"),
        "amenities":        listing.get("amenities", []),
        "wifi":             listing.get("wifi"),
        "workspace":        listing.get("workspace"),
        "latitude":         listing.get("latitude"),
        "longitude":        listing.get("longitude"),
        "room_type":        listing.get("room_type"),
        "score":            0.0,  # No scoring in this baseline
        "score_breakdown":  {},
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_filter_baseline(listings: list[dict[str, Any]], query: str, top_n: int = _TOP_N) -> dict[str, Any]:
    """Run the filter-based baseline and return a response dict.

    Args:
        listings: Pre-loaded list of listing dicts (from agent.services.dataset).
        query:    Raw user query string.
        top_n:    Maximum number of results to return.

    Returns:
        Dict suitable for JSON serialisation with keys:
          - recommendations: list of serialised listings (price-sorted)
          - parsed_constraints: what the regex parser extracted
          - method: "baseline-filter"
          - total_matched: how many listings survived filtering
          - explanation: brief human-readable summary
    """
    constraints = parse_query(query)

    matched = [l for l in listings if _listing_matches(l, constraints)]

    # Sort by price ascending (cheapest first); put None prices last
    matched.sort(key=lambda l: (l.get("price") is None, float(l.get("price") or 0)))

    top = matched[:top_n]

    serialised = [_serialise_listing(l) for l in top]

    # Build a human-readable summary of what was parsed
    parts: list[str] = []
    if constraints["max_price"] is not None:
        parts.append(f"max price ${constraints['max_price']:.0f}/night")
    if constraints["min_bedrooms"] is not None:
        parts.append(f"≥{constraints['min_bedrooms']} bedroom(s)")
    if constraints["min_rating"] is not None:
        parts.append(f"≥{constraints['min_rating']:.1f}★ rating")
    if constraints["amenities"]:
        parts.append(f"amenities: {', '.join(constraints['amenities'])}")
    if constraints["neighborhoods"]:
        parts.append(f"area: {', '.join(constraints['neighborhoods'])}")
    if constraints["room_type"]:
        parts.append(f"room type: {constraints['room_type']}")

    summary = f"Applied {len(parts)} filter(s): {'; '.join(parts)}." if parts else "No specific filters detected — returning cheapest listings."
    summary += f" {len(matched)} listing(s) matched; showing top {len(serialised)}."

    return {
        "recommendations":   serialised,
        "parsed_constraints": {k: v for k, v in constraints.items() if v not in (None, [], "")},
        "method":            "baseline-filter",
        "total_matched":     len(matched),
        "explanation":       summary,
    }
