"""Preference parsing utilities for apartment search queries."""

from __future__ import annotations

import os
import re
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

if HAS_LLM:
    class ApartmentPreferences(BaseModel):
        min_bedrooms: int | None = Field(default=None, description="Minimum number of bedrooms")
        min_bathrooms: float | None = Field(default=None, description="Minimum number of bathrooms")
        max_price: float | None = Field(default=None, description="Maximum price or budget")
        price_preference: Literal["cheap", "expensive", "moderate", "none"] = Field(
            default="none",
            description="Qualitative price preference if explicit budget is not given. Use 'cheap' if user wants affordable/cheap options. Use 'expensive' for luxury. 'moderate' for mid-range. 'none' if unstated."
        )
        preferred_neighborhoods: list[str] = Field(default_factory=list, description="Desired neighborhoods or areas")
        desired_amenities: list[str] = Field(default_factory=list, description="List of desired amenities, e.g. wifi, workspace, gym, laundry, parking")
        remote_work: bool = Field(default=False, description="Whether the user wants remote work suitability")
        quiet_preference: bool = Field(default=False, description="Whether the user prefers a quiet place")
        review_min_rating: float | None = Field(default=None, description="Minimum review rating mentioned")
        room_type: str | None = Field(default=None, description="Room type: 'Entire home/apt' or 'Private room'")



NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

AMENITY_KEYWORDS: dict[str, list[str]] = {
    "wifi": ["wifi", "wi-fi", "internet", "broadband"],
    "workspace": ["workspace", "desk", "office"],
    "gym": ["gym", "fitness"],
    "laundry": ["laundry", "washer", "dryer"],
    "parking": ["parking", "garage"],
    "elevator": ["elevator", "lift"],
    "doorman": ["doorman", "concierge"],
    "pet_friendly": ["pet-friendly", "pet friendly", "pets", "dog", "cat"],
}


def _normalize_number(token: str) -> int | None:
    """Convert either numeric text or small number words into an integer."""

    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def _extract_bedrooms(query: str) -> int | None:
    """Extract a minimum bedroom requirement from user text."""

    patterns = [
        r"\b(?P<num>\d+|one|two|three|four|five)[\s-]*bed(room)?\b",
        r"\b(?P<num>\d+|one|two|three|four|five)\s+bedrooms?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return _normalize_number(match.group("num"))
    return None


def _extract_bathrooms(query: str) -> float | None:
    """Extract a minimum bathroom requirement from user text."""

    patterns = [
        r"at least\s+(?P<num>\d+(?:\.\d+)?)\s+bath(room)?s?\b",
        r"(?P<num>\d+(?:\.\d+)?)\s+bath(room)?s?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group("num"))
    return None


def _extract_budget(query: str) -> float | None:
    """Extract a maximum budget from user text."""

    patterns = [
        r"(?:under|below|max(?:imum)?|up to)\s*\$?(?P<amount>\d[\d,]*)",
        r"budget(?: is| around| of)?\s*\$?(?P<amount>\d[\d,]*)",
        r"\$(?P<amount>\d[\d,]*)\s*(?:max|maximum|budget)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group("amount").replace(",", ""))
    return None


def _extract_preferred_neighborhoods(query: str) -> list[str]:
    """Extract soft neighborhood preferences from common phrasing."""

    pattern = r"(?:in|near|around|prefer(?:ably)? in|ideally in)\s+(?P<areas>[^.;]+)"
    match = re.search(pattern, query, flags=re.IGNORECASE)
    if not match:
        return []

    raw_areas = match.group("areas")
    stop_markers = [" with ", " that ", " and has ", " and is ", " because ", " for "]
    for marker in stop_markers:
        marker_index = raw_areas.lower().find(marker)
        if marker_index != -1:
            raw_areas = raw_areas[:marker_index]
            break

    candidates = re.split(r",|/|\bor\b|\band\b", raw_areas, flags=re.IGNORECASE)
    neighborhoods: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip(" .")
        cleaned = re.sub(r"^(preferably|ideally|maybe|possibly)\s+", "", cleaned, flags=re.IGNORECASE)
        if cleaned:
            neighborhoods.append(cleaned.title())
    return neighborhoods


def _extract_amenities(query: str) -> list[str]:
    """Extract requested amenities from the query."""

    lowered = query.lower()
    amenities: list[str] = []
    for amenity, keywords in AMENITY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            amenities.append(amenity)
    return sorted(set(amenities))


def _extract_remote_work_preference(query: str) -> bool:
    """Detect whether the user cares about remote-work suitability."""

    signals = [
        "work remotely",
        "remote work",
        "wfh",
        "work from home",
        "good internet",
        "strong wifi",
        "quiet place to work",
    ]
    lowered = query.lower()
    return any(signal in lowered for signal in signals)


def _extract_review_preference(query: str) -> float | None:
    """Infer a desired review quality threshold."""

    lowered = query.lower()
    if any(phrase in lowered for phrase in ["good reviews", "great reviews", "highly rated", "well reviewed"]):
        return 4.2
    return None


def _extract_room_type(query: str) -> str | None:
    """Extract coarse room-type preferences when mentioned."""

    lowered = query.lower()
    if "entire place" in lowered or "entire apartment" in lowered or "entire home" in lowered:
        return "Entire home/apt"
    if "private room" in lowered:
        return "Private room"
    return None


def _build_preferences_dict(
    min_bedrooms: int | None,
    min_bathrooms: float | None,
    max_price: float | None,
    preferred_neighborhoods: list[str],
    desired_amenities: list[str],
    remote_work: bool,
    quiet_preference: bool,
    review_min_rating: float | None,
    room_type: str | None,
    price_preference: Literal["cheap", "expensive", "moderate", "none"] = "none",
) -> dict[str, Any]:
    raw_preferences = {
        "min_bedrooms": min_bedrooms,
        "min_bathrooms": min_bathrooms,
        "max_price": max_price,
        "price_preference": price_preference,
        "preferred_neighborhoods": preferred_neighborhoods,
        "desired_amenities": desired_amenities,
        "remote_work": remote_work,
        "quiet_preference": quiet_preference,
        "review_min_rating": review_min_rating,
        "room_type": room_type,
    }

    hard_constraints = {
        key: value
        for key, value in {
            "min_bedrooms": min_bedrooms,
            "min_bathrooms": min_bathrooms,
            "max_price": max_price,
            "room_type": room_type,
            "price_preference": price_preference,
        }.items()
        if value is not None
    }

    soft_preferences = {
        "preferred_neighborhoods": preferred_neighborhoods,
        "desired_amenities": desired_amenities,
        "remote_work": remote_work,
        "quiet_preference": quiet_preference,
        "review_min_rating": review_min_rating,
        "amenity_strictness": 1.0,
        "expanded_neighborhood_search": False,
    }

    relaxable_constraints = {
        "preferred_neighborhoods": {
            "kind": "soft",
            "can_relax": bool(preferred_neighborhoods),
            "requires_user_confirmation": False,
        },
        "review_min_rating": {
            "kind": "soft",
            "can_relax": review_min_rating is not None,
            "requires_user_confirmation": False,
            "minimum": 3.8,
            "step": 0.2,
        },
        "desired_amenities": {
            "kind": "soft",
            "can_relax": bool(desired_amenities),
            "requires_user_confirmation": False,
        },
        "min_bedrooms": {
            "kind": "semi_hard",
            "can_relax": min_bedrooms is not None and min_bedrooms >= 1,
            "requires_user_confirmation": min_bedrooms is not None,
            "relax_to": max((min_bedrooms or 0) - 1, 0),
        },
        "max_price": {
            "kind": "semi_hard",
            "can_relax": max_price is not None,
            "requires_user_confirmation": max_price is not None,
            "suggested_increase_pct": 0.1,
        },
    }

    return {
        "raw_preferences": raw_preferences,
        "hard_constraints": hard_constraints,
        "soft_preferences": soft_preferences,
        "relaxable_constraints": relaxable_constraints,
    }


def _extract_price_preference(query: str) -> Literal["cheap", "expensive", "moderate", "none"]:
    """Extract qualitative price preference from a broad set of synonyms."""

    lowered = query.lower()
    
    cheap_words = ["cheap", "affordable", "budget", "cost-effective", "inexpensive", "bargain", "low cost", "low-cost", "low price", "low-priced", "economic", "economical", "value"]
    expensive_words = ["expensive", "luxury", "high-end", "premium", "pricey", "luxurious", "upscale", "high cost", "high-cost", "lavish"]
    moderate_words = ["moderate", "mid-range", "mid range", "average price", "reasonably priced", "fair price"]
    
    if any(re.search(rf"\b{word}\b", lowered) for word in cheap_words):
        return "cheap"
    if any(re.search(rf"\b{word}\b", lowered) for word in expensive_words):
        return "expensive"
    if any(re.search(rf"\b{word}\b", lowered) for word in moderate_words):
        return "moderate"
        
    return "none"


def parse_preferences_rule_based(user_query: str) -> dict[str, Any]:
    """Parse a query into hard and soft preferences without an LLM."""

    min_bedrooms = _extract_bedrooms(user_query)
    min_bathrooms = _extract_bathrooms(user_query)
    max_price = _extract_budget(user_query)
    preferred_neighborhoods = _extract_preferred_neighborhoods(user_query)
    desired_amenities = _extract_amenities(user_query)
    remote_work = _extract_remote_work_preference(user_query)
    review_min_rating = _extract_review_preference(user_query)
    room_type = _extract_room_type(user_query)
    quiet_preference = "quiet" in user_query.lower()
    price_preference = _extract_price_preference(user_query)

    return _build_preferences_dict(
        min_bedrooms=min_bedrooms,
        min_bathrooms=min_bathrooms,
        max_price=max_price,
        preferred_neighborhoods=preferred_neighborhoods,
        desired_amenities=desired_amenities,
        remote_work=remote_work,
        quiet_preference=quiet_preference,
        review_min_rating=review_min_rating,
        room_type=room_type,
        price_preference=price_preference,
    )


def extract_preferences_llm(user_query: str) -> dict[str, Any] | None:
    """Extract preferences using a structured LLM output."""
    if not HAS_LLM or not os.environ.get("OPENAI_API_KEY"):
        return None

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ApartmentPreferences)
        
        prompt = (
            "Extract user apartment leasing parameters from this query:\\n\\n{query}\\n\\n"
            "CRITICAL INSTRUCTION FOR PRICE PREFERENCE:\\n"
            "You MUST set 'price_preference' accurately based on ANY word that implies price.\\n"
            " - Use 'cheap' if the user mentions ANY of: cheap, affordable, budget, bargain, low cost, inexpensive, economical, value, etc.\\n"
            " - Use 'expensive' if the user mentions ANY of: expensive, luxury, high-end, premium, pricey, upscale, lavish, etc.\\n"
            " - Use 'moderate' for mid-range or reasonably priced.\\n"
            " - Use 'none' only if no price-related quality is mentioned.\\n"
        ).format(query=user_query)
        
        result = structured_llm.invoke(prompt)
        
        return _build_preferences_dict(
            min_bedrooms=result.min_bedrooms,
            min_bathrooms=result.min_bathrooms,
            max_price=result.max_price,
            price_preference=result.price_preference,
            preferred_neighborhoods=result.preferred_neighborhoods,
            desired_amenities=result.desired_amenities,
            remote_work=result.remote_work,
            quiet_preference=result.quiet_preference,
            review_min_rating=result.review_min_rating,
            room_type=result.room_type,
        )
    except Exception as e:
        print(f"Warning: LLM pref parser failed: {e}")
        return None


def parse_preferences(user_query: str) -> dict[str, Any]:
    """Public parsing interface with an optional LLM integration hook."""

    llm_result = extract_preferences_llm(user_query)
    if llm_result is not None:
        return llm_result

    return parse_preferences_rule_based(user_query)
