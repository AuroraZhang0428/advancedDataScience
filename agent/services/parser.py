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
    BaseModel = None
    Field = None
    ChatOpenAI = None

if HAS_LLM:
    class PreferenceWeights(BaseModel):
        review_rating: float = Field(
            default=0.2,
            description="Relative importance of review quality from 0.0 to 1.0.",
        )
        amenity_match: float = Field(
            default=0.2,
            description="Relative importance of amenity fit from 0.0 to 1.0.",
        )
        purpose_alignment: float = Field(
            default=0.2,
            description="Relative importance of remote-work, quiet, or usage fit from 0.0 to 1.0.",
        )
        neighborhood_fit: float = Field(
            default=0.2,
            description="Relative importance of neighborhood, commute, transit, and local-area fit from 0.0 to 1.0.",
        )
        price_score: float = Field(
            default=0.2,
            description="Relative importance of price fit from 0.0 to 1.0.",
        )


    class ApartmentPreferences(BaseModel):
        min_bedrooms: int | None = Field(default=None, description="Minimum number of bedrooms")
        min_bathrooms: float | None = Field(default=None, description="Minimum number of bathrooms")
        price_floor: float | None = Field(
            default=None,
            description="Minimum acceptable nightly or monthly price when the user asks for at least a certain price point.",
        )
        max_price: float | None = Field(default=None, description="Maximum price or budget")
        target_price: float | None = Field(
            default=None,
            description="Desired nightly or monthly target price when the user asks for a place around a certain dollar amount rather than a hard ceiling.",
        )
        price_period: Literal["nightly", "monthly"] = Field(
            default="nightly",
            description="Whether the stated price is nightly or monthly. Default to 'nightly' whenever a price is mentioned without an explicit time unit.",
        )
        price_preference: Literal["cheap", "expensive", "moderate", "none"] = Field(
            default="none",
            description="Qualitative price preference if explicit budget is not given. Use 'cheap' only for words like affordable, inexpensive, bargain, budget-friendly, or low-cost. Do not set 'cheap' just because the user states a budget amount."
        )
        preferred_neighborhoods: list[str] = Field(default_factory=list, description="Desired neighborhoods or areas")
        desired_amenities: list[str] = Field(default_factory=list, description="List of desired amenities, e.g. wifi, workspace, gym, laundry, parking")
        commute_destinations: list[str] = Field(
            default_factory=list,
            description="Workplaces, campuses, or commute anchors the user needs frequent access to.",
        )
        remote_work: bool = Field(default=False, description="Whether the user wants remote work suitability")
        transit_priority: bool = Field(
            default=False,
            description="Whether nearby public transportation or easy commuting is important.",
        )
        preferred_transit_modes: list[Literal["subway", "train", "bus"]] = Field(
            default_factory=list,
            description="Specific transit modes the user prefers, such as subway, train, or bus.",
        )
        food_scene_priority: bool = Field(
            default=False,
            description="Whether strong restaurant, cafe, grocery, or food access matters.",
        )
        quiet_preference: bool = Field(default=False, description="Whether the user prefers a quiet place")
        review_min_rating: float | None = Field(default=None, description="Minimum review rating mentioned")
        room_type: str | None = Field(default=None, description="Room type: 'Entire home/apt' or 'Private room'")
        priority_weights: PreferenceWeights = Field(
            default_factory=PreferenceWeights,
            description="Weighting of review, amenities, purpose, neighborhood, and price based on the user's priorities.",
        )



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


def _extract_price_floor(query: str) -> float | None:
    """Extract a minimum acceptable price from user text."""

    patterns = [
        r"(?:at least|minimum|min(?:imum)? of|starting at|from)\s*\$?(?P<amount>\d[\d,]*)",
        r"(?:not cheaper than|no cheaper than|not below|above|over|more than)\s*\$?(?P<amount>\d[\d,]*)",
        r"\$?(?P<amount>\d[\d,]*)\s*(?:minimum|min(?:imum)?|floor)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group("amount").replace(",", ""))
    return None


def _extract_target_price(query: str, explicit_budget: float | None) -> float | None:
    """Extract a desired target price when the user names a price without making it a hard ceiling."""

    if explicit_budget is not None:
        return None

    patterns = [
        r"(?:around|about|roughly|approximately|close to)\s*\$?(?P<amount>\d[\d,]*)",
        r"\$?(?P<amount>\d[\d,]*)\s*(?:dollars?)?\s+(?:place|listing|apartment|room|stay)\b",
        r"\$(?P<amount>\d[\d,]*)\b",
        r"(?P<amount>\d[\d,]*)\$\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return float(match.group("amount").replace(",", ""))
    return None


def _extract_price_period(query: str, has_explicit_budget: bool) -> Literal["nightly", "monthly"]:
    """Infer the intended time unit for price mentions, defaulting to nightly."""

    lowered = query.lower()
    monthly_markers = [
        "per month",
        "a month",
        "/month",
        "monthly",
        "month rent",
        "monthly rent",
    ]
    nightly_markers = [
        "per night",
        "a night",
        "/night",
        "nightly",
        "per day",
        "daily",
    ]

    if any(marker in lowered for marker in monthly_markers):
        return "monthly"
    if any(marker in lowered for marker in nightly_markers):
        return "nightly"
    if has_explicit_budget:
        return "nightly"
    return "nightly"


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


def _extract_commute_destinations(query: str) -> list[str]:
    """Extract work or school anchors from natural language."""

    patterns = [
        r"(?:work|office|job|commute|travel)\s+(?:in|to|near|around)\s+(?P<place>[^.;,]+)",
        r"(?:student|study|studying|classes)\s+(?:at|in|near)\s+(?P<place>[^.;,]+)",
        r"(?:go to school|attend school)\s+(?:at|in|near)\s+(?P<place>[^.;,]+)",
    ]
    stop_markers = [" with ", " and ", " but ", " because ", " for ", " so "]
    destinations: list[str] = []

    for pattern in patterns:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            raw_place = match.group("place").strip(" .")
            for marker in stop_markers:
                marker_index = raw_place.lower().find(marker)
                if marker_index != -1:
                    raw_place = raw_place[:marker_index]
                    break
            cleaned = re.sub(r"^(the|my)\s+", "", raw_place.strip(), flags=re.IGNORECASE)
            if cleaned:
                destinations.append(cleaned.title())

    deduped: list[str] = []
    seen: set[str] = set()
    for destination in destinations:
        normalized = destination.lower()
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(destination)
    return deduped


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


def _extract_transit_priority(query: str, commute_destinations: list[str]) -> bool:
    """Detect whether transit or easy commuting matters to the user."""

    signals = [
        "subway",
        "train",
        "bus",
        "public transit",
        "transportation",
        "easy commute",
        "close to transit",
        "near transit",
        "near the train",
    ]
    lowered = query.lower()
    return bool(commute_destinations) or any(signal in lowered for signal in signals)


def _extract_preferred_transit_modes(query: str) -> list[str]:
    """Extract specific transit modes the user cares about."""

    lowered = query.lower()
    preferred_modes: list[str] = []

    if any(signal in lowered for signal in ["subway", "subway station", "metro", "underground"]):
        preferred_modes.append("subway")
    if any(
        signal in lowered
        for signal in ["train", "rail", "railroad", "commuter rail", "lirr", "metro-north", "amtrak"]
    ):
        preferred_modes.append("train")
    if any(signal in lowered for signal in ["bus", "buses", "bus stop", "bus line"]):
        preferred_modes.append("bus")

    return sorted(set(preferred_modes))


def _extract_food_scene_priority(query: str) -> bool:
    """Detect whether the surrounding food scene matters."""

    signals = [
        "food",
        "restaurants",
        "restaurant",
        "dining",
        "cafes",
        "cafe",
        "coffee shops",
        "groceries",
        "grocery",
        "takeout",
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
    price_floor: float | None,
    max_price: float | None,
    target_price: float | None,
    price_period: Literal["nightly", "monthly"],
    preferred_neighborhoods: list[str],
    desired_amenities: list[str],
    commute_destinations: list[str],
    remote_work: bool,
    transit_priority: bool,
    preferred_transit_modes: list[str],
    food_scene_priority: bool,
    quiet_preference: bool,
    review_min_rating: float | None,
    room_type: str | None,
    priority_weights: dict[str, float] | None = None,
    price_preference: Literal["cheap", "expensive", "moderate", "none"] = "none",
) -> dict[str, Any]:
    normalized_priority_weights = _normalize_priority_weights(priority_weights)
    raw_preferences = {
        "min_bedrooms": min_bedrooms,
        "min_bathrooms": min_bathrooms,
        "price_floor": price_floor,
        "max_price": max_price,
        "target_price": target_price,
        "price_period": price_period,
        "price_preference": price_preference,
        "preferred_neighborhoods": preferred_neighborhoods,
        "desired_amenities": desired_amenities,
        "commute_destinations": commute_destinations,
        "remote_work": remote_work,
        "transit_priority": transit_priority,
        "preferred_transit_modes": preferred_transit_modes,
        "food_scene_priority": food_scene_priority,
        "quiet_preference": quiet_preference,
        "review_min_rating": review_min_rating,
        "room_type": room_type,
        "priority_weights": normalized_priority_weights,
    }

    hard_constraints = {
        key: value
        for key, value in {
            "min_bedrooms": min_bedrooms,
            "min_bathrooms": min_bathrooms,
            "max_price": max_price,
            "price_period": price_period,
            "room_type": room_type,
            "price_preference": price_preference,
        }.items()
        if value is not None
    }

    soft_preferences = {
        "preferred_neighborhoods": preferred_neighborhoods,
        "desired_amenities": desired_amenities,
        "commute_destinations": commute_destinations,
        "remote_work": remote_work,
        "transit_priority": transit_priority,
        "preferred_transit_modes": preferred_transit_modes,
        "food_scene_priority": food_scene_priority,
        "quiet_preference": quiet_preference,
        "review_min_rating": review_min_rating,
        "price_floor": price_floor,
        "target_price": target_price,
        "priority_weights": normalized_priority_weights,
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
        "target_price": {
            "kind": "semi_hard",
            "can_relax": target_price is not None,
            "requires_user_confirmation": target_price is not None,
            "suggested_increase_pct": 0.5,
        },
        "price_floor": {
            "kind": "semi_hard",
            "can_relax": price_floor is not None,
            "requires_user_confirmation": price_floor is not None,
            "suggested_decrease_pct": 0.15,
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
    
    cheap_words = [
        "cheap",
        "affordable",
        "budget-friendly",
        "cost-effective",
        "inexpensive",
        "bargain",
        "low cost",
        "low-cost",
        "low price",
        "low-priced",
        "economic",
        "economical",
        "value",
    ]
    expensive_words = ["expensive", "luxury", "high-end", "premium", "pricey", "luxurious", "upscale", "high cost", "high-cost", "lavish"]
    moderate_words = ["moderate", "mid-range", "mid range", "average price", "reasonably priced", "fair price"]
    
    if any(re.search(rf"\b{word}\b", lowered) for word in cheap_words):
        return "cheap"
    if any(re.search(rf"\b{word}\b", lowered) for word in expensive_words):
        return "expensive"
    if any(re.search(rf"\b{word}\b", lowered) for word in moderate_words):
        return "moderate"
        
    return "none"


def _normalize_priority_weights(priority_weights: dict[str, Any] | None) -> dict[str, float]:
    """Normalize user-priority weights so they sum to 1.0."""

    default_weights = {
        "review_rating": 0.2,
        "amenity_match": 0.2,
        "purpose_alignment": 0.2,
        "neighborhood_fit": 0.2,
        "price_score": 0.2,
    }
    if not priority_weights:
        return default_weights

    cleaned: dict[str, float] = {}
    total = 0.0
    for key in default_weights:
        raw_value = priority_weights.get(key, default_weights[key])
        try:
            numeric = max(float(raw_value), 0.0)
        except (TypeError, ValueError):
            numeric = default_weights[key]
        cleaned[key] = numeric
        total += numeric

    if total <= 0:
        return default_weights

    return {key: round(value / total, 4) for key, value in cleaned.items()}


def _require_llm_parser() -> None:
    """Ensure the OpenAI-backed parser is available."""

    if not HAS_LLM or ChatOpenAI is None or BaseModel is None or Field is None:
        raise RuntimeError(
            "OpenAI-backed parsing requires langchain_openai and pydantic to be installed."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required because rule-based parsing has been removed.")


def parse_preferences_rule_based(user_query: str) -> dict[str, Any]:
    """Parse a query into hard and soft preferences without an LLM."""

    min_bedrooms = _extract_bedrooms(user_query)
    min_bathrooms = _extract_bathrooms(user_query)
    price_floor = _extract_price_floor(user_query)
    max_price = _extract_budget(user_query)
    target_price = _extract_target_price(user_query, explicit_budget=max_price)
    price_period = _extract_price_period(user_query, has_explicit_budget=max_price is not None)
    preferred_neighborhoods = _extract_preferred_neighborhoods(user_query)
    desired_amenities = _extract_amenities(user_query)
    commute_destinations = _extract_commute_destinations(user_query)
    remote_work = _extract_remote_work_preference(user_query)
    transit_priority = _extract_transit_priority(user_query, commute_destinations)
    preferred_transit_modes = _extract_preferred_transit_modes(user_query)
    food_scene_priority = _extract_food_scene_priority(user_query)
    review_min_rating = _extract_review_preference(user_query)
    room_type = _extract_room_type(user_query)
    quiet_preference = "quiet" in user_query.lower()
    price_preference = _extract_price_preference(user_query)

    return _build_preferences_dict(
        min_bedrooms=min_bedrooms,
        min_bathrooms=min_bathrooms,
        price_floor=price_floor,
        max_price=max_price,
        target_price=target_price,
        price_period=price_period,
        preferred_neighborhoods=preferred_neighborhoods,
        desired_amenities=desired_amenities,
        commute_destinations=commute_destinations,
        remote_work=remote_work,
        transit_priority=transit_priority,
        preferred_transit_modes=preferred_transit_modes,
        food_scene_priority=food_scene_priority,
        quiet_preference=quiet_preference,
        review_min_rating=review_min_rating,
        room_type=room_type,
        priority_weights=None,
        price_preference=price_preference,
    )


def extract_preferences_llm(user_query: str) -> dict[str, Any]:
    """Extract preferences using a structured LLM output."""

    _require_llm_parser()

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ApartmentPreferences)
        
        prompt = (
            "Extract user apartment leasing parameters from this query:\\n\\n{query}\\n\\n"
            "CRITICAL INSTRUCTION FOR PRICE TYPE:\\n"
            "Distinguish a hard floor, hard ceiling budget, and target price.\\n"
            " - Put values like 'at least $200', 'minimum $200', '$200 minimum', or 'not cheaper than $200' into price_floor.\\n"
            " - Put values like 'under $200', '$200 max', 'budget is $200', or 'up to $200' into max_price.\\n"
            " - Put values like '$200 place', 'around $200', or 'about $200 a night' into target_price instead.\\n"
            " - Do not fill all of price_floor, max_price, and target_price unless the user clearly expresses all of them.\\n\\n"
            "CRITICAL INSTRUCTION FOR PRICE PERIOD:\\n"
            "If the user mentions a price or budget without saying monthly/per month, default the price period to nightly.\\n"
            "Only use 'monthly' when the query explicitly says monthly/per month/month rent.\\n\\n"
            "CRITICAL INSTRUCTION FOR PRICE PREFERENCE:\\n"
            "You MUST set 'price_preference' accurately based on ANY word that implies price.\\n"
            " - Use 'cheap' for words like cheap, affordable, budget-friendly, bargain, low cost, inexpensive, economical, value, etc.\\n"
            " - Do NOT use 'cheap' only because the user states a budget amount such as '$200 max' or 'budget is $150'.\\n"
            " - Use 'expensive' if the user mentions ANY of: expensive, luxury, high-end, premium, pricey, upscale, lavish, etc.\\n"
            " - Use 'moderate' for mid-range or reasonably priced.\\n"
            " - Use 'none' only if no price-related quality is mentioned.\\n\\n"
            "CRITICAL INSTRUCTION FOR PRIORITY WEIGHTS:\\n"
            "Infer how important each scoring dimension is from the user's wording and emphasis.\\n"
            "You must set priority_weights across review_rating, amenity_match, purpose_alignment, neighborhood_fit, and price_score.\\n"
            "The weights do not need to sum perfectly because the code will normalize them, but they should reflect relative importance.\\n"
            "Examples:\\n"
            " - If the user emphasizes commute, subway access, neighborhood, or nearby lifestyle, increase neighborhood_fit.\\n"
            " - If the user emphasizes budget, affordability, or exact price targets, increase price_score.\\n"
            " - If the user emphasizes remote work, quietness, or how they will use the space, increase purpose_alignment.\\n"
            " - If the user emphasizes specific amenities, increase amenity_match.\\n"
            " - If the user emphasizes ratings or trust, increase review_rating.\\n\\n"
            "COMMUTE, TRANSIT, AND LIFESTYLE INSTRUCTIONS:\\n"
            " - If the user says they work in a place, study at a school, or commute to an area, add that place or school to commute_destinations.\\n"
            " - If the user mentions subway access, public transit, easy commuting, or has commute destinations, set transit_priority to true.\\n"
            " - Populate preferred_transit_modes when the user specifically prefers subway, train, or bus access.\\n"
            " - If the user cares about restaurants, cafes, grocery access, dining, or the local food scene, set food_scene_priority to true.\\n"
            " - Neighborhood intent is broader than exact neighborhood names; capture both literal area preferences and commute/lifestyle needs.\\n"
        ).format(query=user_query)
        
        result = structured_llm.invoke(prompt)
        priority_weights = (
            result.priority_weights.model_dump()
            if hasattr(result.priority_weights, "model_dump")
            else result.priority_weights.dict()
        )
        
        return _build_preferences_dict(
            min_bedrooms=result.min_bedrooms,
            min_bathrooms=result.min_bathrooms,
            price_floor=result.price_floor,
            max_price=result.max_price,
            target_price=result.target_price,
            price_period=result.price_period,
            price_preference=result.price_preference,
            preferred_neighborhoods=result.preferred_neighborhoods,
            desired_amenities=result.desired_amenities,
            commute_destinations=result.commute_destinations,
            remote_work=result.remote_work,
            transit_priority=result.transit_priority,
            preferred_transit_modes=result.preferred_transit_modes,
            food_scene_priority=result.food_scene_priority,
            quiet_preference=result.quiet_preference,
            review_min_rating=result.review_min_rating,
            room_type=result.room_type,
            priority_weights=priority_weights,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI-backed preference parsing failed: {e}") from e


def parse_preferences(user_query: str) -> dict[str, Any]:
    """Public parsing interface backed only by the LLM parser."""

    return extract_preferences_llm(user_query)
