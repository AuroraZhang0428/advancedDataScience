"""Preference parsing — LLM-only path using structured output."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class PreferenceWeights(BaseModel):
    review_rating: float = Field(default=0.2, description="Relative importance of review quality from 0.0 to 1.0.")
    amenity_match: float = Field(default=0.2, description="Relative importance of amenity fit from 0.0 to 1.0.")
    purpose_alignment: float = Field(default=0.2, description="Relative importance of remote-work, quiet, or usage fit from 0.0 to 1.0.")
    neighborhood_fit: float = Field(default=0.2, description="Relative importance of neighborhood, commute, transit, and local-area fit from 0.0 to 1.0.")
    price_score: float = Field(default=0.2, description="Relative importance of price fit from 0.0 to 1.0.")


class ApartmentPreferences(BaseModel):
    min_guests: int | None = Field(default=None, description="Minimum number of guests the listing should accommodate")
    min_bedrooms: int | None = Field(default=None, description="Minimum number of bedrooms")
    min_bathrooms: float | None = Field(default=None, description="Minimum number of bathrooms")
    price_floor: float | None = Field(
        default=None,
        description="Minimum acceptable price when the user asks for at least a certain price point.",
    )
    max_price: float | None = Field(default=None, description="Maximum price or budget")
    target_price: float | None = Field(
        default=None,
        description="Desired target price when the user asks for a place around a certain dollar amount rather than a hard ceiling.",
    )
    price_period: Literal["nightly", "monthly"] = Field(
        default="nightly",
        description="Whether the stated price is nightly or monthly. Default to 'nightly' whenever a price is mentioned without an explicit time unit.",
    )
    price_preference: Literal["cheap", "expensive", "moderate", "none"] = Field(
        default="none",
        description="Qualitative price preference if explicit budget is not given. Use 'cheap' only for words like affordable, inexpensive, bargain, budget-friendly, or low-cost. Do not set 'cheap' just because the user states a budget amount.",
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


_PARSE_PROMPT = """\
Extract user apartment leasing parameters from this query:\n\n{query}\n\n
CRITICAL INSTRUCTION FOR GUEST CAPACITY:
If the user mentions how many guests, people, or persons the place should fit, put that into min_guests.
Examples include 'for 4 guests', 'fits 6 people', or 'accommodates 2'.

CRITICAL INSTRUCTION FOR PRICE TYPE:
Distinguish a hard floor, hard ceiling budget, and target price.
 - Put values like 'at least $200', 'minimum $200', '$200 minimum', or 'not cheaper than $200' into price_floor.
 - Put values like 'under $200', '$200 max', 'budget is $200', or 'up to $200' into max_price.
 - Put values like '$200 place', 'around $200', or 'about $200 a night' into target_price instead.
 - Do not fill all of price_floor, max_price, and target_price unless the user clearly expresses all of them.

CRITICAL INSTRUCTION FOR PRICE PERIOD:
If the user mentions a price or budget without saying monthly/per month, default the price period to nightly.
Only use 'monthly' when the query explicitly says monthly/per month/month rent.

CRITICAL INSTRUCTION FOR PRICE PREFERENCE:
You MUST set 'price_preference' accurately based on ANY word that implies price.
 - Use 'cheap' for words like cheap, affordable, budget-friendly, bargain, low cost, inexpensive, economical, value, etc.
 - Do NOT use 'cheap' only because the user states a budget amount such as '$200 max' or 'budget is $150'.
 - Use 'expensive' if the user mentions ANY of: expensive, luxury, high-end, premium, pricey, upscale, lavish, etc.
 - Use 'moderate' for mid-range or reasonably priced.
 - Use 'none' only if no price-related quality is mentioned.

CRITICAL INSTRUCTION FOR PRIORITY WEIGHTS:
Infer how important each scoring dimension is from the user's wording and emphasis.
You must set priority_weights across review_rating, amenity_match, purpose_alignment, neighborhood_fit, and price_score.
The weights do not need to sum perfectly because the code will normalize them, but they should reflect relative importance.
Examples:
 - If the user emphasizes commute, subway access, neighborhood, or nearby lifestyle, increase neighborhood_fit.
 - If the user emphasizes budget, affordability, or exact price targets, increase price_score.
 - If the user emphasizes remote work, quietness, or how they will use the space, increase purpose_alignment.
 - If the user emphasizes specific amenities, increase amenity_match.
 - If the user emphasizes ratings or trust, increase review_rating.

COMMUTE, TRANSIT, AND LIFESTYLE INSTRUCTIONS:
 - If the user says they work in a place, study at a school, or commute to an area, add that place or school to commute_destinations.
 - If the user mentions subway access, public transit, easy commuting, or has commute destinations, set transit_priority to true.
 - Populate preferred_transit_modes when the user specifically prefers subway, train, or bus access.
 - If the user cares about restaurants, cafes, grocery access, dining, or the local food scene, set food_scene_priority to true.
 - Neighborhood intent is broader than exact neighborhood names; capture both literal area preferences and commute/lifestyle needs.
"""


def _normalize_priority_weights(priority_weights: dict[str, Any] | None) -> dict[str, float]:
    """Normalize priority weights so they sum to 1.0."""
    default = {"review_rating": 0.2, "amenity_match": 0.2, "purpose_alignment": 0.2, "neighborhood_fit": 0.2, "price_score": 0.2}
    if not priority_weights:
        return default

    cleaned: dict[str, float] = {}
    total = 0.0
    for key in default:
        try:
            val = max(float(priority_weights.get(key, default[key])), 0.0)
        except (TypeError, ValueError):
            val = default[key]
        cleaned[key] = val
        total += val

    if total <= 0:
        return default
    return {k: round(v / total, 4) for k, v in cleaned.items()}


def _build_preferences_dict(
    min_guests: int | None,
    min_bedrooms: int | None,
    min_bathrooms: float | None,
    price_floor: float | None,
    max_price: float | None,
    target_price: float | None,
    price_period: Literal["nightly", "monthly"],
    price_preference: Literal["cheap", "expensive", "moderate", "none"],
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
) -> dict[str, Any]:
    normalized_weights = _normalize_priority_weights(priority_weights)

    raw_preferences = {
        "min_guests": min_guests,
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
        "priority_weights": normalized_weights,
    }

    hard_constraints = {
        k: v for k, v in {
            "min_guests": min_guests,
            "min_bedrooms": min_bedrooms,
            "min_bathrooms": min_bathrooms,
            "max_price": max_price,
            "price_period": price_period,
            "room_type": room_type,
            "price_preference": price_preference,
        }.items() if v is not None
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
        "priority_weights": normalized_weights,
        "amenity_strictness": 1.0,
        "expanded_neighborhood_search": False,
    }

    relaxable_constraints = {
        "preferred_neighborhoods": {"kind": "soft", "can_relax": bool(preferred_neighborhoods), "requires_user_confirmation": False},
        "review_min_rating": {"kind": "soft", "can_relax": review_min_rating is not None, "requires_user_confirmation": False, "minimum": 3.8, "step": 0.2},
        "desired_amenities": {"kind": "soft", "can_relax": bool(desired_amenities), "requires_user_confirmation": False},
        "target_price": {"kind": "semi_hard", "can_relax": target_price is not None, "requires_user_confirmation": target_price is not None, "suggested_increase_pct": 0.5},
        "price_floor": {"kind": "semi_hard", "can_relax": price_floor is not None, "requires_user_confirmation": price_floor is not None, "suggested_decrease_pct": 0.15},
        "min_bedrooms": {"kind": "semi_hard", "can_relax": min_bedrooms is not None and min_bedrooms >= 1, "requires_user_confirmation": min_bedrooms is not None, "relax_to": max((min_bedrooms or 0) - 1, 0)},
        "min_guests": {"kind": "semi_hard", "can_relax": min_guests is not None and min_guests >= 1, "requires_user_confirmation": min_guests is not None, "relax_to": max((min_guests or 0) - 1, 1)},
        "max_price": {"kind": "semi_hard", "can_relax": max_price is not None, "requires_user_confirmation": max_price is not None, "suggested_increase_pct": 0.1},
    }

    return {
        "raw_preferences": raw_preferences,
        "hard_constraints": hard_constraints,
        "soft_preferences": soft_preferences,
        "relaxable_constraints": relaxable_constraints,
    }


def parse_preferences(user_query: str) -> dict[str, Any]:
    """Parse a natural-language query into structured preferences using the LLM."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for preference parsing.")

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ApartmentPreferences)
        result: ApartmentPreferences = structured_llm.invoke(_PARSE_PROMPT.format(query=user_query))

        priority_weights = (
            result.priority_weights.model_dump()
            if hasattr(result.priority_weights, "model_dump")
            else result.priority_weights.dict()
        )

        return _build_preferences_dict(
            min_guests=result.min_guests,
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
    except Exception as exc:
        raise RuntimeError(f"Preference parsing failed: {exc}") from exc
