"""Shared data models used across the leasing agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Listing:
    """Normalized apartment listing used by the scoring pipeline."""

    id: str
    title: str
    neighborhood: str | None = None
    neighborhood_group: str | None = None
    price: float | None = None
    bedrooms: float | None = None
    bathrooms: float | None = None
    area_sqft: float | None = None
    amenities: list[str] = field(default_factory=list)
    review_rating: float | None = None
    wifi: bool | None = None
    workspace: bool | None = None
    quiet_score: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    distance_to_target_area: float | None = None
    purpose_tags: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass into a plain dictionary for graph state."""

        return asdict(self)


@dataclass(slots=True)
class ScoreBreakdown:
    """Normalized component scores for a listing."""

    review_rating: float
    amenity_match: float
    purpose_alignment: float
    neighborhood_fit: float

    def as_dict(self) -> dict[str, float]:
        """Return the score breakdown as a plain dictionary."""

        return {
            "review_rating": self.review_rating,
            "amenity_match": self.amenity_match,
            "purpose_alignment": self.purpose_alignment,
            "neighborhood_fit": self.neighborhood_fit,
        }


@dataclass(slots=True)
class RelaxationDecision:
    """Structured output from the relaxation policy."""

    action: str
    reason: str
    change: dict[str, Any] = field(default_factory=dict)
    user_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the decision into a serializable dictionary."""

        return {
            "action": self.action,
            "reason": self.reason,
            "change": self.change,
            "user_question": self.user_question,
        }
