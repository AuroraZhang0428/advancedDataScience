"""Shared data models used across the leasing agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Listing:
    """Normalized apartment listing used by the scoring pipeline."""

    id: str
    title: str
    host_name: str | None = None
    neighborhood: str | None = None
    neighborhood_group: str | None = None
    price: float | None = None
    accommodates: float | None = None
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
