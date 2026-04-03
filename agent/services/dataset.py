"""Dataset loading and normalization helpers."""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from agent.models import Listing


BEDROOM_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bstudio\b", re.IGNORECASE), 0.5),
    (re.compile(r"\b1[\s-]*bed(room)?\b", re.IGNORECASE), 1.0),
    (re.compile(r"\b2[\s-]*bed(room)?\b", re.IGNORECASE), 2.0),
    (re.compile(r"\b3[\s-]*bed(room)?\b", re.IGNORECASE), 3.0),
    (re.compile(r"\b4[\s-]*bed(room)?\b", re.IGNORECASE), 4.0),
]

AMENITY_KEYWORDS: dict[str, list[str]] = {
    "wifi": ["wifi", "wi-fi", "internet", "broadband"],
    "workspace": ["workspace", "desk", "office", "workstation"],
    "gym": ["gym", "fitness"],
    "laundry": ["laundry", "washer", "dryer"],
    "parking": ["parking", "garage"],
    "elevator": ["elevator", "lift"],
    "doorman": ["doorman", "concierge"],
    "pet_friendly": ["pet", "dog", "cat", "pet-friendly"],
    "kitchen": ["kitchen"],
}

QUIET_POSITIVE = ["quiet", "peaceful", "tranquil", "calm", "serene"]
QUIET_NEGATIVE = ["nightlife", "busy", "lively", "vibrant", "party"]


def _coerce_numeric(value: Any) -> float | None:
    """Convert mixed string or numeric inputs into floats."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("$", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_amenities(value: Any, title: str = "") -> list[str]:
    """Normalize an amenity cell into a simple lowercase list."""

    amenities: list[str] = []
    if isinstance(value, list):
        amenities = [str(item).strip().lower() for item in value if str(item).strip()]
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    amenities = [str(item).strip().lower() for item in parsed if str(item).strip()]
            except (SyntaxError, ValueError):
                amenities = []
        if not amenities:
            amenities = [token.strip().lower() for token in re.split(r"[|,;/]", text) if token.strip()]

    inferred_text = title.lower()
    for amenity, keywords in AMENITY_KEYWORDS.items():
        if amenity in amenities:
            continue
        if any(keyword in inferred_text for keyword in keywords):
            amenities.append(amenity)

    return sorted(set(amenities))


def _infer_bedrooms(title: str, explicit_value: Any, room_type: str | None) -> float | None:
    """Infer bedrooms from explicit data or title heuristics."""

    explicit_numeric = _coerce_numeric(explicit_value)
    if explicit_numeric is not None:
        return explicit_numeric

    lowered = title.lower()
    for pattern, inferred in BEDROOM_PATTERNS:
        if pattern.search(lowered):
            return inferred

    normalized_room_type = (room_type or "").lower()
    if "shared room" in normalized_room_type:
        return 0.0
    if "private room" in normalized_room_type:
        return 1.0
    if "entire" in normalized_room_type:
        return 1.0
    return None


def _infer_bathrooms(explicit_value: Any, room_type: str | None) -> float | None:
    """Infer bathrooms when the dataset does not provide one directly."""

    explicit_numeric = _coerce_numeric(explicit_value)
    if explicit_numeric is not None:
        return explicit_numeric

    normalized_room_type = (room_type or "").lower()
    if "shared room" in normalized_room_type:
        return 0.5
    if "private room" in normalized_room_type:
        return 1.0
    if "entire" in normalized_room_type:
        return 1.0
    return None


def _infer_review_rating(row: dict[str, Any]) -> float | None:
    """Create a pseudo rating when explicit star ratings are unavailable."""

    existing = _coerce_numeric(
        row.get("review_rating") or row.get("rating") or row.get("review_scores_rating")
    )
    if existing is not None:
        if existing > 5.0:
            return max(0.0, min(existing / 20.0, 5.0))
        return max(0.0, min(existing, 5.0))

    review_count = _coerce_numeric(row.get("number_of_reviews")) or 0.0
    recent_reviews = _coerce_numeric(row.get("number_of_reviews_ltm")) or 0.0
    reviews_per_month = _coerce_numeric(row.get("reviews_per_month")) or 0.0

    if review_count == 0 and recent_reviews == 0 and reviews_per_month == 0:
        return 3.2

    review_signal = min(review_count / 200.0, 1.0)
    recent_signal = min(recent_reviews / 24.0, 1.0)
    activity_signal = min(reviews_per_month / 4.0, 1.0)
    return round(3.0 + 2.0 * ((0.55 * review_signal) + (0.25 * recent_signal) + (0.20 * activity_signal)), 2)


def _infer_wifi(title: str, amenities: list[str], explicit_value: Any) -> bool | None:
    """Infer WiFi availability from explicit values or keywords."""

    if explicit_value is not None and str(explicit_value).strip() != "":
        text = str(explicit_value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    if "wifi" in amenities:
        return True
    if any(token in title.lower() for token in ["wifi", "wi-fi", "internet"]):
        return True
    return None


def _infer_workspace(title: str, amenities: list[str], explicit_value: Any) -> bool | None:
    """Infer workspace availability from explicit values or title keywords."""

    if explicit_value is not None and str(explicit_value).strip() != "":
        text = str(explicit_value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    if "workspace" in amenities:
        return True
    if any(token in title.lower() for token in ["office", "desk", "workspace", "workstation"]):
        return True
    return None


def _infer_quiet_score(title: str, explicit_value: Any) -> float | None:
    """Infer a quietness score in the range [0, 1]."""

    numeric = _coerce_numeric(explicit_value)
    if numeric is not None:
        if numeric > 1.0:
            numeric = numeric / 10.0
        return max(0.0, min(numeric, 1.0))

    lowered = title.lower()
    if any(token in lowered for token in QUIET_POSITIVE):
        return 0.85
    if any(token in lowered for token in QUIET_NEGATIVE):
        return 0.35
    return 0.60


def _derive_purpose_tags(title: str, amenities: list[str], workspace: bool | None, wifi: bool | None) -> list[str]:
    """Derive purpose tags that scoring can later reuse."""

    tags: set[str] = set()
    lowered = title.lower()
    if workspace or wifi or any(token in lowered for token in ["remote", "office", "desk", "workspace"]):
        tags.add("remote_work")
    if any(token in lowered for token in ["family", "spacious", "large"]) or "laundry" in amenities:
        tags.add("family_friendly")
    if any(token in lowered for token in ["luxury", "doorman", "penthouse"]):
        tags.add("premium")
    return sorted(tags)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source columns into a leasing-friendly schema."""

    renamed = df.rename(
        columns={
            "name": "title",
            "neighbourhood": "neighborhood",
            "neighbourhood_group": "neighborhood_group",
            "baths": "bathrooms",
        }
    ).copy()

    for column in ["price", "bedrooms", "bathrooms", "area_sqft", "latitude", "longitude"]:
        if column not in renamed.columns:
            renamed[column] = None

    normalized_rows: list[dict[str, Any]] = []
    for row in renamed.to_dict(orient="records"):
        title = str(row.get("title") or "Untitled listing").strip()
        room_type = row.get("room_type")
        amenities = _normalize_amenities(row.get("amenities"), title)
        wifi = _infer_wifi(title, amenities, row.get("wifi"))
        workspace = _infer_workspace(title, amenities, row.get("workspace"))

        listing = Listing(
            id=str(row.get("id") or ""),
            title=title,
            neighborhood=str(row.get("neighborhood") or "").strip() or None,
            neighborhood_group=str(row.get("neighborhood_group") or "").strip() or None,
            price=_coerce_numeric(row.get("price")),
            bedrooms=_infer_bedrooms(title, row.get("bedrooms"), room_type),
            bathrooms=_infer_bathrooms(row.get("bathrooms"), room_type),
            area_sqft=_coerce_numeric(row.get("area_sqft")),
            amenities=amenities,
            review_rating=_infer_review_rating(row),
            wifi=wifi,
            workspace=workspace,
            quiet_score=_infer_quiet_score(title, row.get("quiet_score")),
            latitude=_coerce_numeric(row.get("latitude")),
            longitude=_coerce_numeric(row.get("longitude")),
            distance_to_target_area=_coerce_numeric(row.get("distance_to_target_area")),
            purpose_tags=_derive_purpose_tags(title, amenities, workspace, wifi),
            raw=row,
        )
        normalized_rows.append(listing.to_dict())

    return pd.DataFrame(normalized_rows)


def load_listings(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load and normalize listing rows from a CSV file."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)
    normalized = normalize_dataframe(dataframe)
    return normalized.to_dict(orient="records")
