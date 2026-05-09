"""Review comment loading and topic-aware ranking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


# Keywords used to score how relevant a comment is to each topic
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "cleanliness": ["clean", "dirty", "spotless", "messy", "filthy", "hygiene", "tidy", "dust", "immaculate", "stain"],
    "safety": ["safe", "unsafe", "security", "lock", "sketchy", "dangerous", "secure", "theft", "crime"],
    "noise": ["quiet", "noisy", "loud", "noise", "soundproof", "silent", "peaceful", "disruptive", "heard", "thin walls"],
    "wifi": ["wifi", "wi-fi", "internet", "connection", "slow", "fast internet", "broadband", "signal", "streaming"],
    "amenities": ["amenities", "gym", "kitchen", "washer", "dryer", "parking", "elevator", "pool", "laundry", "facilities"],
    "newness": ["new", "modern", "renovated", "updated", "fresh", "old", "dated", "renovation", "remodel", "worn"],
}

# Maps words in a user query → topic key
_QUERY_TO_TOPIC: dict[str, str] = {
    "clean": "cleanliness", "cleaning": "cleanliness", "cleanliness": "cleanliness",
    "dirty": "cleanliness", "hygiene": "cleanliness", "spotless": "cleanliness",
    "safe": "safety", "safety": "safety", "secure": "safety", "dangerous": "safety", "unsafe": "safety",
    "quiet": "noise", "noise": "noise", "noisy": "noise", "loud": "noise", "peaceful": "noise",
    "wifi": "wifi", "internet": "wifi", "connection": "wifi", "broadband": "wifi",
    "amenities": "amenities", "amenity": "amenities", "gym": "amenities", "laundry": "amenities",
    "new": "newness", "modern": "newness", "renovated": "newness", "updated": "newness",
}


def load_reviews_index(
    dataset_path: str | Path,
) -> dict[str, list[dict[str, str]]]:
    """Build a reviews index from the pre-baked sample_reviews column in the dataset CSV.

    The column contains a JSON array of dicts with keys: date, name, text, critical.
    Returns dict mapping listing_id (str) → list of comment dicts
    with keys: reviewer_name, date, comment, critical.
    """
    path = Path(dataset_path)
    if not path.exists():
        return {}

    try:
        import json as _json
        df = pd.read_csv(path, usecols=["id", "sample_reviews"], dtype={"id": str})
        index: dict[str, list[dict[str, str]]] = {}
        for _, row in df.iterrows():
            lid = str(row["id"])
            raw = row.get("sample_reviews")
            if not raw or (isinstance(raw, float)):
                continue
            try:
                records = _json.loads(str(raw))
                index[lid] = [
                    {
                        "reviewer_name": r.get("name", "Guest"),
                        "date": r.get("date", ""),
                        "comment": r.get("text", ""),
                        "critical": bool(r.get("critical", False)),
                    }
                    for r in records
                    if r.get("text", "").strip()
                ]
            except (ValueError, TypeError):
                continue
        return index
    except Exception:
        return {}


def detect_topics(user_query: str, soft_preferences: dict[str, Any]) -> list[str]:
    """Detect which review topics the user cares about from query text and preferences."""
    topics: set[str] = set()

    for word in re.findall(r"\b\w+\b", user_query.lower()):
        if word in _QUERY_TO_TOPIC:
            topics.add(_QUERY_TO_TOPIC[word])

    if soft_preferences.get("quiet_preference"):
        topics.add("noise")
    if soft_preferences.get("remote_work"):
        topics.add("wifi")
    if soft_preferences.get("desired_amenities"):
        topics.add("amenities")

    return list(topics)


def _comment_topic_score(comment: str, topics: list[str]) -> float:
    """Return how many topic keyword hits the comment contains."""
    if not topics:
        return 0.0
    text = comment.lower()
    score = 0.0
    for topic in topics:
        for keyword in TOPIC_KEYWORDS.get(topic, []):
            if keyword in text:
                score += 1.0
    return score


def get_listing_comments(
    listing_id: str,
    reviews_index: dict[str, list[dict[str, str]]],
    topics: list[str],
) -> dict[str, Any]:
    """Return comments split into shown (3) and more (all remaining).

    Shown (3 cards):
      - 2 reviews ranked by topic relevance (falls back to recency if no topics)
      - 1 most recent critical review pinned third (falls back to a third recent
        review if no critical reviews exist for this listing)

    More: all remaining reviews in recency order (as stored in the dataset).
    The frontend paginates these 4 at a time.

    All sorting is pure in-memory keyword matching — no model call.
    ``has_topics`` is returned so the frontend can label cards appropriately.
    """
    reviews = reviews_index.get(str(listing_id), [])
    if not reviews:
        return {"shown": [], "more": [], "total": 0, "has_topics": bool(topics)}

    recent = [r for r in reviews if not r.get("critical")]
    critical = [r for r in reviews if r.get("critical")]

    # Rank non-critical reviews by topic relevance (or keep recency order)
    if topics and recent:
        recent = sorted(
            recent,
            key=lambda r: _comment_topic_score(r["comment"], topics),
            reverse=True,
        )

    # Build the 3 shown cards
    shown: list[dict] = []
    shown.extend(recent[:2])
    if critical:
        shown.append(critical[0])
    elif len(recent) > 2:
        shown.append(recent[2])

    shown_ids = {id(r) for r in shown}
    remaining = [r for r in reviews if id(r) not in shown_ids]
    # "More" is always in recency order (as stored in the dataset)

    return {
        "shown": shown,
        "more": remaining,          # full list — no truncation; frontend paginates
        "total": len(reviews),
        "has_topics": bool(topics),
    }
