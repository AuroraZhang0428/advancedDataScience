"""Baseline 2 — Standard LLM Chatbot.

Send the user query directly to GPT-4o-mini as a plain chat prompt.
Listings are serialised into the prompt as structured text.
The LLM picks and explains recommendations with NO structured
filtering / scoring pipeline on our side.

Context-window strategy
───────────────────────
The full dataset is too large for a single prompt.  We therefore:
  1. Take a random sample of up to MAX_LISTINGS_IN_PROMPT listings.
  2. Represent each listing as a compact one-liner to minimise tokens.
  3. Ask the model to return a JSON array so we can parse its picks.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"
MAX_LISTINGS_IN_PROMPT = 60    # balance context-window vs coverage (reduced for speed)
TOP_N = 5                      # listings to ask the LLM to recommend
RANDOM_SEED = 42

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful NYC apartment-finder assistant.
You will be given a list of Airbnb listings and a user query.
Your job is to recommend the best listings that match the user's needs.

Respond ONLY with a valid JSON array (no markdown, no preamble) containing
exactly the listings you recommend, in ranked order. Each element must be an
object with these keys:
  - id          (string)
  - title       (string)
  - neighborhood (string)
  - price       (number or null)
  - bedrooms    (number or null)
  - bathrooms   (number or null)
  - review_rating (number or null)
  - amenities   (array of strings)
  - explanation (string — 1-2 sentences explaining why this listing fits)

Return between 1 and {top_n} listings. If no listings are a good fit, return
an empty array [].
""".strip()

USER_PROMPT_TEMPLATE = """User request: {query}

Available listings ({count} shown):
{listings_text}

Recommend the best matches from the list above. Return only JSON."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _listing_to_line(listing: dict[str, Any]) -> str:
    """Compact single-line representation of a listing for the prompt."""
    price = listing.get("price")
    price_str = f"${float(price):.0f}/night" if price is not None else "price unknown"

    beds = listing.get("bedrooms")
    beds_str = f"{int(beds)}BR" if beds is not None else "BR?"

    baths = listing.get("bathrooms")
    baths_str = f"{float(baths):.1f}BA" if baths is not None else "BA?"

    rating = listing.get("review_rating")
    rating_str = f"{float(rating):.2f}★" if rating is not None else "no rating"

    nbhd = (listing.get("neighborhood") or listing.get("neighborhood_group") or "Unknown")
    title = listing.get("title", "Untitled")[:60]

    amenities = listing.get("amenities") or []
    amenity_str = ", ".join(str(a) for a in amenities[:6]) if amenities else "none listed"

    rt = listing.get("room_type", "")

    return (
        f'[{listing.get("id", "?")}] "{title}" | {nbhd} | {rt} | '
        f'{beds_str}/{baths_str} | {price_str} | {rating_str} | amenities: {amenity_str}'
    )


def _sample_listings(listings: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Return a reproducible random sample of up to n listings."""
    if len(listings) <= n:
        return listings
    rng = random.Random(seed)
    return rng.sample(listings, n)


def _parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Extract and parse a JSON array from the LLM response, tolerating markdown fences."""
    # Strip common markdown wrapping
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return json.loads(cleaned)


def _serialise_recommendation(rec: dict[str, Any]) -> dict[str, Any]:
    """Normalise an LLM-returned recommendation dict for the API response."""
    price = rec.get("price")
    rating = rec.get("review_rating")
    return {
        "id":               str(rec.get("id", "")),
        "title":            rec.get("title", "Untitled"),
        "neighborhood":     rec.get("neighborhood", "Unknown area"),
        "neighborhood_group": rec.get("neighborhood_group", ""),
        "price":            float(price) if price is not None else None,
        "bedrooms":         rec.get("bedrooms"),
        "bathrooms":        rec.get("bathrooms"),
        "review_rating":    float(rating) if rating is not None else None,
        "amenities":        rec.get("amenities", []),
        "wifi":             None,
        "workspace":        None,
        "latitude":         rec.get("latitude"),
        "longitude":        rec.get("longitude"),
        "room_type":        rec.get("room_type"),
        "score":            0.0,
        "score_breakdown":  {},
        "llm_explanation":  rec.get("explanation", ""),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_llm_chatbot_baseline(
    listings: list[dict[str, Any]],
    query: str,
    api_key: str,
    top_n: int = TOP_N,
    max_listings: int = MAX_LISTINGS_IN_PROMPT,
) -> dict[str, Any]:
    """Run the LLM chatbot baseline and return a response dict.

    Args:
        listings:     Pre-loaded listing dicts.
        query:        Raw user query string.
        api_key:      OpenAI API key.
        top_n:        Max recommendations to request from the model.
        max_listings: Max listings to include in the prompt.

    Returns:
        Dict with keys:
          - recommendations: list of dicts (LLM-chosen, with explanation field)
          - method: "baseline-llm"
          - listings_shown: how many listings were given to the LLM
          - explanation: high-level summary
          - raw_response: the model's raw text (for debugging)
    """
    client = OpenAI(api_key=api_key)

    sample = _sample_listings(listings, max_listings, RANDOM_SEED)
    listings_text = "\n".join(_listing_to_line(l) for l in sample)

    system = SYSTEM_PROMPT.format(top_n=top_n)
    user_msg = USER_PROMPT_TEMPLATE.format(
        query=query,
        count=len(sample),
        listings_text=listings_text,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    raw_text = response.choices[0].message.content or ""

    try:
        parsed = _parse_llm_json(raw_text)
    except (json.JSONDecodeError, ValueError):
        # Return empty recommendations rather than crashing
        parsed = []

    recommendations = [_serialise_recommendation(r) for r in parsed if isinstance(r, dict)]

    explanation = (
        f"GPT-4o-mini selected {len(recommendations)} listing(s) from a random sample of "
        f"{len(sample)} (out of {len(listings)} total). "
        "No structured filtering or scoring was applied — the model read the listings "
        "as plain text and chose based on its own understanding of the query."
    )

    return {
        "recommendations":  recommendations,
        "method":           "baseline-llm",
        "listings_shown":   len(sample),
        "explanation":      explanation,
        "raw_response":     raw_text,
    }
