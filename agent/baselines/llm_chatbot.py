"""Baseline 2 — Standard LLM Chatbot.

Simulates what a generic ChatGPT-style chatbot would do if you pasted your
apartment search query into it:
  1. Load the dataset and condense it into a text summary (the chatbot cannot
     run code or filter — it only sees text).
  2. Send the user's query + the data summary to GPT-4o-mini in a single
     chat completion call (no tools, no structured output, no multi-step loop).
  3. Parse the LLM's free-text response for listing IDs/titles and match them
     back to the dataset for a consistent card display.

Key intentional limitations (faithful to a plain chatbot):
  • No hard filtering — the LLM decides what "matches" based on text alone.
  • No iterative refinement or constraint relaxation.
  • No structured scoring — cannot compute composite scores.
  • Context window limit forces a sample of listings (not the full 10 k+ dataset).
  • Output is whatever the LLM says — no validation or grounding guarantee.

Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any


# ---------------------------------------------------------------------------
# Dataset summariser — converts listings to a compact text context
# ---------------------------------------------------------------------------

_MAX_LISTINGS_IN_CONTEXT = 200   # chatbot context-window budget
_SEED = 42


def _summarise_listing(listing: dict[str, Any]) -> str:
    """Single-line textual summary of a listing for the chatbot prompt."""
    price = listing.get("price")
    price_str = f"${float(price):.0f}/night" if price is not None else "price unknown"
    amenities = ", ".join(str(a) for a in listing.get("amenities", [])[:5]) or "none listed"
    return (
        f"[ID:{listing.get('id')}] {listing.get('title', 'Untitled')} | "
        f"{listing.get('neighborhood') or listing.get('neighborhood_group', 'Unknown area')} | "
        f"{price_str} | "
        f"beds:{listing.get('bedrooms')} bath:{listing.get('bathrooms')} | "
        f"rating:{listing.get('review_rating')} | "
        f"wifi:{listing.get('wifi')} workspace:{listing.get('workspace')} | "
        f"amenities: {amenities}"
    )


def _build_context(listings: list[dict[str, Any]], query: str) -> str:
    """
    Select up to _MAX_LISTINGS_IN_CONTEXT listings to show the chatbot.
    We do a very light keyword pre-filter to make the sample query-relevant,
    which is what a user would do manually when pasting data into ChatGPT.
    """
    # Extract rough price ceiling from query for a very coarse pre-filter
    price_match = re.search(r"\$\s*(\d[\d,]*)", query)
    price_ceiling = None
    if price_match:
        try:
            price_ceiling = float(price_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Light pre-filter: drop listings obviously outside budget
    if price_ceiling:
        affordable = [l for l in listings if (l.get("price") or 9999) <= price_ceiling * 1.5]
        if len(affordable) >= 50:
            listings = affordable

    # Randomly sample to fit the context budget
    rng = random.Random(_SEED)
    sample = rng.sample(listings, min(_MAX_LISTINGS_IN_CONTEXT, len(listings)))
    return "\n".join(_summarise_listing(l) for l in sample)


# ---------------------------------------------------------------------------
# LLM chatbot call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a helpful apartment search assistant for New York City.
The user will describe what they are looking for in plain English.
You have access to a sample of available NYC apartment listings provided below.
Your job is to recommend the 5 best listings for the user based on their request.

For each recommendation, output a JSON block (and nothing else after the JSON) in this exact format:
[
  {
    "id": "<listing id from the data>",
    "title": "<listing title>",
    "neighborhood": "<neighborhood>",
    "price": <price per night as number>,
    "reason": "<1–2 sentence explanation of why this listing fits the user>"
  },
  ...
]

Rules:
- Use ONLY listing IDs that appear in the provided data. Do not invent listings.
- If fewer than 5 listings match, return only those that do.
- Be honest about trade-offs. Mention price, location, and amenity fit.
- Do not include any text outside the JSON array.
"""


def _call_llm(query: str, context: str, api_key: str) -> str:
    """Single-turn GPT chat call. Returns raw response text."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed.") from exc

    client = OpenAI(api_key=api_key)
    user_message = (
        f"User request: {query}\n\n"
        f"Available listings:\n{context}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Response parser — extract recommendations from LLM free text
# ---------------------------------------------------------------------------

def _parse_llm_response(
    raw: str,
    listings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Parse the LLM's JSON response and ground each item against the real dataset.
    Returns (serialized_recommendations, explanations).
    """
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the JSON array
    arr_match = re.search(r"\[.*\]", clean, re.DOTALL)
    if not arr_match:
        return [], [f"Chatbot response could not be parsed as JSON:\n{raw[:500]}"]

    try:
        items: list[dict] = json.loads(arr_match.group(0))
    except json.JSONDecodeError:
        return [], [f"Chatbot returned malformed JSON:\n{arr_match.group(0)[:500]}"]

    listing_map = {str(l.get("id")): l for l in listings}

    serialized: list[dict[str, Any]] = []
    explanations: list[str] = []

    for item in items:
        lid = str(item.get("id", ""))
        listing = listing_map.get(lid)

        if listing:
            # Ground in real data
            price = listing.get("price")
            serialized.append({
                "id": listing.get("id", ""),
                "title": listing.get("title", item.get("title", "Untitled")),
                "neighborhood": listing.get("neighborhood") or listing.get("neighborhood_group") or item.get("neighborhood", "Unknown"),
                "neighborhood_group": listing.get("neighborhood_group", ""),
                "price": float(price) if price is not None else None,
                "bedrooms": listing.get("bedrooms"),
                "bathrooms": listing.get("bathrooms"),
                "review_rating": listing.get("review_rating"),
                "amenities": listing.get("amenities", []),
                "wifi": listing.get("wifi"),
                "workspace": listing.get("workspace"),
                "quiet_score": listing.get("quiet_score"),
                "purpose_tags": listing.get("purpose_tags", []),
                "score": 0.0,
                "score_breakdown": {},
                "llm_fit_score": None,
                "llm_rank_reason": item.get("reason", ""),
                "deterministic_score": None,
                "latitude": listing.get("latitude"),
                "longitude": listing.get("longitude"),
            })
        else:
            # LLM hallucinated an ID — include a note but try to use LLM data
            serialized.append({
                "id": lid,
                "title": item.get("title", "Unknown listing"),
                "neighborhood": item.get("neighborhood", "Unknown"),
                "neighborhood_group": "",
                "price": item.get("price"),
                "bedrooms": None,
                "bathrooms": None,
                "review_rating": None,
                "amenities": [],
                "wifi": None,
                "workspace": None,
                "quiet_score": None,
                "purpose_tags": [],
                "score": 0.0,
                "score_breakdown": {},
                "llm_fit_score": None,
                "llm_rank_reason": item.get("reason", "") + " ⚠️ Listing ID not found in dataset.",
                "deterministic_score": None,
                "latitude": None,
                "longitude": None,
            })

        reason = item.get("reason", "")
        if reason:
            title = item.get("title", f"Listing {lid}")
            explanations.append(f"**{title}**: {reason}")

    return serialized, explanations


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_llm_chatbot_baseline(
    listings: list[dict[str, Any]],
    query: str,
    api_key: str,
) -> dict[str, Any]:
    """
    End-to-end LLM chatbot baseline.

    Simulates pasting listings data + the user query into a generic chat
    interface (e.g. ChatGPT). No tools, no pipeline, no structured scoring.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the LLM chatbot baseline.")

    context = _build_context(listings, query)
    n_context = min(_MAX_LISTINGS_IN_CONTEXT, len(listings))

    raw_response = _call_llm(query, context, api_key)
    recommendations, explanations = _parse_llm_response(raw_response, listings)

    trace: list[dict[str, str]] = [
        {
            "step": "Sampled dataset into text context",
            "detail": (
                f"Selected {n_context} of {len(listings):,} listings to fit the chatbot's "
                f"context window. No filtering applied — random sample with light price pre-sort."
            ),
        },
        {
            "step": "Single-turn LLM call (GPT-4o-mini)",
            "detail": (
                "Sent user query + text listing summaries to GPT-4o-mini in one chat completion. "
                "No tools, no iteration, no structured output schema — plain conversation."
            ),
        },
        {
            "step": "Parsed free-text JSON response",
            "detail": (
                f"Extracted {len(recommendations)} recommendations from LLM output. "
                "Grounded listing IDs against the real dataset to verify accuracy."
            ),
        },
    ]

    return {
        "baseline": "llm_chatbot",
        "baseline_label": "Standard LLM Chatbot",
        "baseline_description": (
            f"Plain ChatGPT-style query: user message + {n_context} listing summaries "
            "sent in one call to GPT-4o-mini. No tools, no pipeline, no scoring, "
            "no constraint relaxation — just the model's best guess."
        ),
        "recommendations": recommendations,
        "explanations": explanations,
        "agent_trace": trace,
        "relaxation_history": [],
        "need_user_input": False,
        "user_question": None,
    }
