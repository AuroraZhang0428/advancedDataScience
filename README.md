# LangGraph Apartment Recommendation Agent

This project is a stateful apartment recommendation agent built with LangGraph, OpenAI, and Google Maps Platform.

It takes a natural-language housing query, uses an LLM to interpret the user's intent, retrieves and ranks candidate listings, enriches them with live geographic context, and returns final recommendations with explanations.

## Current State

The current system is an LLM-driven hybrid pipeline:

- LLM-only query parsing
- deterministic hard filtering
- deterministic shortlist retrieval scoring
- LLM stage-one ranking over the shortlist
- Google Maps live enrichment
- LLM stage-two reranking using live transit / food / grocery / commute evidence
- LLM-driven relaxation or clarification decisions
- LLM-written final explanations

Non-LLM runtime fallbacks have been removed. The demo now expects:

- `OPENAI_API_KEY`
- `GOOGLE_MAPS_API_KEY`

## What The Agent Understands

The parser can currently extract:

- bedroom and bathroom minimums
- `max_price`
- `target_price`
- `price_floor`
- nightly vs monthly price period
- qualitative price preference like `cheap`, `moderate`, or `expensive`
- preferred neighborhoods
- amenities
- work / school / commute destinations
- transit priority and preferred transit modes
- food-scene priority
- remote-work and quiet preferences
- review expectations
- query-specific priority weights for ranking

## High-Level Pipeline

```text
load_data
  -> parse_preferences
  -> filter_listings
  -> score_rank
  -> enrich_candidates
  -> evaluate_results
      -> explain                     when results are sufficient
      -> relax_or_ask               when results are weak
            -> filter_listings      when the agent retries
            -> END                  when user clarification is needed
            -> explain              when the agent stops
```

## How Ranking Works

### 1. Hard filtering

Listings are first filtered by strict constraints such as:

- minimum bedrooms
- minimum bathrooms
- `max_price`
- room type

### 2. Deterministic shortlist retrieval

After filtering, the code computes a coarse retrieval score across the whole filtered dataset.

This retrieval score uses:

- review quality
- amenity match, if amenities were requested
- purpose alignment, if the user cares about remote work or quietness
- neighborhood fit, if the query includes area / commute / transit / food intent
- price fit, using `target_price`, `max_price`, `price_floor`, and qualitative price preference when relevant

Important current behavior:

- shortlist size is `30`
- query-specific priority weights are inferred by the LLM once, then reused consistently across all listings for that query
- non-applicable components are excluded from the retrieval score instead of being treated as perfect matches

### 3. Stage-one LLM ranking

The top 30 shortlisted listings are passed to the LLM, which:

- directly judges holistic fit
- returns component scores
- returns an overall fit score
- reranks the shortlist

### 4. Google Maps enrichment

Shortlisted listings are enriched with live location context from Google Maps Platform:

- nearby subway / train / bus / transit hub results
- nearby food venues
- nearby grocery venues
- commute times to named destinations

### 5. Stage-two LLM reranking

After enrichment, the LLM reranks the candidates again using the live geographic evidence.

Current design intent:

- live neighborhood evidence is primary
- the earlier shortlist score is treated only as coarse retrieval context
- stage two is no longer strongly anchored to a pre-blended prior score

## Relaxation And Clarification

When results are weak, the agent evaluates whether:

- there are enough strong matches
- the top candidates match the target price well enough
- the top candidates satisfy a soft price floor well enough
- there are enough viable results to stop

Then the relaxation policy decides whether to:

- relax a soft preference
- ask the user for clarification
- stop and explain the best available options

The policy is currently LLM-driven, but it still chooses from a bounded set of allowed actions prepared by the code.

## Explanations

Final recommendations include:

- listing title
- host name
- neighborhood
- price
- score breakdown
- live neighborhood evidence
- trade-offs

Then the explanation draft is rewritten by the LLM into more natural recommendation text.

## Folder Structure

```text
agent/
|-- config.py
|-- graph.py
|-- models.py
|-- run_demo.py
|-- state.py
|-- nodes/
|   |-- enrich_candidates.py
|   |-- evaluate_results.py
|   |-- explain.py
|   |-- filter_listings.py
|   |-- load_data.py
|   |-- parse_preferences.py
|   |-- relax_or_ask.py
|   `-- score_rank.py
|-- policies/
|   `-- relaxation.py
`-- services/
    |-- dataset.py
    |-- explanation.py
    |-- google_maps.py
    |-- neighborhoods.py
    |-- parser.py
    `-- scoring.py
```

## Installation

Create a Python 3.11+ environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Running The Demo

From the project root:

```bash
python -m agent.run_demo
```

You can also pass keys and a custom query explicitly:

```bash
python -m agent.run_demo --api-key YOUR_OPENAI_KEY --google-maps-api-key YOUR_GOOGLE_KEY --query "I want to live in Manhattan around $500 a night and be very close to multiple subway stations."
```

The demo will prompt for any missing required keys.

## Notes On The Current Architecture

- This is no longer a pure rule-based recommender.
- The system now uses the LLM for parsing, ranking, adaptive decision-making, and explanation generation.
- The shortlist generation step is still deterministic and is used as retrieval scaffolding before the LLM performs deeper judgment.
- Google Maps enrichment is required in the current runtime flow.

## Next Steps

- Improve the reasoning stage so the agent is better at deciding when to ask the user for clarification versus when to relax preferences automatically.
- In particular, very unrealistic price points should trigger clarification more reliably instead of allowing the agent to expand neighborhoods or make a weaker relaxation first.
- Add a separate user-facing explanation field when clarification is requested, so the agent explicitly states why new input is needed and what was wrong with the previous constraints.
