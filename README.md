# LangGraph Apartment Leasing Agent

This repository now includes a stateful apartment recommendation agent built with LangGraph. The agent loads a rental dataset, parses a natural-language query into structured preferences, filters listings by hard constraints, ranks them with deterministic Python scoring, evaluates whether the results are strong enough, optionally relaxes selected preferences, and returns explanations that describe both benefits and trade-offs.

## What the agent does

- Loads a cleaned listing dataset from disk with pandas.
- Separates user intent into hard constraints, soft preferences, and relaxable or semi-hard constraints.
- Applies hard filtering first.
- Scores viable listings with transparent Python logic.
- Decides whether there are enough strong matches.
- Uses a rule-based relaxation policy to loosen selected preferences or ask for clarification.
- Tracks prior attempts and relaxation history in graph state.
- Produces human-readable recommendation explanations.

## Folder structure

```text
agent/
├── __init__.py
├── config.py
├── state.py
├── graph.py
├── models.py
├── run_demo.py
├── nodes/
│   ├── __init__.py
│   ├── load_data.py
│   ├── parse_preferences.py
│   ├── filter_listings.py
│   ├── score_rank.py
│   ├── evaluate_results.py
│   ├── relax_or_ask.py
│   └── explain.py
├── services/
│   ├── __init__.py
│   ├── dataset.py
│   ├── parser.py
│   ├── scoring.py
│   └── explanation.py
└── policies/
    ├── __init__.py
    └── relaxation.py
```

## Installation

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the demo

Use the packaged demo entrypoint:

```bash
python -m agent.run_demo
```

Or pass a custom query:

```bash
python -m agent.run_demo --query "I want a 2-bedroom apartment with good WiFi, good reviews, and preferably in Chelsea."
```

By default the demo reads `listings.csv` from the repository root. You can override it:

```bash
python -m agent.run_demo --dataset path/to/cleaned_listings.csv
```

## LangGraph workflow

The graph follows this flow:

```text
load_data
  -> parse_preferences
  -> filter_listings
  -> score_rank
  -> evaluate_results
      -> explain                     when results are sufficient
      -> relax_or_ask               when results are weak
            -> filter_listings      when the policy chooses another attempt
            -> END                  when user clarification is needed
            -> explain              when the policy stops
```

The central state tracks the user query, parsed preferences, filtered and scored listings, attempt count, relaxation history, user questions, and final explanations.

## Where to plug in a real LLM

The current code works out of the box with deterministic fallbacks. The places intentionally prepared for LLM integration are:

- `agent/services/parser.py`
  Replace or augment the rule-based parser with an LLM that extracts structured intent.
- `agent/services/explanation.py`
  Add a model-based explanation polishing step after the deterministic explanation draft is produced.
- `agent/policies/relaxation.py`
  Keep the current rule-based policy or add an LLM-assisted trade-off decision module.

Search for `TODO:` comments in those files for the exact integration points.

## Design notes

- Ranking and filtering remain deterministic and inspectable.
- The graph state is simple enough for a student team to extend.
- The loader derives missing apartment-style fields when the dataset is closer to an Airbnb export than a leasing schema.
