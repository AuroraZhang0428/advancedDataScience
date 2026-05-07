# NestAI — Smart Apartment Finder

A stateful, AI-powered apartment recommendation agent built with **LangGraph**, **Flask**, and a clean web frontend. Describe what you're looking for in plain English and the agent parses your preferences, filters thousands of NYC listings, scores and ranks them, and returns top picks with explanations.

> **Works without any API keys.** All LLM and Google Maps calls are optional — the system falls back to fully deterministic rule-based parsing, scoring, and explanations when no keys are configured.

---

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or higher |
| [uv](https://docs.astral.sh/uv/) *(recommended)* or pip | latest |
| Git | any |

---

### Mac Setup

Open **Terminal** and run the following commands one by one:

```bash
# 1. Clone the repository
git clone https://github.com/AuroraZhang0428/advancedDataScience.git
cd advancedDataScience

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install flask flask-cors

# 4. Start the web server
python app.py
```

Then open your browser and go to **http://localhost:5050**

To stop the server press `Ctrl + C` in the Terminal window.

---

### Windows Setup

Open **Command Prompt** (`Win + R` → type `cmd` → Enter) or **PowerShell** and run:

```bat
:: 1. Clone the repository
git clone https://github.com/AuroraZhang0428/advancedDataScience.git
cd advancedDataScience

:: 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

:: 3. Install dependencies
pip install -r requirements.txt
pip install flask flask-cors

:: 4. Start the web server
python app.py
```

Then open your browser and go to **http://localhost:5050**

To stop the server press `Ctrl + C` in the Command Prompt window.

---

### Using uv (Mac or Windows — faster alternative)

If you have [uv](https://docs.astral.sh/uv/) installed:

```bash
# Mac / Linux
git clone https://github.com/AuroraZhang0428/advancedDataScience.git
cd advancedDataScience
uv venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows
uv pip install -r requirements.txt
uv pip install flask flask-cors
python app.py
```

---

## Optional API Keys (for full AI mode)

The app runs in three modes depending on which keys are set:

| Mode | Keys needed | Features |
|---|---|---|
| **Offline** (default) | none | Rule-based parsing, deterministic scoring, plain text explanations |
| **AI mode** | `OPENAI_API_KEY` | LLM parsing, LLM reranking, AI-written explanations |
| **Full AI + Maps** | `OPENAI_API_KEY` + `GOOGLE_MAPS_API_KEY` | Everything above + live transit/food/commute enrichment |

You can enter your OpenAI API key directly in the **Settings** panel inside the web UI — no environment setup required.

To set keys via environment variables:

**Mac / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_MAPS_API_KEY="AIza..."
python app.py
```

**Windows (Command Prompt):**
```bat
set OPENAI_API_KEY=sk-...
set GOOGLE_MAPS_API_KEY=AIza...
python app.py
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:GOOGLE_MAPS_API_KEY="AIza..."
python app.py
```

---

## Using the Web UI

1. Open **http://localhost:5050** in your browser
2. Type a natural-language query into the search box, for example:
   - *"2-bedroom with WiFi and good reviews in Brooklyn under $200/night"*
   - *"Quiet remote-work apartment near a subway station in Manhattan"*
   - *"Affordable private room in Williamsburg"*
3. Or click one of the **quick-pick chips** below the search box
4. Click **Search** and wait for the AI agent to process your request (~5–15 seconds)
5. Click any result card to see the full score breakdown and explanation
6. Use the **Settings** panel (top right) to add an API key or change the dataset

---

## What The Agent Understands

The parser extracts:

- Bedroom and bathroom minimums
- `max_price`, `target_price`, `price_floor` with nightly vs monthly period
- Qualitative price preference (`cheap`, `moderate`, `expensive`)
- Preferred neighborhoods and areas
- Requested amenities (WiFi, workspace, gym, laundry, parking, etc.)
- Work / school / commute destinations
- Transit priority and preferred modes (subway, train, bus)
- Food-scene priority
- Remote-work and quiet preferences
- Review quality expectations
- Query-specific priority weights for ranking

---

## Pipeline Overview

```text
load_data
  → parse_preferences          (LLM or rule-based fallback)
  → filter_listings            (deterministic hard constraints)
  → score_rank                 (deterministic scoring + optional LLM reranking)
  → enrich_candidates          (optional Google Maps enrichment)
  → evaluate_results
      → explain                when results are sufficient
      → relax_or_ask           when results are weak
            → filter_listings  on retry
            → END              when user clarification is needed
            → explain          when agent stops
```

---

## Folder Structure

```text
advancedDataScience/
├── app.py                        Flask API server + frontend serving
├── requirements.txt
├── matched_subset_dataset.csv    NYC Airbnb listings dataset
├── frontend/
│   ├── index.html                Web UI
│   ├── app.js                    Search logic and card rendering
│   └── style.css                 UI styles
└── agent/
    ├── config.py                 Scoring weights and thresholds
    ├── graph.py                  LangGraph workflow definition
    ├── models.py                 Data models
    ├── state.py                  Typed agent state
    ├── run_demo.py               Command-line demo
    ├── nodes/
    │   ├── load_data.py
    │   ├── parse_preferences.py
    │   ├── filter_listings.py
    │   ├── score_rank.py
    │   ├── enrich_candidates.py
    │   ├── evaluate_results.py
    │   ├── relax_or_ask.py
    │   └── explain.py
    ├── policies/
    │   └── relaxation.py         Adaptive relaxation policy
    └── services/
        ├── dataset.py            CSV loading and normalization
        ├── parser.py             Preference extraction (LLM + rule-based)
        ├── scoring.py            Filtering, scoring, and ranking
        ├── explanation.py        Recommendation explanation generation
        ├── google_maps.py        Live neighborhood enrichment
        └── neighborhoods.py      Neighborhood scoring helpers
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'flask'`**
→ Make sure your virtual environment is activated before running `python app.py`.
Mac: `source .venv/bin/activate` · Windows: `.venv\Scripts\activate`

**Port 5050 already in use**
→ Kill the existing process or change the port in `app.py` (last line: `port=5050`).

**`Dataset not found` error in the browser**
→ Make sure `matched_subset_dataset.csv` is in the project root directory. The file is included in the repository.

**Search returns no results**
→ Try a broader query (remove bedroom count or price limits). The dataset is NYC Airbnb listings — neighborhood names like Brooklyn, Manhattan, Williamsburg, Chelsea work best.

---

## Architecture Notes

- The system is a **hybrid pipeline**: deterministic retrieval scaffolding + optional LLM judgment.
- All LLM calls (parsing, ranking, relaxation, explanation) have deterministic rule-based fallbacks — the app works fully offline.
- Google Maps enrichment adds live transit, food, grocery, and commute data but is entirely optional.
- The shortlist size is 30; the top 5 are returned as final recommendations.
