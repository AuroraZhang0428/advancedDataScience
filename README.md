# NestAI — Smart Apartment Finder

An agentic AI-powered apartment recommendation system built with **LangGraph**, **Flask**, and a clean web frontend. Describe what you're looking for in plain English — the agent parses your preferences, then autonomously decides how to search, adapt, and rank listings using a ReAct tool-calling loop.

> **Requires an OpenAI API key.** The agent uses `gpt-4o-mini` to parse your query, reason about search results, and adapt its strategy in real time.

---

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or higher |
| OpenAI API key | Required |
| [uv](https://docs.astral.sh/uv/) *(recommended)* or pip | latest |

---

### Mac Setup

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

Then open **http://localhost:5050** in your browser.

---

### Windows Setup

Open **Command Prompt** or **PowerShell** and run:

```bat
git clone https://github.com/AuroraZhang0428/advancedDataScience.git
cd advancedDataScience
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install flask flask-cors
python app.py
```

Then open **http://localhost:5050** in your browser.

---

### Using uv (faster alternative)

```bash
git clone https://github.com/AuroraZhang0428/advancedDataScience.git
cd advancedDataScience
uv venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
uv pip install -r requirements.txt
uv pip install flask flask-cors
python app.py
```

---

## API Keys

### Required

| Key | Purpose |
|---|---|
| `OPENAI_API_KEY` | Query parsing, reasoning, and explanation generation |

You can enter your key directly in the **Settings** panel in the web UI, or set it as an environment variable:

**Mac / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
python app.py
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-..."
python app.py
```

### Optional

| Key | Purpose |
|---|---|
| `GOOGLE_MAPS_API_KEY` | Live transit, food scene, and commute enrichment |

The agent will use the `enrich_with_location` tool automatically when this key is available.

---

## Using the Web UI

1. Open **http://localhost:5050**
2. Enter your OpenAI API key in the **Settings** panel (top right)
3. Type a natural-language query, for example:
   - *"2-bedroom with WiFi and good reviews in Brooklyn under $200/night"*
   - *"Quiet remote-work apartment near a subway station in Manhattan"*
   - *"Affordable private room in Williamsburg"*
4. Click **Search** — the agent will reason through the results and adapt if needed (~5–15 seconds)
5. Click any result card to see the full score breakdown and explanation

---

## How the Agent Works

NestAI uses a **ReAct (Reason + Act)** architecture. Instead of a fixed pipeline, the LLM orchestrator decides which tools to call and in what order, observes the results, and adapts its strategy based on what it finds.

### Agent loop

```text
User query
  → parse_preferences    (LLM structured output → hard constraints + soft preferences)
  → orchestrate          (ReAct loop)
       LLM reasons → calls a tool → observes result → reasons again → ...
       until: finalize_recommendations  or  ask_user
  → return results
```

### Tools available to the agent

| Tool | What it does |
|---|---|
| `filter_listings` | Apply hard constraints, report how many listings match |
| `score_and_rank` | Score filtered listings, report quality assessment |
| `check_price_range` | Inspect price distribution before adjusting budget |
| `adjust_constraint` | Relax a hard constraint (price, bedrooms, bathrooms) |
| `adjust_preference` | Shift a soft preference (neighborhoods, amenities, rating) |
| `enrich_with_location` | Add live transit / food / commute data via Google Maps |
| `ask_user` | Pause and ask the user a clarifying question |
| `finalize_recommendations` | Generate polished explanations and end the search |

The agent adapts autonomously — for example, if no listings match it will `check_price_range` to understand the market before deciding whether to raise the budget or ask the user, rather than blindly inflating the price.

---

## What the Agent Understands

The LLM parser extracts:

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

## Folder Structure

```text
advancedDataScience/
├── app.py                          Flask API server + frontend serving
├── requirements.txt
├── matched_subset_dataset.csv      NYC Airbnb listings dataset
├── frontend/
│   ├── index.html                  Web UI
│   ├── app.js                      Search logic and card rendering
│   └── style.css                   UI styles
└── agent/
    ├── config.py                   Scoring weights and thresholds
    ├── graph.py                    LangGraph workflow (3 nodes)
    ├── models.py                   Data models
    ├── state.py                    Typed agent state
    ├── orchestrator.py             ReAct loop — LLM decides tool call order
    ├── tools.py                    Tool schemas and executors
    ├── run_demo.py                 Command-line demo
    ├── nodes/
    │   ├── load_data.py            Load and normalise the CSV dataset
    │   ├── parse_preferences.py    Extract structured preferences from query
    │   └── orchestrate.py          LangGraph node wrapping the ReAct loop
    └── services/
        ├── dataset.py              CSV loading and normalisation
        ├── parser.py               LLM preference extraction
        ├── scoring.py              Filtering, scoring, and ranking
        ├── explanation.py          Recommendation explanation generation
        ├── google_maps.py          Live neighbourhood enrichment
        └── neighborhoods.py        Neighbourhood scoring helpers
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'flask'`**
→ Make sure your virtual environment is activated before running `python app.py`.
Mac: `source .venv/bin/activate` · Windows: `.venv\Scripts\activate`

**`OPENAI_API_KEY is required` error**
→ Enter your API key in the Settings panel in the web UI, or set it as an environment variable before starting the server.

**Port 5050 already in use**
→ Kill the existing process or change the port in `app.py` (last line: `port=5050`).

**`Dataset not found` error in the browser**
→ Make sure `matched_subset_dataset.csv` is in the project root. The file is included in the repository.

**Search returns no results**
→ Try a broader query (remove bedroom count or price limits). The dataset is NYC Airbnb listings — neighbourhood names like Brooklyn, Manhattan, Williamsburg, Chelsea work best. The agent will also try to adapt automatically before giving up.
