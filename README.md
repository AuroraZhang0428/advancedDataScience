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
| `OPENAI_API_KEY` | Query parsing, reasoning, ranking, and explanation generation |

You must create a `.env` file in the root directory of the project and add your key:

```env
OPENAI_API_KEY=sk-...
```

Alternatively, you can export it in your terminal before running the app:

```bash
export OPENAI_API_KEY="sk-..."
python app.py
```

### Optional

| Key | Purpose |
|---|---|
| `GOOGLE_MAPS_API_KEY` | Live transit, food scene, and commute enrichment |

When this key is set, the agent can call `enrich_with_location` to fetch real-time neighborhood data (nearby subway stops, restaurants, grocery stores, and actual commute times) and use it for a second-stage reranking of the shortlist.

---

## Using the Web UI

1. Open **http://localhost:5050**
2. Type a natural-language query, for example:
   - *"2-bedroom with WiFi and good reviews in Brooklyn under $200/night"*
   - *"Quiet remote-work apartment near a subway station in Manhattan"*
   - *"Affordable private room in Williamsburg"*
3. Click **Search** — the agent reasons through results and adapts if needed (typically 60–90 seconds depending on query complexity and whether Google Maps enrichment is triggered)
4. Click any result card to see the full score breakdown, an AI-generated explanation, and an interactive **Google Maps** embed of the location
5. If any of your requirements were adjusted during the search (e.g. a neighborhood was expanded or a preference relaxed), the card and detail panel will show exactly what changed and why — **hard requirement changes** are flagged separately from **soft preference adjustments**
6. If the agent needs a decision only you can make (e.g. your budget is too low for the area), it will pause and ask you a yes/no question before resuming the search
7. Click the **⚡ Compare Methods** button at the top right to see how NestAI compares against basic filter searches and standard LLM chatbots. After a search, click **⚖️ Compare Baselines** to see results side-by-side

---

## How the Agent Works

NestAI uses a **ReAct (Reason + Act)** architecture. Instead of a fixed pipeline, the LLM orchestrator decides which tools to call and in what order, observes the results, and adapts its strategy based on what it finds.

### Pipeline

```text
User query
  → Node 1: load_data        (load and normalise the CSV dataset — runs once at startup)
  → Node 2: parse_preferences (one GPT call → structured hard constraints + soft preferences)
  → Node 3: orchestrate       (ReAct loop)
       LLM reasons → calls a tool → observes result → reasons again → ...
       until: finalize_recommendations  or  ask_user
  → return results
```

### Tools available to the agent

| Tool | What it does |
|---|---|
| `filter_listings` | Apply hard constraints; report how many listings match |
| `score_and_rank` | Score filtered listings against soft preferences; report quality assessment |
| `check_price_range` | Inspect price distribution before adjusting budget |
| `adjust_constraint` | Relax a hard constraint (price, bedrooms, bathrooms) |
| `adjust_preference` | Shift a soft preference (neighborhoods, amenities, review rating) |
| `enrich_with_location` | Add live transit / food / commute data via Google Maps |
| `ask_user` | Pause and ask the user a clarifying question |
| `finalize_recommendations` | Generate explanations and end the search |

### Decision ladder

When results are insufficient, the agent follows a strict priority order before escalating to the user:

1. **Relax soft preferences autonomously** — identifies the weakest scoring component across the top results and relaxes that preference first (expand neighborhoods, lower amenity strictness, or lower the minimum review rating). Each relaxation has a floor to prevent over-relaxation, and step sizes shrink as the floor approaches.
2. **Relax hard constraints** — only if the filtered pool is too thin (fewer than 5 listings) after soft relaxation. Checks market prices before raising the budget; reduces bedroom count autonomously only for 3BR+.
3. **Ask the user** — only when the decision genuinely requires human input (e.g. budget needs a >15% increase, or reducing from 2BR to 1BR).

### Scoring

Each listing is scored across five components:

| Component | What it measures |
|---|---|
| `review_rating` | Guest rating quality, weighted by review volume |
| `amenity_match` | How well listing amenities match requested ones |
| `purpose_alignment` | Fit for remote work and/or quiet preference |
| `neighborhood_fit` | Neighborhood name match, commute proximity, transit, food scene |
| `price_score` | Price fit relative to budget, target, or qualitative preference |

**Purpose alignment** uses a 30/70 blend: 30% from listing structural fields (wifi column, workspace column, quiet score) and 70% from guest review signals — keyword matching across review text for wifi quality, workspace quality, and noise level. Reviews are treated as ground truth over listing claims.

After deterministic scoring, a **stage-1 LLM reranker** refines the order using full listing details and reviews. If `enrich_with_location` is called, a **stage-2 LLM reranker** re-orders the shortlist using the live Google Maps neighborhood data.

### Session / resume flow

When the agent calls `ask_user`, the full search state is saved server-side with a UUID session token (30-minute TTL). The frontend shows the question with Yes / No buttons. When the user answers:
- **Yes** — the proposed constraint change is applied, and the ReAct loop resumes from where it paused
- **No** — the question is marked as declined, and the agent must find a different path (different relaxation or finalize with current results)

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
- Food-scene priority and cuisine preferences
- Remote-work and quiet preferences
- Review quality expectations (hard floors vs soft targets treated differently)
- Query-specific priority weights for ranking

---

## Folder Structure

```text
advancedDataScience/
├── app.py                          Flask API server + frontend serving + session management
├── requirements.txt
├── matched_subset_dataset.csv      NYC Airbnb listings dataset (Inside Airbnb)
├── frontend/
│   ├── index.html                  Web UI
│   ├── app.js                      Search logic, card rendering, relaxation badges
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
        ├── dataset.py              CSV loading and normalisation helpers
        ├── parser.py               LLM preference extraction (structured output)
        ├── scoring.py              Filtering, scoring, and two-stage ranking
        ├── explanation.py          Parallel explanation generation per recommendation
        ├── google_maps.py          Live neighbourhood enrichment (Places + Routes APIs)
        ├── neighborhoods.py        Static neighbourhood scoring helpers
        └── listing_links.py        Airbnb URL verification
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

**Some Airbnb links show a 404 page**
→ The dataset is a historical snapshot (Inside Airbnb). Listings that existed when the data was collected may have since been removed from Airbnb. This is a known dataset limitation and does not affect the recommendation logic.
