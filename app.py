"""Flask API server for the LangGraph apartment leasing agent frontend."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path

# Load .env before anything else so OPENAI_API_KEY is available to all services
load_dotenv()

from agent.config import DEFAULT_CONFIG
from agent.graph import build_graph
from agent.baselines.filter_search import run_filter_baseline
from agent.baselines.llm_chatbot import run_llm_chatbot_baseline
from agent.services.dataset import load_listings
from agent.services.reviews import detect_topics, get_listing_comments, load_reviews_index
from agent.services.listing_links import attach_listing_links
from agent.orchestrator import run_orchestrator

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

PROJECT_ROOT = Path(__file__).resolve().parent

# Build the graph once at startup
_graph = None

# Reviews index: listing_id → list of comment dicts (loaded lazily on first search)
_reviews_index: dict = {}
_reviews_loaded = False


def _ensure_reviews_loaded() -> None:
    global _reviews_index, _reviews_loaded
    if _reviews_loaded:
        return
    dataset_path = PROJECT_ROOT / str(DEFAULT_CONFIG.dataset_path)
    _reviews_index = load_reviews_index(dataset_path)
    _reviews_loaded = True

# ── In-memory session store: {session_id: {state, created_at}} ───────────────
_sessions: dict[str, dict] = {}
_SESSION_TTL_MINUTES = 30


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _purge_old_sessions() -> None:
    """Remove sessions older than SESSION_TTL_MINUTES."""
    cutoff = datetime.utcnow() - timedelta(minutes=_SESSION_TTL_MINUTES)
    expired = [sid for sid, s in _sessions.items() if s["created_at"] < cutoff]
    for sid in expired:
        del _sessions[sid]


def _listing_to_dict(listing: dict[str, Any]) -> dict[str, Any]:
    """Serialize a listing dict with safe float handling."""
    price = listing.get("price")
    score = listing.get("score", 0.0)
    score_breakdown = listing.get("score_breakdown", {})

    return {
        "id": listing.get("id", ""),
        "title": listing.get("title", "Untitled"),
        "neighborhood": listing.get("neighborhood") or listing.get("neighborhood_group") or "Unknown area",
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
        "score": float(score),
        "score_breakdown": {k: float(v) for k, v in score_breakdown.items()},
        "llm_fit_score": listing.get("llm_fit_score"),
        "llm_rank_reason": listing.get("llm_rank_reason"),
        "latitude": listing.get("latitude"),
        "longitude": listing.get("longitude"),
    }



def _format_pref_value(value: Any) -> Any:
    """Convert preference values into JSON-safe, frontend-friendly values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v not in (None, "", [], {})]
    if isinstance(value, dict):
        return {str(k): _format_pref_value(v) for k, v in value.items() if v not in (None, "", [], {})}
    return str(value)


def _compact_dict(data: Any) -> dict[str, Any]:
    """Remove empty values so the frontend only displays meaningful preferences."""
    if not isinstance(data, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in data.items():
        formatted = _format_pref_value(value)
        if formatted not in (None, "", [], {}):
            compact[key] = formatted
    return compact


def _build_detected_preferences(result: dict[str, Any]) -> dict[str, Any]:
    """Expose what the agent understood from the user's natural-language query."""
    return {
        "raw_preferences": _compact_dict(result.get("raw_preferences", {})),
        "hard_constraints": _compact_dict(result.get("hard_constraints", {})),
        "soft_preferences": _compact_dict(result.get("soft_preferences", {})),
    }


def _build_agent_trace(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a rich per-step trace from the orchestrator's message history.

    Each step corresponds to one tool call made by the ReAct orchestrator.
    Returns a list of dicts with keys: tool, step, category, args, observation, detail.
    Falls back to a simplified human-readable trace if no message history is present.
    """
    import json as _json

    TOOL_LABELS: dict[str, str] = {
        "filter_listings":        "Filtered listings",
        "score_and_rank":         "Scored & ranked candidates",
        "check_price_range":      "Checked price distribution",
        "adjust_constraint":      "Adjusted hard constraint",
        "adjust_preference":      "Adjusted soft preference",
        "enrich_with_location":   "Enriched with location data",
        "ask_user":               "Asked for clarification",
        "finalize_recommendations": "Finalized recommendations",
    }
    TOOL_CATEGORY: dict[str, str] = {
        "filter_listings":          "search",
        "score_and_rank":           "score",
        "check_price_range":        "explore",
        "adjust_constraint":        "adapt",
        "adjust_preference":        "adapt",
        "enrich_with_location":     "enrich",
        "ask_user":                 "clarify",
        "finalize_recommendations": "complete",
    }

    messages: list[dict] = result.get("orchestrator_messages", []) or []

    # Map tool_call_id → observation string (role == "tool" messages)
    tool_obs: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id:
                tool_obs[tc_id] = msg.get("content", "")

    trace: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            tool_name: str = fn.get("name", "unknown")
            try:
                args: dict = _json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            tc_id = tc.get("id", "")
            observation = tool_obs.get(tc_id, "")
            # First line of the observation serves as the collapsed summary
            detail = observation.split("\n")[0].strip() if observation else ""
            trace.append({
                "tool":        tool_name,
                "step":        TOOL_LABELS.get(tool_name, tool_name.replace("_", " ").title()),
                "category":    TOOL_CATEGORY.get(tool_name, "search"),
                "args":        args,
                "observation": observation,
                "detail":      detail,
            })

    if trace:
        return trace

    # ── Fallback: no orchestrator_messages (older state / non-orchestrator path) ──
    fallback: list[dict[str, Any]] = [
        {
            "step": "Parsed preferences",
            "detail": "Converted the natural-language request into structured search constraints and preferences.",
        }
    ]
    filtered_count = len(result.get("filtered_listings", []) or [])
    fallback.append({
        "step": "Filtered listings",
        "detail": (
            f"Applied hard constraints and kept {filtered_count} matching listings."
            if filtered_count else "Applied hard constraints to narrow the dataset."
        ),
    })
    scored_count = len(result.get("scored_listings", []) or result.get("shortlisted_listings", []) or [])
    fallback.append({
        "step": "Scored and ranked candidates",
        "detail": (
            f"Ranked {scored_count} candidates using price, location, reviews, amenities, and lifestyle fit."
            if scored_count else "Compared remaining listings against the user's soft preferences."
        ),
    })
    diagnostics = result.get("results_diagnostics", {}) or {}
    good_count = diagnostics.get("good_count")
    fallback.append({
        "step": "Checked result quality",
        "detail": (
            f"The evaluator found {good_count} high-quality matches before finalizing or adapting the search."
            if good_count is not None else "The agent evaluated whether the results satisfied the search goal."
        ),
    })
    for item in result.get("relaxation_history", []) or []:
        action = str(item.get("action") or "Adjusted search").replace("_", " ").title()
        change = item.get("change")
        reason = item.get("reason") or "The agent adapted the search strategy to improve results."
        fallback.append({"step": action, "detail": f"{change}. {reason}" if change else reason})
    if result.get("need_user_input"):
        fallback.append({
            "step": "Asked for clarification",
            "detail": result.get("user_question") or "The agent needs one more user decision before continuing.",
        })
    else:
        final_count = len(result.get("final_recommendations", []) or [])
        fallback.append({
            "step": "Finalized recommendations",
            "detail": f"Generated {final_count} ranked recommendation{'s' if final_count != 1 else ''} with explanations.",
        })
    return fallback

def _apply_user_answer(state: dict[str, Any], question_key: str | None, answer: str) -> dict[str, Any]:
    """Update agent state based on the user's yes/no answer to a clarification question."""
    import re as _re

    state = dict(state)
    is_yes = answer.lower().strip() in {"yes", "y", "ok", "sure", "yeah", "yep", "okay", "sure thing"}

    # Save question text BEFORE clearing it — needed for the regex fallback below.
    saved_question_text: str = state.get("user_question") or ""

    state["need_user_input"] = False
    state["user_question"] = None

    if not is_yes:
        # User declined — record the constraint key so the orchestrator knows not to
        # ask again without polluting the context with the full question text.
        questions_asked = list(state.get("questions_asked", []))
        if question_key and question_key not in questions_asked:
            questions_asked.append(question_key)
        state["questions_asked"] = questions_asked
        return state

    # User accepted — replace questions_asked (which contains stale question texts from
    # the first run) with the resolved key so the resumed orchestrator context says
    # "already resolved: max_price" rather than "declined: <question text>".
    state["questions_asked"] = []
    resolved = list(state.get("questions_resolved", []))
    if question_key and question_key not in resolved:
        resolved.append(question_key)
    state["questions_resolved"] = resolved

    # Apply the proposed constraint change.
    hard = dict(state.get("hard_constraints", {}))

    if question_key == "min_bedrooms":
        proposed = state.get("question_proposed_value")
        current = hard.get("min_bedrooms")
        if proposed is not None:
            hard["min_bedrooms"] = max(int(float(proposed)), 0)
        elif current is not None:
            hard["min_bedrooms"] = max(int(float(current)) - 1, 0)

    elif question_key == "max_price":
        proposed = state.get("question_proposed_value")

        # Fallback: the LLM sometimes omits proposed_value even though it cited a specific
        # dollar amount in the question text.  Parse the LARGEST $ figure mentioned so we
        # jump straight to the proposed budget instead of inching up ~15% at a time.
        if proposed is None and saved_question_text:
            amounts = _re.findall(r"\$(\d+(?:\.\d+)?)", saved_question_text)
            if amounts:
                proposed = max(float(a) for a in amounts)

        current = hard.get("max_price")
        if proposed is not None:
            hard["max_price"] = round(float(proposed), 2)
        elif current is not None:
            # Last-resort fallback: a 40% jump is enough to matter without being absurd.
            hard["max_price"] = round(float(current) * 1.40, 2)

    state["hard_constraints"] = hard

    # Flush stale search intermediates so the resumed orchestrator re-runs
    # filter_listings and score_and_rank with the updated constraints.
    state["filtered_listings"] = []
    state["scored_listings"] = []
    state["shortlisted_listings"] = []
    state["sufficient_results"] = False

    return state


def _resume_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    """Re-run the orchestrator with updated state after a user clarification.

    The state already has listings + parsed preferences; we just re-invoke
    the ReAct orchestrator so it can re-run tool calls with the updated
    hard_constraints or soft_preferences. The full iteration budget is used
    so the agent has room to filter → score → optionally adjust → finalize.
    """
    try:
        result = run_orchestrator(state)
        return result
    except Exception:
        # Fall back: return state as-is so the frontend can show what we have
        return state


def _build_response(result: dict[str, Any], user_query: str = "") -> dict[str, Any]:
    """Convert final graph state into the JSON response shape."""
    _ensure_reviews_loaded()
    topics = detect_topics(user_query, result.get("soft_preferences", {}))

    # _listing_to_dict produces a clean copy with only display fields —
    # the original scored listing dicts (used for ranking) are never touched
    # from this point on. airbnb_url is added only to these display copies.
    recommendations = []
    for r in result.get("final_recommendations", []):
        rec = _listing_to_dict(r)
        rec["comments"] = get_listing_comments(str(rec["id"]), _reviews_index, topics)
        recommendations.append(rec)

    # Link lookup runs after ranking is fully frozen and operates only on the
    # display-copy dicts above. It cannot influence scores or ordering.
    attach_listing_links(recommendations)
    explanations = result.get("final_explanations", [])
    relaxation_history = result.get("relaxation_history", [])
    need_user_input = result.get("need_user_input", False)
    user_question = result.get("user_question", None)

    response: dict[str, Any] = {
        "recommendations": recommendations,
        "explanations": explanations,
        "relaxation_history": relaxation_history,
        "need_user_input": need_user_input,
        "user_question": user_question,
        "detected_preferences": _build_detected_preferences(result),
        "agent_trace": _build_agent_trace(result),
    }

    # If the agent needs clarification, save state and return a session token
    if need_user_input and user_question:
        _purge_old_sessions()
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "state": dict(result),
            "created_at": datetime.utcnow(),
        }
        response["session_id"] = session_id
        # question_key is set directly in state by _ask_user when the question
        # is about a specific constraint (max_price, min_bedrooms) so the
        # frontend shows yes/no buttons and the backend knows which constraint to update.
        response["question_key"] = result.get("question_key")
        response["question_proposed_value"] = result.get("question_proposed_value")

    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    api_key = data.get("api_key", "").strip()
    dataset = data.get("dataset", str(DEFAULT_CONFIG.dataset_path))

    if not query:
        return jsonify({"error": "Query is required."}), 400

    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.exists():
        return jsonify({"error": f"Dataset not found: {dataset_path}"}), 400

    # api_key from the request is ignored — key lives in .env on the server
    # (os.environ["OPENAI_API_KEY"] is already set by load_dotenv at startup)

    try:
        graph = get_graph()
        result = graph.invoke(
            {
                "user_query": query,
                "dataset_path": str(dataset_path),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(_build_response(result, user_query=query))


@app.route("/api/clarify", methods=["POST"])
def clarify():
    """Accept the user's answer to a clarification question and resume the pipeline."""
    data = request.get_json(force=True)
    session_id = data.get("session_id", "").strip()
    answer = data.get("answer", "").strip()
    question_key = data.get("question_key")
    proposed_value = data.get("proposed_value")

    if not session_id:
        return jsonify({"error": "session_id is required."}), 400
    if not answer:
        return jsonify({"error": "answer is required."}), 400

    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found or expired. Please start a new search."}), 404

    saved_state = dict(session["state"])

    # If the frontend sends a proposed_value (the exact amount the orchestrator proposed),
    # store it on the state so _apply_user_answer can use it directly instead of
    # falling back to a fixed percentage increase.
    if proposed_value is not None:
        saved_state["question_proposed_value"] = proposed_value

    # Apply the user's answer to update constraints
    updated_state = _apply_user_answer(saved_state, question_key, answer)

    try:
        result = _resume_pipeline(updated_state)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    # If the resumed pipeline needs yet another clarification, the session is
    # updated in _build_response with a fresh session_id.
    del _sessions[session_id]

    original_query = saved_state.get("user_query", "")
    return jsonify(_build_response(result, user_query=original_query))


@app.route("/api/search/baseline-filter", methods=["POST"])
def search_baseline_filter():
    """Baseline 1 — filter-based search: regex parsing + hard filters + price sort."""
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    dataset = data.get("dataset", str(DEFAULT_CONFIG.dataset_path))

    if not query:
        return jsonify({"error": "Query is required."}), 400

    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.exists():
        return jsonify({"error": f"Dataset not found: {dataset_path}"}), 400

    try:
        listings = load_listings(str(dataset_path))
        result = run_filter_baseline(listings, query)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/api/search/baseline-llm", methods=["POST"])
def search_baseline_llm():
    """Baseline 2 — standard LLM chatbot: single-turn GPT call on sampled data."""
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    api_key = data.get("api_key", "").strip()
    dataset = data.get("dataset", str(DEFAULT_CONFIG.dataset_path))

    if not query:
        return jsonify({"error": "Query is required."}), 400

    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.exists():
        return jsonify({"error": f"Dataset not found: {dataset_path}"}), 400

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not resolved_key:
        return jsonify({"error": "OPENAI_API_KEY is required for the LLM chatbot baseline."}), 400

    try:
        listings = load_listings(str(dataset_path))
        result = run_llm_chatbot_baseline(listings, query, resolved_key)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "active_sessions": len(_sessions)})


if __name__ == "__main__":
    # use_reloader=False prevents Flask from watching .uv-cache and restarting
    # in an infinite loop when uv writes package files inside the project folder.
    app.run(debug=True, port=5050, use_reloader=False)
