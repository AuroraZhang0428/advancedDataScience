"""Flask API server for the LangGraph apartment leasing agent frontend."""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from agent.config import DEFAULT_CONFIG
from agent.graph import build_graph

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# Build the graph once at startup
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


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
        "deterministic_score": listing.get("deterministic_score"),
        "latitude": listing.get("latitude"),
        "longitude": listing.get("longitude"),
    }


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

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif "OPENAI_API_KEY" not in os.environ:
        os.environ.pop("OPENAI_API_KEY", None)

    try:
        graph = get_graph()
        result = graph.invoke(
            {
                "user_query": query,
                "dataset_path": dataset,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    recommendations = [_listing_to_dict(r) for r in result.get("final_recommendations", [])]
    explanations = result.get("final_explanations", [])
    relaxation_history = result.get("relaxation_history", [])
    need_user_input = result.get("need_user_input", False)
    user_question = result.get("user_question", None)

    return jsonify(
        {
            "recommendations": recommendations,
            "explanations": explanations,
            "relaxation_history": relaxation_history,
            "need_user_input": need_user_input,
            "user_question": user_question,
        }
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
