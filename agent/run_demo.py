"""Command-line demo for the LangGraph apartment leasing agent."""

from __future__ import annotations

import argparse
from typing import Any

from agent.config import DEFAULT_CONFIG
from agent.graph import build_graph


DEFAULT_QUERY = (
    "I want a 2-bedroom apartment with at least 1 bathroom, strong WiFi because I work remotely, "
    "good reviews, useful amenities, and preferably in Greenwich Village or Chelsea."
)


def _format_listing(listing: dict[str, Any]) -> str:
    """Create a compact display string for one recommendation."""

    neighborhood = listing.get("neighborhood") or listing.get("neighborhood_group") or "Unknown area"
    price = listing.get("price")
    price_text = f"${float(price):,.0f}" if price is not None else "price unavailable"
    return (
        f"- {listing.get('title', 'Untitled')} | score={float(listing.get('score', 0.0)):.2f} | "
        f"{neighborhood} | {price_text}"
    )


def main() -> None:
    """Run the compiled graph with a sample or user-supplied query."""

    parser = argparse.ArgumentParser(description="Run the apartment leasing agent demo.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="User apartment search query.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_CONFIG.dataset_path),
        help="Path to the cleaned rental dataset CSV.",
    )
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke(
        {
            "user_query": args.query,
            "dataset_path": args.dataset,
        }
    )

    print("\n=== Top Recommendations ===")
    recommendations = result.get("final_recommendations", [])
    if not recommendations:
        print("No final recommendations were produced.")
    else:
        for listing in recommendations:
            print(_format_listing(listing))

    print("\n=== Explanations ===")
    explanations = result.get("final_explanations", [])
    if not explanations:
        print("No explanation was produced yet.")
    else:
        for index, explanation in enumerate(explanations, start=1):
            print(f"{index}. {explanation}")

    print("\n=== Relaxation History ===")
    history = result.get("relaxation_history", [])
    if not history:
        print("No relaxations were needed.")
    else:
        for entry in history:
            print(entry)

    print("\n=== Need User Input ===")
    print(result.get("need_user_input", False))
    if result.get("user_question"):
        print(result["user_question"])


if __name__ == "__main__":
    main()
