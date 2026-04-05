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

    host_name = listing.get("host_name") or listing.get("raw", {}).get("host_name") or "Unknown host"
    neighborhood = listing.get("neighborhood") or listing.get("neighborhood_group") or "Unknown area"
    price = listing.get("price")
    price_text = f"${float(price):,.0f}" if price is not None else "price unavailable"
    location_context = dict(listing.get("location_context") or {})
    context_suffix = ""
    transit_parts: list[str] = []
    if location_context.get("nearby_subway_count"):
        transit_parts.append(f"subway={int(location_context['nearby_subway_count'])}")
    if location_context.get("nearby_train_count"):
        transit_parts.append(f"train={int(location_context['nearby_train_count'])}")
    if location_context.get("nearby_bus_count"):
        transit_parts.append(f"bus={int(location_context['nearby_bus_count'])}")
    if location_context.get("nearby_transit_hub_count"):
        transit_parts.append(f"hubs={int(location_context['nearby_transit_hub_count'])}")
    if location_context.get("average_commute_minutes") is not None:
        context_suffix = f" | commute~{float(location_context['average_commute_minutes']):.0f} min"
    elif transit_parts:
        context_suffix = " | " + ", ".join(transit_parts)
    elif location_context.get("nearby_transit_count") is not None:
        context_suffix = f" | transit={int(location_context['nearby_transit_count'])}"
    return (
        f"- {listing.get('title', 'Untitled')} | score={float(listing.get('score', 0.0)):.2f} | "
        f"host={host_name} | {neighborhood} | {price_text}{context_suffix}"
    )


def main() -> None:
    """Run the compiled graph with a sample or user-supplied query."""

    parser = argparse.ArgumentParser(description="Run the apartment leasing agent demo.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="User apartment search query.")
    parser.add_argument("--api-key", default=None, help="OpenAI API Key for GenAI features.")
    parser.add_argument(
        "--google-maps-api-key",
        default=None,
        help="Google Maps Platform API key for live location enrichment.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_CONFIG.dataset_path),
        help="Path to the cleaned rental dataset CSV.",
    )
    args = parser.parse_args()

    import os
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    elif "OPENAI_API_KEY" not in os.environ:
        import getpass
        key = getpass.getpass("Enter your OpenAI API key for required LLM parsing, ranking, relaxation, and explanations: ")
        if key.strip():
            os.environ["OPENAI_API_KEY"] = key.strip()
    if "OPENAI_API_KEY" not in os.environ or not str(os.environ.get("OPENAI_API_KEY", "")).strip():
        raise RuntimeError("OPENAI_API_KEY is required. Non-LLM fallbacks have been removed.")
    if args.google_maps_api_key:
        os.environ["GOOGLE_MAPS_API_KEY"] = args.google_maps_api_key
    elif "GOOGLE_MAPS_API_KEY" not in os.environ:
        import getpass
        maps_key = getpass.getpass(
            "Enter your Google Maps Platform API key for required live location enrichment: "
        )
        if maps_key.strip():
            os.environ["GOOGLE_MAPS_API_KEY"] = maps_key.strip()
    if "GOOGLE_MAPS_API_KEY" not in os.environ or not str(os.environ.get("GOOGLE_MAPS_API_KEY", "")).strip():
        raise RuntimeError("GOOGLE_MAPS_API_KEY is required. Google Maps fallback has been removed.")

    graph = build_graph()
    
    # Prompt for user query
    active_query = args.query
    if active_query == DEFAULT_QUERY:
        user_input = input("What type of apartment are you looking for? (Press Enter to use the default test query): ")
        if user_input.strip():
            active_query = user_input.strip()

    result = graph.invoke(
        {
            "user_query": active_query,
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

    print("\n=== Google Maps Enrichment ===")
    diagnostics = result.get("google_enrichment_diagnostics", {})
    if not diagnostics:
        print("No Google Maps diagnostics were recorded.")
    else:
        print(diagnostics)

    print("\n=== Need User Input ===")
    print(result.get("need_user_input", False))
    if result.get("user_question"):
        print(result["user_question"])


if __name__ == "__main__":
    main()
