"""ReAct orchestrator: the LLM decides which tools to call and in what order.

Flow:
  1. Build a system prompt explaining the agent's job and available tools.
  2. Send the user query + parsed preferences as context.
  3. Loop: call the LLM → execute any tool calls → feed observations back.
  4. Stop when the LLM calls finalize_recommendations or ask_user, or hits
     the iteration cap.

Requires OPENAI_API_KEY to be set.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent.state import AgentState
from agent.tools import TERMINAL_TOOLS, TOOL_SCHEMAS, execute_tool

_MAX_ITERATIONS = 12


def _system_prompt() -> str:
    return """\
You are an intelligent NYC Airbnb listing agent. Your job is to find the best \
matching listings for the user by reasoning about evidence — not by following a \
fixed pipeline.

You have access to a full Airbnb dataset and the following tools:

  • filter_listings         — apply hard constraints; tells you how many listings survive
  • score_and_rank          — score filtered listings; reports quality (SUFFICIENT / INSUFFICIENT)
                              and per-listing component scores (review, price, neighborhood, etc.)
  • check_price_range       — shows price distribution in the dataset for a bedroom tier;
                              use this to understand whether a budget is realistic before touching it
  • adjust_constraint       — changes a hard constraint (max_price, min_bedrooms, min_bathrooms);
                              call filter_listings again after to see the new count
  • adjust_preference       — changes a soft preference (preferred_neighborhoods, desired_amenities,
                              review_min_rating, amenity_strictness); call score_and_rank after
  • enrich_with_location    — adds live transit, food, and commute data via Google Maps to the
                              shortlisted listings; useful when location context matters
  • ask_user                — pauses the search and asks the user one question; use when a real
                              trade-off requires a human decision
  • finalize_recommendations — produces final output and ends the search

RULE: You MUST always call a tool. Never send a plain text response without a tool \
call — that silently ends the search with no results for the user.

RULE: The user is always better served by honest imperfect results than by no results. \
After a reasonable number of adaptation attempts, call finalize_recommendations even \
if quality is still INSUFFICIENT.

How to use your tools:
- score_and_rank tells you not just whether results are good, but *why* — the component \
  breakdown (review, price, neighborhood, purpose, amenity) shows exactly which dimension \
  is weak. Use that evidence to decide what, if anything, to change.
- adjust_preference can soften any soft constraint without distorting the user's intent much.
- adjust_constraint changes hard rules — bigger impact, so reason carefully before using it.
- check_price_range gives you market context before touching a budget.
- ask_user is for situations where you cannot make a reasonable inference on the user's \
  behalf — for example: the budget is far below market rate and raising it significantly \
  would change the nature of the search; the user named a neighborhood with no listings \
  and you don't know which nearby area they'd accept; or two very different directions are \
  equally valid and only the user can choose. Do not ask about things you can resolve \
  yourself (e.g. minor preference relaxation). Ask at most one targeted question per turn, \
  and phrase it so the user can answer concisely.

Reason from what you observe. You choose which tools to call, in what order, and when \
to stop adapting and finalize."""


def _context_message(state: AgentState) -> str:
    hc = state.get("hard_constraints") or {}
    sp = state.get("soft_preferences") or {}
    n_listings = len(state.get("listings") or [])

    lines: list[str] = [
        f"User query: {state.get('user_query', '')}",
        "",
        "Parsed hard constraints:",
    ]
    for k, v in hc.items():
        if v is not None:
            lines.append(f"  {k}: {v}")

    lines.append("\nParsed soft preferences:")
    for k, v in sp.items():
        if v is not None and v != [] and v != {}:
            lines.append(f"  {k}: {v}")

    lines.append(f"\nDataset: {n_listings} listings available.")
    return "\n".join(lines)


def run_orchestrator(state: AgentState) -> AgentState:
    """Run the ReAct loop. Requires OPENAI_API_KEY to be set."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Set it in the Settings panel or as an environment variable."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Run: pip install openai") from exc

    client = OpenAI(api_key=api_key)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _context_message(state)},
    ]

    # Mutable working copy — tools update this as they execute
    working: dict[str, Any] = dict(state)
    working.setdefault("relaxation_history", [])
    working.setdefault("questions_asked", [])
    working.setdefault("need_user_input", False)
    working.setdefault("user_question", None)

    for _ in range(_MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
        )

        assistant_msg = response.choices[0].message

        # Preserve the full message object for history (convert to dict for JSON safety)
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            **(
                {
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ]
                }
                if assistant_msg.tool_calls
                else {}
            ),
        })

        if not assistant_msg.tool_calls:
            # LLM stopped without calling a tool — shouldn't happen with a well-formed
            # system prompt, but handle gracefully
            break

        terminal_reached = False
        for tc in assistant_msg.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, ValueError):
                tool_args = {}

            observation, updates = execute_tool(tool_name, tool_args, working)
            working.update(updates)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": observation,
            })

            if tool_name in TERMINAL_TOOLS:
                terminal_reached = True

        if terminal_reached:
            break

    working["orchestrator_messages"] = messages

    # Safety net: if the loop exhausted iterations without the agent finalizing,
    # finalize with whatever scored results exist so the user always gets output.
    if "final_recommendations" not in working and working.get("scored_listings"):
        _, updates = execute_tool("finalize_recommendations", {}, working)
        working.update(updates)

    return working
