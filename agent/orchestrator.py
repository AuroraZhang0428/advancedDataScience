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

_MAX_ITERATIONS = 4


def _system_prompt() -> str:
    return """\
You are NestAI, an NYC apartment-finding agent. Your job: find the best listings for the user using the tools below. Always call a tool — never send plain text.

Tools:
  filter_listings        — apply hard constraints; returns match count
  score_and_rank         — score filtered listings; reports SUFFICIENT/INSUFFICIENT
  check_price_range      — inspect price distribution before adjusting budget
  adjust_constraint      — relax a hard constraint (max_price, min_bedrooms, min_bathrooms)
  adjust_preference      — soften a soft preference (neighborhoods, amenities, review_min_rating)
  enrich_with_location   — add live transit/food/commute data via Google Maps
  ask_user               — ask ONE focused question when only the user can decide
  finalize_recommendations — output final results and end the search

Strategy:
  1. Call filter_listings, then score_and_rank.
  2. If quality is SUFFICIENT, call finalize_recommendations immediately.
  3. If INSUFFICIENT, check score breakdown to see what’s weak, then adjust ONE thing and re-score.
  4. After at most one adaptation attempt, always finalize — imperfect results beat no results.
  5. Use ask_user only for genuine blockers (budget far below market, irreconcilable trade-off)."""


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
