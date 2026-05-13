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

_MAX_ITERATIONS = 8


def _system_prompt() -> str:
    return """\
You are NestAI, an NYC apartment-finding agent. Your job: find the best listings for the user using the tools below. Always call a tool — never send plain text.

Tools:
  filter_listings          — apply hard constraints; returns match count
  score_and_rank           — score filtered listings; reports SUFFICIENT/INSUFFICIENT
  check_price_range        — inspect price distribution before adjusting budget
  adjust_constraint        — relax a hard constraint (max_price, min_bedrooms, min_bathrooms)
  adjust_preference        — soften a soft preference (neighborhoods, amenities, review_min_rating)
  enrich_with_location     — add live transit/food/commute data via Google Maps
  ask_user                 — ask ONE focused question when only the user can decide
  finalize_recommendations — output final results and end the search

── NORMAL FLOW ──────────────────────────────────────────────────────────────
  1. If the query is too vague (no location, budget, size, or purpose at all):
       → ask_user with an open-ended question. Do not search with zero signal.
  2. Call filter_listings → score_and_rank.
  3. If SUFFICIENT → finalize_recommendations immediately.
  4. If INSUFFICIENT → follow the DECISION LADDER below, then re-score once, then finalize.

── DECISION LADDER (use in order, stop at first action taken) ───────────────
  Step A — Relax soft preferences autonomously (never need user approval):
    Identify the weakest scoring component across the top results and relax that
    preference. Do NOT follow a fixed order — always relax the weakest component first.
    You may relax the same preference multiple times across separate iterations as long
    as you re-score between each relaxation and have not yet hit the floor.

    • Weak neighborhood_fit  → adjust_preference: expand preferred_neighborhoods by
                               adding adjacent or similar neighborhoods, or remove the
                               constraint entirely if the pool remains too thin after expanding.

    • Weak amenity_match     → adjust_preference: lower amenity_strictness.
                               Floor: never go below 0.75.
                               Step size: choose a value in [0.05, (current − 0.75) / 2].
                               This guarantees at least 2 steps before hitting the floor.

    • Weak review_rating     → adjust_preference: lower review_min_rating.
                               Floor determination (read the original user query carefully):
                                 - If the user set a HARD FLOOR ("no lower than X", "at least X",
                                   "minimum X", "must be X"): floor = that value. Do not relax below it.
                                 - If the user set a TARGET ("around X", "good reviews", "highly rated",
                                   "decent reviews", or any soft/vague phrasing): floor = target − 0.5.
                                 - If the user mentioned no specific rating: floor = 4.0.
                               Step size: choose a value in [0.05, (current − floor) / 2].
                               This guarantees at least 2 steps before hitting the floor.

  Step B — Relax hard constraints when the result pool is too thin:
    WHEN to consider raising max_price:
      • filter_listings returned 0 results — the budget eliminates everything.
      • filter_listings returned < 5 results AND soft relaxations (Step A) did not
        help — the pool is too thin to fill the 5 recommendations shown to the user.
    WHEN NOT to raise max_price:
      • score_and_rank returned SUFFICIENT — results are good enough; finalize immediately.
      • filter_listings returned ≥ 5 results — the budget is adequate; the pool is large
        enough to produce good recommendations without touching hard constraints.
      • Weak price_score alone is NOT a reason to raise — those listings are already
        within budget; raising the cap only adds pricier options that score worse.

    If raising max_price is warranted (< 5 results after Step A):
        - Call check_price_range to see real market prices for the same constraints.
        - If budget is within 15% of market median → adjust_constraint max_price ≤15%.
        - If budget needs >15% increase → go to Step C (ask user).
    • < 5 results due to min_bedrooms ≥ 3:
        → adjust_constraint min_bedrooms by −1 autonomously.
    • < 5 results due to min_bedrooms = 2:
        → go to Step C (ask user) — reducing to 1BR is a major lifestyle change.

  Step C — ask_user when the decision genuinely belongs to the user:
    • Budget needs to increase by >15% to get ANY results: ask_user with
      question_key="max_price" AND proposed_value=<exact dollar amount as a number>.
      CRITICAL: Always include proposed_value — the backend applies it directly when
      the user says yes. Omitting it causes a tiny fallback increase that will fail.
      Example: ask_user(question="Your budget is $80/night but private rooms near
      Williamsburg typically start at $110. Would you like to raise your budget to $110?",
      question_key="max_price", proposed_value=110)
    • Need to reduce from 2BR to 1BR: ask_user with question_key="min_bedrooms"
      AND proposed_value=1.
    • Two requirements fundamentally conflict with no good compromise: ask_user (no key).

── RULES ────────────────────────────────────────────────────────────────────
  • Never invent a constraint that was not in the original query (e.g. do not add
    min_bedrooms if the user never mentioned bedrooms).
  • After each Step A relaxation + re-score: if score_and_rank returns SUFFICIENT,
    finalize immediately — imperfect results beat asking the user to spend more.
    If still INSUFFICIENT and filter count < 5, you may enter Step B to widen the pool.
  • Never chain more than one Step A relaxation before re-scoring.
  • ask_user ends the turn. Do not call it unless truly necessary."""


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

    # Surface resolved clarifications (user said YES and constraint is already updated).
    questions_resolved = state.get("questions_resolved") or []
    if questions_resolved:
        lines.append(
            "\nUser already approved these constraint changes (already applied to hard_constraints above): "
            + ", ".join(str(q) for q in questions_resolved)
            + ". Do NOT ask about them again — proceed with the updated values."
        )

    # Surface declined clarifications so the agent knows not to repeat them.
    questions_asked = state.get("questions_asked") or []
    if questions_asked:
        lines.append(
            "\nUser has already declined these clarification options: "
            + "; ".join(str(q) for q in questions_asked)
            + ". Do NOT ask again — try a different relaxation or finalize."
        )

    return "\n".join(lines)


def run_orchestrator(state: AgentState, max_iterations: int = _MAX_ITERATIONS) -> AgentState:
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

    for _ in range(max_iterations):
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
