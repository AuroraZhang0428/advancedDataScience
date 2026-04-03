"""LangGraph node for adaptive relaxation or clarification."""

from __future__ import annotations

from typing import Any

from agent.policies.relaxation import choose_relaxation_action
from agent.state import AgentState


def _merge_preferences(current: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge preference updates without mutating the original dictionary."""

    merged = dict(current)
    merged.update(updates)
    return merged


def relax_or_ask_node(state: AgentState) -> AgentState:
    """Choose the next adaptation step when current results are weak."""

    decision = choose_relaxation_action(state)
    response: AgentState = {
        "latest_decision": decision.to_dict(),
        "need_user_input": False,
        "user_question": None,
    }

    if decision.action == "relax_soft":
        new_soft_preferences = _merge_preferences(
            state.get("soft_preferences", {}),
            decision.change.get("soft_preferences", {}),
        )
        history_entry = {
            "attempt": int(state.get("attempt_count", 0)) + 1,
            "action": decision.action,
            "change": decision.change,
            "reason": decision.reason,
        }
        response.update(
            {
                "attempt_count": int(state.get("attempt_count", 0)) + 1,
                "soft_preferences": new_soft_preferences,
                "relaxation_history": list(state.get("relaxation_history", [])) + [history_entry],
            }
        )
        return response

    if decision.action == "ask_user":
        question_key = decision.change.get("question_key")
        questions_asked = list(state.get("questions_asked", []))
        if question_key and question_key not in questions_asked:
            questions_asked.append(question_key)

        history_entry = {
            "attempt": int(state.get("attempt_count", 0)),
            "action": decision.action,
            "change": decision.change,
            "reason": decision.reason,
        }
        response.update(
            {
                "need_user_input": True,
                "user_question": decision.user_question,
                "questions_asked": questions_asked,
                "relaxation_history": list(state.get("relaxation_history", [])) + [history_entry],
            }
        )
        return response

    history_entry = {
        "attempt": int(state.get("attempt_count", 0)),
        "action": decision.action,
        "change": decision.change,
        "reason": decision.reason,
    }
    response["relaxation_history"] = list(state.get("relaxation_history", [])) + [history_entry]
    return response


def relax_or_ask_route(state: AgentState) -> str:
    """Route the graph after the adaptive decision step."""

    if state.get("need_user_input"):
        return "wait_user"
    latest_action = (state.get("latest_decision") or {}).get("action")
    if latest_action == "relax_soft":
        return "retry"
    return "explain"
