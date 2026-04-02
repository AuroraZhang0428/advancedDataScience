from typing import Dict, Any


def init_preference_state() -> Dict[str, Any]:
    return {
        "hard_constraints": {
            "max_price": None,
            "location": None,
            "guests": None,
            "room_type": None,
        },
        "soft_preferences": {
            "amenities": [],
            "vibe": None,
            "purpose": None,
            "preferred_area": None,
        }
    }


def update_preferences_from_input(state: Dict[str, Any], user_input: Dict[str, Any]) -> Dict[str, Any]:
    for section in ["hard_constraints", "soft_preferences"]:
        if section in user_input:
            for key, value in user_input[section].items():
                state[section][key] = value
    return state


def check_missing_required(state: Dict[str, Any]):
    required = ["max_price", "location", "guests"]
    missing = [k for k in required if not state["hard_constraints"].get(k)]
    return missing


def generate_followup_question(missing_fields):
    if not missing_fields:
        return None

    field = missing_fields[0]

    questions = {
        "max_price": "What is your maximum budget per night?",
        "location": "Which city or area are you staying in?",
        "guests": "How many guests will be staying?",
    }

    return questions.get(field, "Can you provide more details?")