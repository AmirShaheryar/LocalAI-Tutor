import os
import sys
import json
from datetime import datetime, timezone

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. RESOLVE ABSOLUTE PATHS (same pattern as your other modules)
current_script_path = os.path.abspath(__file__)
tracing_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(tracing_dir)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

state_path = os.path.join(project_root, "data", "knowledge_state.json")

# 2. BKT PARAMETERS
# These are standard "reasonable defaults" used across BKT literature/tutoring systems.
# p_init:    probability a topic is already known before any attempt
# p_transit: probability of learning it after ONE attempt (even a wrong one --
#            you can learn from getting something wrong)
# p_slip:    probability of a KNOWN topic being answered wrong by mistake (silly error)
# p_guess:   probability of an UNKNOWN topic being answered right by luck (e.g. MCQ guess)
BKT_PARAMS = {
    "p_init": 0.3,
    "p_transit": 0.15,
    "p_slip": 0.1,
    "p_guess": 0.2,
}


def _load_state():
    """Loads the full knowledge state (all topics) from disk, or starts fresh."""
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def _bkt_update(prior_mastery, is_correct):
    """
    Standard 2-step BKT update:
    Step A: revise belief about CURRENT mastery based on the observed answer (Bayes' rule)
    Step B: account for the chance they LEARNED something from this attempt
    """
    p_slip = BKT_PARAMS["p_slip"]
    p_guess = BKT_PARAMS["p_guess"]
    p_transit = BKT_PARAMS["p_transit"]

    if is_correct:
        # P(known | answered correctly)
        numerator = prior_mastery * (1 - p_slip)
        denominator = numerator + (1 - prior_mastery) * p_guess
    else:
        # P(known | answered incorrectly)
        numerator = prior_mastery * p_slip
        denominator = numerator + (1 - prior_mastery) * (1 - p_guess)

    posterior = numerator / denominator if denominator > 0 else prior_mastery

    # Step B: apply learning transition -- even if they didn't know it,
    # attempting the question gives some chance they now do
    updated_mastery = posterior + (1 - posterior) * p_transit

    return round(updated_mastery, 4)


def record_attempt(topic_id, is_correct, time_taken=None):
    """
    Call this every time a student answers a quiz question.
    Updates and persists that topic's mastery score using BKT.
    """
    state = _load_state()

    topic_state = state.get(topic_id, {
        "mastery": BKT_PARAMS["p_init"],
        "attempts": 0,
        "correct": 0,
        "history": [],
    })

    new_mastery = _bkt_update(topic_state["mastery"], is_correct)

    topic_state["mastery"] = new_mastery
    topic_state["attempts"] += 1
    topic_state["correct"] += 1 if is_correct else 0
    topic_state["history"].append({
        "correct": is_correct,
        "time_taken": time_taken,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    state[topic_id] = topic_state
    _save_state(state)

    print(f" [{topic_id}] mastery updated: {new_mastery:.2f} "
          f"({'✔ correct' if is_correct else '✘ incorrect'}, attempt #{topic_state['attempts']})")

    return new_mastery


def get_mastery(topic_id):
    """Returns current mastery for a topic, or the default p_init if never attempted."""
    state = _load_state()
    return state.get(topic_id, {}).get("mastery", BKT_PARAMS["p_init"])


def get_all_mastery():
    """Returns {topic_id: mastery} for every topic attempted so far -- feeds Day 8's dashboard."""
    state = _load_state()
    return {topic: data["mastery"] for topic, data in state.items()}