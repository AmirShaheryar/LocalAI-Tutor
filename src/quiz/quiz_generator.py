import os
import sys
import json
import re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. RESOLVE ABSOLUTE PATHS (same pattern as your other modules)
current_script_path = os.path.abspath(__file__)
quiz_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(quiz_dir)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

output_quiz_dir = os.path.join(project_root, "outputs", "quizzes")
os.makedirs(output_quiz_dir, exist_ok=True)

import ollama
from src.retrieval.hybrid_retriever import hybrid_retrieve

OLLAMA_MODEL = "llama3.2:3b"

QUIZ_SYSTEM_PROMPT = """You are a quiz-generation engine for a study app. You create
practice questions STRICTLY from the study material given to you.

RULES YOU MUST FOLLOW:
1. Use ONLY the CONTEXT provided. Never invent facts outside it.
2. Output ONLY raw JSON. No markdown code fences, no explanation, no preamble,
   no text before or after the JSON object.
3. Multiple-choice distractors (wrong options) must be PLAUSIBLE and EDUCATIONAL --
   related to the topic and a common misconception, not random unrelated text.
   Example of a BAD distractor for "What does CPU stand for?": "Colorful Plastic Unit"
   Example of a GOOD distractor: "Central Programming Unit" (plausible mix-up)
4. Every question must include a "topic_id" -- a short lowercase snake_case tag
   summarizing the sub-topic it tests (e.g. "backpropagation", "gradient_descent").
5. Follow this EXACT JSON schema:

{
  "mcq": [
    {
      "topic_id": "string",
      "question": "string",
      "options": ["string", "string", "string", "string"],
      "correct_answer": "string (must exactly match one of the options)",
      "explanation": "string, one sentence on why the answer is correct"
    }
  ],
  "fill_in_blank": [
    {
      "topic_id": "string",
      "question": "string with a ____ blank",
      "correct_answer": "string"
    }
  ],
  "short_answer": [
    {
      "topic_id": "string",
      "question": "string",
      "model_answer": "string, 1-2 sentence ideal answer"
    }
  ]
}
"""

def _build_context_block(chunks):
    """Joins retrieved chunks into plain study material text for the quiz prompt.
    (Simpler than rag_engine's version -- quizzes don't need citation tags.)"""
    return "\n\n".join(chunk["text"] for chunk in chunks)


def _extract_json(raw_text):
    """Strips markdown fences or stray text the model might add despite instructions,
    then returns the parsed JSON dict. Raises ValueError if it still can't be parsed."""
    text = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    if not text.startswith("{"):
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            text = brace_match.group(0)

    return json.loads(text)  # raises json.JSONDecodeError if still broken

def _validate_quiz_schema(quiz):
    """Checks the parsed JSON actually has the shape we asked for.
    Returns (is_valid, error_message)."""
    required_keys = {"mcq", "fill_in_blank", "short_answer"}
    if not required_keys.issubset(quiz.keys()):
        missing = required_keys - quiz.keys()
        return False, f"Missing top-level keys: {missing}"

    for q in quiz.get("mcq", []):
        if not all(k in q for k in ("topic_id", "question", "options", "correct_answer")):
            return False, "An MCQ entry is missing required fields."
        if q["correct_answer"] not in q["options"]:
            return False, f"MCQ correct_answer '{q['correct_answer']}' not found in its own options list."

    for q in quiz.get("fill_in_blank", []):
        if not all(k in q for k in ("topic_id", "question", "correct_answer")):
            return False, "A fill-in-blank entry is missing required fields."

    for q in quiz.get("short_answer", []):
        if not all(k in q for k in ("topic_id", "question", "model_answer")):
            return False, "A short-answer entry is missing required fields."

    return True, None