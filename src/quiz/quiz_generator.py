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

def generate_quiz(topic_query, num_mcq=3, num_fib=2, num_short=2, max_retries=2):
    """
    Full Day 6 pipeline: retrieve relevant course content -> prompt the LLM for a
    structured JSON quiz -> validate -> retry once if broken.
    """
    print(f" Retrieving material for topic: {topic_query!r}")
    chunks = hybrid_retrieve(topic_query)

    if not chunks:
        print("No relevant material found -- cannot generate a quiz on this topic.")
        return None

    context_block = _build_context_block(chunks)

    user_prompt = f"""CONTEXT:
{context_block}

Generate a quiz from the CONTEXT above with exactly:
- {num_mcq} multiple-choice questions
- {num_fib} fill-in-the-blank questions
- {num_short} short-answer questions

Output ONLY the JSON object, nothing else."""

    last_error = None
    for attempt in range(1, max_retries + 2):
        print(f" Generating quiz (attempt {attempt})...")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = response["message"]["content"]

        try:
            quiz = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"JSON parse failed: {e}"
            print(f" ⚠️  {last_error}")
            continue

        is_valid, error_message = _validate_quiz_schema(quiz)
        if is_valid:
            print(" ✅ Valid quiz generated.")
            return quiz

        last_error = f"Schema validation failed: {error_message}"
        print(f" ⚠️  {last_error}")

    print(f" ❌ Giving up after {max_retries + 1} attempts. Last error: {last_error}")
    return None

def save_quiz(quiz, topic_query):
    """Saves a generated quiz to outputs/quizzes/<slug>_quiz.json."""
    slug = re.sub(r"[^a-z0-9]+", "_", topic_query.lower()).strip("_")
    out_path = os.path.join(output_quiz_dir, f"{slug}_quiz.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(quiz, f, indent=2, ensure_ascii=False)

    print(f" 💾 Saved to {out_path}")
    return out_path


def print_quiz(quiz):
    """Pretty-prints a quiz to the console for quick manual review."""
    print("\n===== MULTIPLE CHOICE =====")
    for i, q in enumerate(quiz.get("mcq", []), start=1):
        print(f"\n{i}. [{q['topic_id']}] {q['question']}")
        for opt in q["options"]:
            marker = "✔" if opt == q["correct_answer"] else " "
            print(f"   [{marker}] {opt}")
        print(f"   Explanation: {q.get('explanation', '')}")

    print("\n===== FILL IN THE BLANK =====")
    for i, q in enumerate(quiz.get("fill_in_blank", []), start=1):
        print(f"\n{i}. [{q['topic_id']}] {q['question']}")
        print(f"   Answer: {q['correct_answer']}")

    print("\n===== SHORT ANSWER =====")
    for i, q in enumerate(quiz.get("short_answer", []), start=1):
        print(f"\n{i}. [{q['topic_id']}] {q['question']}")
        print(f"   Model answer: {q['model_answer']}")


if __name__ == "__main__":
    print(" Local Quiz Generator -- Day 6\n")
    topic = input("Enter a topic/question to build a quiz around: ").strip()

    quiz = generate_quiz(topic)
    if quiz:
        print_quiz(quiz)
        save_quiz(quiz, topic)