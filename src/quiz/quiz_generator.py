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

import difflib
import ollama
from src.retrieval.hybrid_retriever import hybrid_retrieve

OLLAMA_MODEL = "llama3.2:3b"

QUIZ_SYSTEM_PROMPT = """You are a quiz-generation engine for a study app. You create
practice questions STRICTLY from the study material given to you.

RULES YOU MUST FOLLOW:

1. Use ONLY the CONTEXT provided. Never invent facts outside it.

2. Output ONLY raw JSON. No markdown code fences, no explanation, no preamble,
   no text before or after the JSON object.

3. SELF-CONTAINED QUESTIONS. The reader will NOT have the source table, figure,
   or paragraph in front of them -- only the question text. So:
   - Never write a question that only makes sense next to a table (e.g. don't ask
     "up to and including X, how many requests/second?" if that phrasing only
     resolves against a table row). Instead restate the needed fact directly:
     "According to the cost analysis, processing 20 requests/second saves the
     company how much per year?"
   - Do not use vague pointers like "the proposed change" or "this method" unless
     you name the specific thing in the same sentence.

4. NO DUPLICATE FACTS. Each question (across MCQ, fill-in-blank, and short-answer
   combined) must test a DIFFERENT fact or concept from the context. Do not ask
   the same underlying fact twice with different wording.

5. DISTRACTORS MUST MATCH THE ANSWER'S TYPE AND UNIT. If the correct answer is a
   dollar amount, all wrong options must also be dollar amounts. If it's a time
   duration, all wrong options must also be time durations. Never mix categories
   (e.g. do not put "$300,000" as a distractor for a question whose answer is a
   duration like "13 months"). Distractors should be plausible, educational
   near-misses -- values or ideas a student who misread the material might pick --
   not random unrelated facts.
   Example of a BAD distractor for "What does CPU stand for?": "Colorful Plastic Unit"
   Example of a GOOD distractor: "Central Programming Unit" (plausible mix-up)

6. Every MCQ needs exactly 4 options, all texts distinct, exactly one correct.
   Never use "All of the above" or "None of the above" as an option.

7. Every question must include a "topic_id" -- a short lowercase snake_case tag
   summarizing the sub-topic it tests (e.g. "backpropagation", "gradient_descent").
   Reuse the SAME topic_id across questions that genuinely share a sub-topic, so
   results can be grouped -- but never reuse it for two questions testing the
   identical fact (that's rule 4).

8. Tag every question with a "difficulty": "easy", "medium", or "hard", based on
   whether it tests direct recall vs. applying/connecting multiple facts from the
   context.

9. For short-answer questions, include a "key_points" list of 2-4 short phrases
   the answer should cover, so the student can self-grade against something
   concrete instead of just a vague model answer.

10. Follow this EXACT JSON schema:

{
  "mcq": [
    {
      "topic_id": "string",
      "difficulty": "easy | medium | hard",
      "question": "string, fully self-contained",
      "options": ["string", "string", "string", "string"],
      "correct_answer": "string (must exactly match one of the options)",
      "explanation": "string, one sentence on why the answer is correct"
    }
  ],
  "fill_in_blank": [
    {
      "topic_id": "string",
      "difficulty": "easy | medium | hard",
      "question": "string, fully self-contained, with a ____ blank",
      "correct_answer": "string"
    }
  ],
  "short_answer": [
    {
      "topic_id": "string",
      "difficulty": "easy | medium | hard",
      "question": "string, fully self-contained",
      "model_answer": "string, 1-2 sentence ideal answer",
      "key_points": ["string", "string"]
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

_CATEGORY_PATTERNS = [
    ("currency", re.compile(r"\$\s?\d[\d,]*(\.\d+)?")),
    ("percentage", re.compile(r"\d+(\.\d+)?\s?%")),
    ("time_duration", re.compile(
        r"\b\d+(\.\d+)?\s?(second|sec|minute|min|hour|hr|day|week|month|mo|year|yr)s?\b",
        re.IGNORECASE,
    )),
    ("rate", re.compile(r"\b\d+(\.\d+)?\s?/\s?(second|sec|minute|min|hour|hr)\b", re.IGNORECASE)),
    ("plain_number", re.compile(r"^\s*\d[\d,]*(\.\d+)?\s*$")),
]


def _categorize_value(text):
    """Rough category of an MCQ option's data type, used to catch distractors
    that mix units with the correct answer (e.g. a dollar amount as a wrong
    option for a question whose answer is a time duration). Returns None for
    options that aren't clearly numeric/quantitative -- those are left alone,
    since type-checking prose options isn't meaningful."""
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return None


def _normalize(text):
    return re.sub(r"\s+", " ", text.lower().strip())


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
        if len(q["options"]) != 4:
            return False, f"MCQ '{q['question'][:50]}...' must have exactly 4 options, got {len(q['options'])}."
        if len(set(_normalize(o) for o in q["options"])) != 4:
            return False, f"MCQ '{q['question'][:50]}...' has duplicate/near-duplicate options."
        if q["correct_answer"] not in q["options"]:
            return False, f"MCQ correct_answer '{q['correct_answer']}' not found in its own options list."
        for banned in ("all of the above", "none of the above"):
            if any(banned in _normalize(o) for o in q["options"]):
                return False, f"MCQ '{q['question'][:50]}...' uses a banned option ('{banned}')."

    for q in quiz.get("fill_in_blank", []):
        if not all(k in q for k in ("topic_id", "question", "correct_answer")):
            return False, "A fill-in-blank entry is missing required fields."
        if "___" not in q["question"] and "____" not in q["question"]:
            return False, f"Fill-in-blank '{q['question'][:50]}...' has no blank."

    for q in quiz.get("short_answer", []):
        if not all(k in q for k in ("topic_id", "question", "model_answer")):
            return False, "A short-answer entry is missing required fields."

    return True, None


def _validate_quiz_quality(quiz):
    """Semantic checks beyond raw schema shape -- catches the two failure modes
    that slip past _validate_quiz_schema: distractors of the wrong type/unit,
    and two questions that quietly test the same fact. Returns (is_valid, error_message)."""

    # 1. Distractor type/unit consistency for each MCQ.
    for q in quiz.get("mcq", []):
        categories = {opt: _categorize_value(opt) for opt in q["options"]}
        known = {c for c in categories.values() if c is not None}
        if len(known) > 1:
            return False, (
                f"MCQ '{q['question'][:60]}...' mixes distractor types {known} -- "
                f"all options must share the same unit/category as the correct answer."
            )

    # 2. Duplicate-fact detection across ALL questions (mcq + fill_in_blank + short_answer).
    all_questions = [
        (q["question"], "mcq") for q in quiz.get("mcq", [])
    ] + [
        (q["question"], "fill_in_blank") for q in quiz.get("fill_in_blank", [])
    ] + [
        (q["question"], "short_answer") for q in quiz.get("short_answer", [])
    ]
    normalized = [_normalize(q) for q, _ in all_questions]
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            similarity = difflib.SequenceMatcher(None, normalized[i], normalized[j]).ratio()
            if similarity > 0.55:
                return False, (
                    f"Question {i+1} ({all_questions[i][1]}) and question {j+1} "
                    f"({all_questions[j][1]}) appear to test the same fact: "
                    f"'{all_questions[i][0][:50]}...' vs '{all_questions[j][0][:50]}...'"
                )

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

    messages = [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(1, max_retries + 2):
        print(f" Generating quiz (attempt {attempt})...")
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        raw_text = response["message"]["content"]

        try:
            quiz = _extract_json(raw_text)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"JSON parse failed: {e}"
            print(f" ⚠️  {last_error}")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": (
                f"That was not valid JSON ({e}). Re-output the FULL quiz as raw JSON "
                f"only, matching the schema exactly. No markdown fences, no commentary."
            )})
            continue

        is_valid, error_message = _validate_quiz_schema(quiz)
        if not is_valid:
            last_error = f"Schema validation failed: {error_message}"
            print(f" ⚠️  {last_error}")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": (
                f"Fix this specific problem and re-output the FULL corrected quiz as "
                f"raw JSON only: {error_message}"
            )})
            continue

        is_quality, quality_error = _validate_quiz_quality(quiz)
        if not is_quality:
            last_error = f"Quality check failed: {quality_error}"
            print(f" ⚠️  {last_error}")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": (
                f"Fix this specific problem and re-output the FULL corrected quiz as "
                f"raw JSON only: {quality_error}"
            )})
            continue

        print(" ✅ Valid, high-quality quiz generated.")
        return quiz

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
        print(f"\n{i}. [{q['topic_id']} | {q.get('difficulty', '?')}] {q['question']}")
        for opt in q["options"]:
            marker = "✔" if opt == q["correct_answer"] else " "
            print(f"   [{marker}] {opt}")
        print(f"   Explanation: {q.get('explanation', '')}")

    print("\n===== FILL IN THE BLANK =====")
    for i, q in enumerate(quiz.get("fill_in_blank", []), start=1):
        print(f"\n{i}. [{q['topic_id']} | {q.get('difficulty', '?')}] {q['question']}")
        print(f"   Answer: {q['correct_answer']}")

    print("\n===== SHORT ANSWER =====")
    for i, q in enumerate(quiz.get("short_answer", []), start=1):
        print(f"\n{i}. [{q['topic_id']} | {q.get('difficulty', '?')}] {q['question']}")
        print(f"   Model answer: {q['model_answer']}")
        if q.get("key_points"):
            print(f"   Key points to self-grade against: {', '.join(q['key_points'])}")


if __name__ == "__main__":
    print(" Local Quiz Generator -- Day 6\n")
    topic = input("Enter a topic/question to build a quiz around: ").strip()

    quiz = generate_quiz(topic)
    if quiz:
        print_quiz(quiz)
        save_quiz(quiz, topic)