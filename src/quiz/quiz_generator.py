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

3. SELF-CONTAINED QUESTIONS — CRITICAL. The student sees ONLY the question
   text, with NO access to the PDF, tables, or video. Every question must
   stand alone and be answerable from memory after studying the material.

   REQUIRED: Begin every question by anchoring it to the scenario. Use a
   phrase like "In the [document/scenario name] cost-benefit analysis, ..."
   or "According to the [specific module/project name] analysis in [source], ..."
   Name concrete nouns from the context: company/project name, module names
   (e.g. "sorted-index module"), specific changes being evaluated, and units.

   NEVER use bare vague references the student cannot resolve:
   - BAD: "What is the value function for every additional request per second?"
     (Which system? Which analysis? Which document?)
   - GOOD: "In the search-engine cost-benefit analysis, what is the value
     function (in dollars per year) for each additional request per second
     the system can handle?"
   - BAD: "The company saves $____ per year by adding a second server."
     (Which company? Which scenario?)
   - GOOD: "In the cost-benefit analysis for scaling the search system,
     how much does the company save per year by adding a second server?"
   - BAD: "The payback period is calculated as total cost divided by
     annual savings of $____ per year." (Savings for WHICH change?)
   - GOOD: "For restructuring the sorted-index module (total cost $50,000),
     the payback period equals total cost divided by annual savings of
     $____ per year — what is that annual savings figure?"

   Do not use "the proposed change", "this method", "the system", or
   "the company" unless the same sentence already names the specific
   project, document, or module they refer to.

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
    """Join retrieved chunks into labeled study material for the quiz prompt.
    Each chunk is tagged with source file and page/timestamp so the LLM can
    anchor questions to concrete scenario details."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown source")
        if meta.get("type") == "pdf_text":
            location = f"page {meta.get('page', '?')}"
        elif meta.get("type") == "video_transcript":
            secs = int(meta.get("start", 0))
            location = f"{secs // 60:02d}:{secs % 60:02d}"
        else:
            location = "unknown location"
        lines.append(f"[Excerpt {i} — {source}, {location}]\n{chunk['text']}")
    return "\n\n".join(lines)


def _derive_scope_label(topic_query, source_filter=None):
    """Human-readable label the LLM must echo in every question stem."""
    if source_filter:
        doc_name = re.sub(r"\.(pdf|txt|md)$", "", source_filter, flags=re.IGNORECASE)
        doc_name = doc_name.replace("_", " ").replace("-", " ")
        return f"{doc_name} ({topic_query})"
    return topic_query


_CLARITY_ANCHORS = re.compile(
    r"\b(according to|in the|based on|for the|from the|within the|"
    r"cost[- ]benefit|analysis of|when evaluating|for restructuring|"
    r"for scaling|search[- ]engine|sorted[- ]index|noise words|"
    r"second server|requests/second|requests per second)\b",
    re.IGNORECASE,
)
_VAGUE_ONLY = re.compile(
    r"\b(the company|the system|the model|this method|the proposed|"
    r"the value function|the payback period)\b",
    re.IGNORECASE,
)


def _validate_question_clarity(question, scope_label):
    """Reject questions that rely on unstated context the student can't see."""
    q = question.strip()
    if len(q) < 40:
        return False, f"Question too short to be self-contained: '{q[:50]}...'"

    has_anchor = bool(_CLARITY_ANCHORS.search(q))
    has_vague = bool(_VAGUE_ONLY.search(q))

    scope_words = [w.lower() for w in re.findall(r"\w+", scope_label) if len(w) > 3]
    mentions_scope = any(w in q.lower() for w in scope_words)

    if has_vague and not (has_anchor or mentions_scope):
        return False, (
            f"Question uses vague references without naming the scenario: "
            f"'{q[:70]}...' — rewrite to start with the document/scenario "
            f"({scope_label}) and name the specific module or change."
        )

    if q.lower().startswith(("what is the ", "how much does the ", "the ")) and not (has_anchor or mentions_scope):
        return False, (
            f"Question opens vaguely: '{q[:70]}...' — begin with "
            f"'In the {scope_label}...' or 'According to...' and name "
            "the specific system, module, or change being tested."
        )

    return True, None


def _repair_json(text):
    """Fix common JSON syntax mistakes from small LLM outputs."""
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    text = re.sub(r"\}(\s*)\{", r"},\1{", text)
    text = re.sub(r"\](\s*)\[", r"],\1[", text)
    text = re.sub(r'"\s*\n\s*"', '",\n"', text)
    return text


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

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_json(text))


def _normalize_quiz(quiz):
    """Fill optional fields the schema asks for but validation doesn't require."""
    for q in quiz.get("mcq", []):
        q.setdefault("difficulty", "medium")
        q.setdefault("explanation", "")
    for q in quiz.get("fill_in_blank", []):
        q.setdefault("difficulty", "medium")
    for q in quiz.get("short_answer", []):
        q.setdefault("difficulty", "medium")
        q.setdefault("key_points", [])
    return quiz


def _call_ollama(messages, json_mode=True):
    """Single Ollama chat call with low temperature and optional JSON mode."""
    kwargs = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "options": {"temperature": 0.1, "num_predict": 4096},
    }
    if json_mode:
        kwargs["format"] = "json"
    response = ollama.chat(**kwargs)
    return response["message"]["content"]


def _save_debug_response(raw_text, attempt, error):
    """Persist the last failed LLM response so parse errors can be inspected."""
    debug_path = os.path.join(output_quiz_dir, "_debug_last_failure.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(f"Attempt: {attempt}\nError: {error}\n\n--- RAW RESPONSE ---\n{raw_text}")
    print(f" 📝 Saved failed response to {debug_path}")

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
        if not re.search(r"_{2,}", q["question"]):
            return False, f"Fill-in-blank '{q['question'][:50]}...' has no blank."

    for q in quiz.get("short_answer", []):
        if not all(k in q for k in ("topic_id", "question", "model_answer")):
            return False, "A short-answer entry is missing required fields."

    return True, None


def _validate_quiz_quality(quiz, scope_label=""):
    """Semantic checks beyond raw schema shape -- catches distractor type mismatches,
    duplicate facts, and vague question wording. Returns (is_valid, error_message)."""

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
            if similarity > 0.75:
                return False, (
                    f"Question {i+1} ({all_questions[i][1]}) and question {j+1} "
                    f"({all_questions[j][1]}) appear to test the same fact: "
                    f"'{all_questions[i][0][:50]}...' vs '{all_questions[j][0][:50]}...'"
                )

    # 3. Self-contained wording — every question must anchor to the scenario.
    if scope_label:
        for q in quiz.get("mcq", []):
            ok, err = _validate_question_clarity(q["question"], scope_label)
            if not ok:
                return False, err
        for q in quiz.get("fill_in_blank", []):
            ok, err = _validate_question_clarity(q["question"], scope_label)
            if not ok:
                return False, err
        for q in quiz.get("short_answer", []):
            ok, err = _validate_question_clarity(q["question"], scope_label)
            if not ok:
                return False, err

    return True, None


_SECTION_SCHEMAS = {
    "mcq": {
        "prompt": (
            'Generate ONLY a JSON object with a single "mcq" array containing '
            "exactly {count} multiple-choice questions. Follow the MCQ schema "
            "from the system prompt. No other top-level keys."
        ),
        "key": "mcq",
    },
    "fill_in_blank": {
        "prompt": (
            'Generate ONLY a JSON object with a single "fill_in_blank" array '
            "containing exactly {count} fill-in-the-blank questions. Each "
            "question must include a blank written as ____. No other top-level keys."
        ),
        "key": "fill_in_blank",
    },
    "short_answer": {
        "prompt": (
            'Generate ONLY a JSON object with a single "short_answer" array '
            "containing exactly {count} short-answer questions. Include "
            '"key_points" for each. No other top-level keys.'
        ),
        "key": "short_answer",
    },
}


def _clarity_instructions(scope_label):
    return (
        f"QUIZ SCOPE: {scope_label}\n\n"
        "CLARITY REQUIREMENT: Every question MUST begin by naming this scope "
        f"({scope_label}). Include the specific module, change, or scenario "
        "from the context (e.g. 'sorted-index module', 'eliminating noise words', "
        "'adding a second server'). A student with no PDF open must know exactly "
        "what is being asked."
    )


def _generate_quiz_sections(context_block, num_mcq, num_fib, num_short, topic_query, scope_label):
    """Fallback: generate each question type in a separate LLM call for reliability."""
    counts = {"mcq": num_mcq, "fill_in_blank": num_fib, "short_answer": num_short}
    quiz = {"mcq": [], "fill_in_blank": [], "short_answer": []}

    for section_name, spec in _SECTION_SCHEMAS.items():
        count = counts[section_name]
        if count <= 0:
            continue

        section_prompt = f"""CONTEXT:
{context_block}

{_clarity_instructions(scope_label)}

{spec["prompt"].format(count=count)}

Output ONLY the JSON object, nothing else."""

        messages = [
            {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
            {"role": "user", "content": section_prompt},
        ]

        for attempt in range(1, 4):
            print(f" Generating {section_name} (attempt {attempt})...")
            raw_text = _call_ollama(messages)
            try:
                parsed = _extract_json(raw_text)
            except (json.JSONDecodeError, ValueError) as e:
                print(f" ⚠️  {section_name} JSON parse failed: {e}")
                _save_debug_response(raw_text, attempt, e)
                messages = [
                    {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        section_prompt
                        + f"\n\nYour previous output was invalid JSON ({e}). "
                        "Re-output ONLY valid raw JSON matching the schema."
                    )},
                ]
                continue

            items = parsed.get(spec["key"], [])
            if not isinstance(items, list) or len(items) != count:
                got = len(items) if isinstance(items, list) else "non-list"
                print(f" ⚠️  {section_name}: expected {count} items, got {got}")
                messages = [
                    {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        section_prompt
                        + f"\n\nYou returned {got} items but need exactly {count}. "
                        "Re-output ONLY valid raw JSON."
                    )},
                ]
                continue

            clarity_failed = False
            for item in items:
                ok, clarity_err = _validate_question_clarity(item.get("question", ""), scope_label)
                if not ok:
                    print(f" ⚠️  {section_name} clarity check failed: {clarity_err}")
                    messages = [
                        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            section_prompt
                            + f"\n\nFix this clarity problem and re-output ONLY valid raw JSON: "
                            f"{clarity_err}"
                        )},
                    ]
                    clarity_failed = True
                    break
            if clarity_failed:
                continue

            quiz[spec["key"]] = items
            break
        else:
            print(f" ❌ Failed to generate {section_name} section after 3 attempts.")
            return None

    return _normalize_quiz(quiz)


def generate_quiz(topic_query, num_mcq=3, num_fib=2, num_short=2, max_retries=2, source_filter=None):
    """
    Full Day 6 pipeline: retrieve relevant course content -> prompt the LLM for a
    structured JSON quiz -> validate (schema + quality) -> retry with targeted
    feedback if broken.

    source_filter: optional filename (from get_indexed_sources()) to restrict
    retrieval to a single uploaded document, so quiz questions don't mix
    content across unrelated files.
    """
    print(f" Retrieving material for topic: {topic_query!r}"
          + (f" (scoped to {source_filter})" if source_filter else ""))
    chunks = hybrid_retrieve(topic_query, source_filter=source_filter)

    if not chunks:
        print("No relevant material found -- cannot generate a quiz on this topic.")
        return None

    context_block = _build_context_block(chunks)
    scope_label = _derive_scope_label(topic_query, source_filter)

    user_prompt = f"""CONTEXT:
{context_block}

{_clarity_instructions(scope_label)}

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
        raw_text = _call_ollama(messages)

        try:
            quiz = _normalize_quiz(_extract_json(raw_text))
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"JSON parse failed: {e}"
            print(f" ⚠️  {last_error}")
            _save_debug_response(raw_text, attempt, e)
            messages = [
                {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    user_prompt
                    + f"\n\nYour previous output was invalid JSON ({e}). "
                    "Re-output the FULL quiz as valid raw JSON only. "
                    "Ensure commas between every array element and escape "
                    "any double quotes inside string values."
                )},
            ]
            continue

        is_valid, error_message = _validate_quiz_schema(quiz)
        if not is_valid:
            last_error = f"Schema validation failed: {error_message}"
            print(f" ⚠️  {last_error}")
            messages = [
                {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    user_prompt
                    + f"\n\nFix this specific problem and re-output the FULL quiz "
                    f"as raw JSON only: {error_message}"
                )},
            ]
            continue

        is_quality, quality_error = _validate_quiz_quality(quiz, scope_label)
        if not is_quality:
            last_error = f"Quality check failed: {quality_error}"
            print(f" ⚠️  {last_error}")
            messages = [
                {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    user_prompt
                    + f"\n\nFix this specific problem and re-output the FULL quiz "
                    f"as raw JSON only: {quality_error}"
                )},
            ]
            continue

        print(" ✅ Valid, high-quality quiz generated.")
        return quiz

    print(f" ❌ Full-quiz generation failed ({last_error}). Trying section-by-section...")
    quiz = _generate_quiz_sections(
        context_block, num_mcq, num_fib, num_short, topic_query, scope_label
    )
    if not quiz:
        print(f" ❌ Giving up. Last error: {last_error}")
        return None

    is_valid, error_message = _validate_quiz_schema(quiz)
    if not is_valid:
        print(f" ❌ Section fallback failed schema check: {error_message}")
        return None

    is_quality, quality_error = _validate_quiz_quality(quiz, scope_label)
    if not is_quality:
        if quality_error and ("vague" in quality_error.lower() or "opens vaguely" in quality_error.lower()):
            print(f" ❌ Section fallback failed clarity check: {quality_error}")
            return None
        print(f" ⚠️  Section fallback passed schema but failed quality: {quality_error}")
        print(" ✅ Returning quiz anyway (non-clarity quality issue is non-fatal).")

    print(" ✅ Quiz generated via section-by-section fallback.")
    return quiz

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