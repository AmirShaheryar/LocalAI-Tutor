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