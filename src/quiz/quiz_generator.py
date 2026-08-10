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