import os
import sys
import json
 
import streamlit as st
import chromadb
import ollama
 
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
 
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
 
from src.synthesis.rag_engine import build_context_block, format_timestamp, SYSTEM_PROMPT, OLLAMA_MODEL
from src.retrieval.hybrid_retriever import hybrid_retrieve
from src.quiz.quiz_generator import generate_quiz
from src.tracing.knowledge_tracer import grade_quiz, get_all_mastery, recommend_review
 
st.set_page_config(page_title="LocalAI-Tutor", layout="wide")
 
image_client = chromadb.PersistentClient(path=os.path.join(project_root, "data", "chroma_db"))
image_collection = image_client.get_or_create_collection(name="course_images")
