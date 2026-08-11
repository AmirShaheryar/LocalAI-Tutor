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


 
def get_images_for_page(source, page):
    """Looks up any diagrams indexed from this exact PDF page (Day 3's CLIP index)."""
    try:
        results = image_collection.get(
            where={"$and": [{"source": source}, {"page": page}]}
        )
        return [meta["image_path"] for meta in results.get("metadatas", [])]
    except Exception:
        return []
 
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
 
st.title("LocalAI-Tutor")
tab1, tab2, tab3 = st.tabs(["Study Workspace", "Practice & Quiz Center", "Analytics"])
