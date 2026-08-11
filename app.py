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


with tab1:
    st.subheader("Ask a question about your course material")
 
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
 
    user_question = st.chat_input("Ask something...")
 
    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)
 
        with st.chat_message("assistant"):
            with st.spinner("Searching database and reranking context..."):
                chunks = hybrid_retrieve(user_question)
 
            if not chunks:
                answer = "I don't have enough information in the provided materials to answer that."
                st.markdown(answer)
            else:
                context_block = build_context_block(chunks)
                user_prompt = f"""CONTEXT:
{context_block}
 
QUESTION: {user_question}
 
Answer the question using only the CONTEXT above, citing every claim."""
 
                with st.spinner("Generating response..."):
                    response = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    answer = response["message"]["content"]
 
                st.markdown(answer)
 
                shown_images = set()
                for chunk in chunks:
                    meta = chunk["metadata"]
 
                    if meta["type"] == "pdf_text":
                        images = get_images_for_page(meta["source"], meta["page"])
                        for img_path in images:
                            if img_path not in shown_images and os.path.exists(img_path):
                                st.image(img_path, caption=f"Diagram from Page {meta['page']}", width=350)
                                shown_images.add(img_path)
 
                    elif meta["type"] == "video_transcript":
                        ts_label = format_timestamp(meta["start"])
                        btn_key = f"jump_{meta['source']}_{meta['start']}"
                        if st.button(f"Jump to {ts_label} in {meta['source']}", key=btn_key):
                            st.session_state["active_video"] = meta["source"]
                            st.session_state["video_start"] = meta["start"]
 
        st.session_state.chat_history.append(("assistant", answer))
 
    if st.session_state.get("active_video"):
        video_path = os.path.join(project_root, "data", "raw_media", st.session_state["active_video"])
        if os.path.exists(video_path):
            st.divider()
            st.video(video_path, start_time=int(st.session_state.get("video_start", 0)))


with tab2:
    st.subheader("Generate a practice quiz")
 
    topic_input = st.text_input("Topic to quiz yourself on:")
 
    if st.button("Generate Quiz"):
        with st.spinner("Reranking context..."):
            quiz = generate_quiz(topic_input)
        if quiz:
            st.session_state.current_quiz = quiz
            st.session_state.quiz_submitted = False
        else:
            st.error("Couldn't generate a quiz -- try a topic covered in your material.")
 
    quiz = st.session_state.current_quiz
    if quiz:
        st.divider()
        mcq_answers = []
        for i, q in enumerate(quiz.get("mcq", [])):
            st.markdown(f"**{i+1}. {q['question']}**")
            choice = st.radio("Choose one:", q["options"], key=f"mcq_{i}", label_visibility="collapsed")
            mcq_answers.append(choice)
 
        fib_answers = []
        for i, q in enumerate(quiz.get("fill_in_blank", [])):
            st.markdown(f"**{q['question']}**")
            ans = st.text_input("Your answer:", key=f"fib_{i}")
            fib_answers.append(ans)
 
        for i, q in enumerate(quiz.get("short_answer", [])):
            st.markdown(f"**{q['question']}**")
            st.text_area("Your answer (self-graded):", key=f"sa_{i}")
            with st.expander("Show model answer"):
                st.write(q["model_answer"])
 
        if st.button("Submit Quiz"):
            all_answers = mcq_answers + fib_answers
            grade_quiz(quiz, all_answers)
            st.session_state.quiz_submitted = True
 
        if st.session_state.quiz_submitted:
            st.success("Quiz submitted! Feedback below:")
            for i, q in enumerate(quiz.get("mcq", [])):
                is_correct = mcq_answers[i] == q["correct_answer"]
                icon = "CORRECT" if is_correct else "WRONG"
                st.markdown(f"**[{icon}] Q{i+1}:** {q['explanation']}")
 