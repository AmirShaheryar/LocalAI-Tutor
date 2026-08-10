import os
import sys
import json

import streamlit as st

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# app.py lives at project root, but keep this for safety if ever run from elsewhere
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.synthesis.rag_engine import generate_answer
from src.quiz.quiz_generator import generate_quiz
from src.tracing.knowledge_tracer import grade_quiz, get_all_mastery, recommend_review

st.set_page_config(page_title="LocalAI-Tutor", layout="wide")

# Session state -- survives reruns WITHIN one browser session, resets on refresh.
# (Remember: no localStorage/browser storage in this environment -- this is
# Streamlit's own in-memory session mechanism, totally separate concern.)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

st.title("📚 LocalAI-Tutor")
tab1, tab2, tab3 = st.tabs(["💬 Study Workspace", "📝 Practice & Quiz Center", "📊 Analytics"])

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
            with st.spinner("Searching database..."):
                answer = generate_answer(user_question, stream_output=False)
            st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))

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
                icon = "✅" if is_correct else "❌"
                st.markdown(f"{icon} **Q{i+1}:** {q['explanation']}")