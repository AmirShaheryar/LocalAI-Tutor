import os
import sys
import json

import streamlit as st
import chromadb
import ollama
import fitz  # PyMuPDF

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.synthesis.rag_engine import build_context_block, format_timestamp, SYSTEM_PROMPT, OLLAMA_MODEL
from src.retrieval.hybrid_retriever import hybrid_retrieve
import src.retrieval.hybrid_retriever as hybrid_retriever_module
from src.quiz.quiz_generator import generate_quiz
from src.tracing.knowledge_tracer import grade_quiz, get_all_mastery, recommend_review
from src.parsers.pdf_parser import parse_pdf
from src.parsers.media_parser import transcribe_lecture
from src.database.vector_store import process_and_index_pdf, process_and_index_transcript
from src.database.clip_indexer import process_and_index_images

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


def render_pdf_page_image(source, page_num):
    """
    Rasterizes one PDF page to a PNG on demand -- this lets you SEE the actual
    cited page (layout, figures, everything), not just the extracted text.
    Caches rendered pages to data/rendered_pages/ so repeat views are instant.
    """
    pdf_path = os.path.join(project_root, "data", "raw_pdfs", source)
    if not os.path.exists(pdf_path):
        return None

    cache_dir = os.path.join(project_root, "data", "rendered_pages")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{source}_p{page_num}.png")

    if os.path.exists(cache_path):
        return cache_path

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # PyMuPDF pages are 0-indexed, your citations are 1-indexed
        pix = page.get_pixmap(dpi=150)
        pix.save(cache_path)
        doc.close()
        return cache_path
    except Exception:
        return None


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

st.title("LocalAI-Tutor")
tab1, tab2, tab3 = st.tabs(["Study Workspace", "Practice & Quiz Center", "Analytics"])

# ---------------- TAB 1: STUDY WORKSPACE ----------------
with tab1:
    st.subheader("Upload study material")

    uploaded_file = st.file_uploader(
        "Upload a PDF or lecture video", type=["pdf", "mp4", "mkv", "mov"]
    )

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if st.button("Process & Index this file"):
            if file_ext == "pdf":
                save_dir = os.path.join(project_root, "data", "raw_pdfs")
            else:
                save_dir = os.path.join(project_root, "data", "raw_media")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if file_ext == "pdf":
                with st.spinner("Parsing PDF (text, math, images)..."):
                    structured = parse_pdf(save_path)
                    day1_json_path = os.path.join(project_root, "outputs", "day1_structured_output.json")
                    os.makedirs(os.path.dirname(day1_json_path), exist_ok=True)
                    with open(day1_json_path, "w", encoding="utf-8") as jf:
                        json.dump(structured, jf, indent=2, ensure_ascii=False)

                with st.spinner("Indexing text into ChromaDB..."):
                    process_and_index_pdf(save_path)

                with st.spinner("Indexing diagrams (CLIP)..."):
                    process_and_index_images(day1_json_path)

            else:
                with st.spinner("Transcribing video -- this can take a few minutes..."):
                    transcribe_lecture(save_path, output_dir=os.path.join(project_root, "outputs"))

                with st.spinner("Indexing transcript into ChromaDB..."):
                    transcript_json_path = os.path.join(
                        project_root, "outputs",
                        os.path.splitext(uploaded_file.name)[0] + "_transcript.json"
                    )
                    process_and_index_transcript(transcript_json_path)

            # the BM25 keyword index is built ONCE and cached in memory the first
            # time hybrid_retrieve() runs -- force it to rebuild so newly uploaded
            # content is actually searchable, not just sitting in ChromaDB unseen
            hybrid_retriever_module._bm25_index = None

            st.success(f"'{uploaded_file.name}' processed and indexed! Ask about it below.")

    st.divider()
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
                shown_pdf_pages = set()
                for i, chunk in enumerate(chunks):
                    meta = chunk["metadata"]

                    if meta["type"] == "pdf_text":
                        # any embedded diagrams on this page (Day 3 CLIP index)
                        images = get_images_for_page(meta["source"], meta["page"])
                        for img_path in images:
                            if img_path not in shown_images and os.path.exists(img_path):
                                st.image(img_path, caption=f"Diagram from Page {meta['page']}", width=350)
                                shown_images.add(img_path)

                        # button to view the actual PDF page, regardless of whether
                        # it has embedded diagrams -- this is the "PDF opener"
                        page_key_tuple = (meta["source"], meta["page"])
                        if page_key_tuple not in shown_pdf_pages:
                            shown_pdf_pages.add(page_key_tuple)
                            page_btn_key = f"viewpdf_{meta['source']}_{meta['page']}_{i}"
                            if st.button(f"View PDF Page {meta['page']} ({meta['source']})", key=page_btn_key):
                                st.session_state["active_pdf_page"] = (meta["source"], meta["page"])

                    elif meta["type"] == "video_transcript":
                        ts_label = format_timestamp(meta["start"])
                        btn_key = f"jump_{meta['source']}_{meta['start']}_{i}"
                        if st.button(f"Jump to {ts_label} in {meta['source']}", key=btn_key):
                            st.session_state["active_video"] = meta["source"]
                            st.session_state["video_start"] = meta["start"]

        st.session_state.chat_history.append(("assistant", answer))

    if st.session_state.get("active_video"):
        video_path = os.path.join(project_root, "data", "raw_media", st.session_state["active_video"])
        if os.path.exists(video_path):
            st.divider()
            st.video(video_path, start_time=int(st.session_state.get("video_start", 0)))

    if st.session_state.get("active_pdf_page"):
        source, page_num = st.session_state["active_pdf_page"]
        page_img_path = render_pdf_page_image(source, page_num)
        if page_img_path:
            st.divider()
            st.image(page_img_path, caption=f"{source} -- Page {page_num}")
        else:
            st.error(f"Couldn't render page {page_num} from {source}.")

# ---------------- TAB 2: PRACTICE & QUIZ CENTER ----------------
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

# ---------------- TAB 3: ANALYTICS ----------------
with tab3:
    st.subheader("Topic Mastery")

    mastery_data = get_all_mastery()

    if not mastery_data:
        st.info("No quiz attempts yet -- take a quiz in the Practice tab first.")
    else:
        st.bar_chart(mastery_data)

        st.divider()
        st.subheader("Recommended Review")
        with st.spinner("Searching database..."):
            recs = recommend_review()

        if not recs:
            st.success("You're above threshold on every tracked topic!")
        else:
            for rec in recs:
                st.warning(f"**{rec['topic_id']}** (mastery: {rec['mastery']:.2f}) -- {rec['recommendation']}")