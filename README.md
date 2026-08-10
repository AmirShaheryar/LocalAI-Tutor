# LocalAI-Tutor

A fully **local, privacy-first AI tutoring system** that turns your own course materials (textbook PDFs and lecture recordings) into an interactive study assistant — grounded RAG Q&A with exact citations, auto-generated practice quizzes, and adaptive mastery tracking. Everything runs on-device via [Ollama](https://ollama.com); no data ever leaves your machine.

> Built as a portfolio / Final Year Project demonstrating a full local ML pipeline: ingestion → retrieval → generation → assessment → learning analytics.

---

## ✨ What it does

1. **Ingests** your PDFs and lecture videos, extracting structured text, math notation, embedded diagrams, and timestamped transcripts.
2. **Answers questions** grounded strictly in your material, using hybrid search (semantic + keyword) and cross-encoder reranking, with exact citations like `[PDF Page 14]` or `[Video @ 03:15]`.
3. **Generates practice quizzes** (MCQ, fill-in-the-blank, short answer) directly from your content.
4. **Tracks what you actually know** — mastery scores per topic, with recommendations on exactly what to review next. *(in progress)*
5. **Wraps it all in a dashboard** — chat, quizzes, and analytics in one local web app. *(in progress)*

---

## 🏗️ Architecture

```
                    ┌─────────────┐      ┌──────────────┐
   PDF/Video  ───▶  │   Parsers   │ ───▶ │  ChromaDB /   │
                    │ (Day 1 & 2) │      │  CLIP vectors │
                    └─────────────┘      │  (Day 3)      │
                                          └──────┬────────┘
                                                 │
                                   ┌─────────────▼─────────────┐
                                   │   Hybrid Retriever         │
                                   │ Vector + BM25 → RRF fusion │
                                   │ → Cross-Encoder rerank     │
                                   │        (Day 4)             │
                                   └─────────────┬──────────────┘
                                                 │
                       ┌─────────────────────────┼─────────────────────────┐
                       ▼                                                   ▼
             ┌──────────────────┐                                ┌──────────────────┐
             │   RAG Engine      │                                │  Quiz Generator   │
             │ Ollama + citation │                                │  Structured JSON  │
             │  prompt (Day 5)   │                                │  quizzes (Day 6)  │
             └──────────────────┘                                └──────────────────┘
```

All models run locally through **Ollama** (LLM) and **Hugging Face** (embeddings/reranking) — no external API calls, no cloud dependency.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| PDF parsing | PyMuPDF (fitz) |
| Audio/video transcription | faster-whisper |
| Vector database | ChromaDB (persistent) |
| Text embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Image embeddings | `openai/clip-vit-base-patch32` |
| Keyword search | `rank_bm25` |
| Reranking | `BAAI/bge-reranker-base` (cross-encoder) |
| LLM | Ollama (`llama3.2:3b`) |
| Chunking | `langchain-text-splitters` |
| UI *(planned)* | Streamlit |
| Knowledge tracing *(planned)* | scikit-learn / Bayesian Knowledge Tracing |

---

## 📁 Project Structure

```
LocalAI-Tutor/
├── data/
│   ├── raw_pdfs/            # source PDFs
│   ├── raw_media/           # source lecture videos
│   ├── extracted_images/    # images pulled from PDFs
│   └── chroma_db/           # persistent vector store
├── outputs/
│   ├── day1_structured_output.json
│   ├── Lecture_transcript.json
│   └── quizzes/             # generated quiz JSON files
├── src/
│   ├── parsers/
│   │   ├── pdf_parser.py        # Day 1 — PDF → structured JSON
│   │   └── media_parser.py      # Day 2 — video/audio → timestamped transcript
│   ├── database/
│   │   ├── vector_store.py      # Day 3 — index text into ChromaDB (MiniLM)
│   │   └── clip_indexer.py      # Day 3 — index images into ChromaDB (CLIP)
│   ├── retrieval/
│   │   └── hybrid_retriever.py  # Day 4 — vector + BM25 + reranking
│   ├── synthesis/
│   │   └── rag_engine.py        # Day 5 — Ollama + citation-grounded answers
│   └── quiz/
│       └── quiz_generator.py    # Day 6 — structured JSON quiz generation
└── requirements.txt
```

---

## 🚀 Setup

### 1. Install Ollama and pull a model
```bash
# https://ollama.com
ollama pull llama3.2:3b
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

> **Note:** first run will download embedding/reranking models from Hugging Face
> (MiniLM ~90MB, CLIP ~600MB, bge-reranker ~1.1GB). After the first run, everything
> is cached locally and works fully offline.

### 3. Add your study material
Drop files into:
```
data/raw_pdfs/your_textbook.pdf
data/raw_media/your_lecture.mp4
```

### 4. Run the pipeline, in order
```bash
python src/parsers/pdf_parser.py        # Day 1: parse PDF
python src/parsers/media_parser.py      # Day 2: transcribe lecture
python src/database/vector_store.py     # Day 3: index text
python src/database/clip_indexer.py     # Day 3: index images
```

### 5. Ask questions
```bash
python src/synthesis/rag_engine.py
```

### 6. Generate a quiz
```bash
python src/quiz/quiz_generator.py
```

---

## 📊 Project Status

| Day | Feature | Status |
|---|---|---|
| 1 | PDF parsing (text, math, images) | ✅ Done |
| 2 | Video/audio transcription | ✅ Done |
| 3 | Vector storage (text + image embeddings) | ✅ Done |
| 4 | Hybrid search + cross-encoder reranking | ✅ Done |
| 5 | LLM synthesis + citation engine | ✅ Done |
| 6 | Automated quiz generator | ✅ Done |
| 7 | Knowledge tracing / mastery scoring | 🔲 Planned |
| 8 | Streamlit dashboard | 🔲 Planned |
| 9 | Full integration + visual citations | 🔲 Planned |
| 10 | Evaluation & performance tuning | 🔲 Planned |

---

## ⚠️ Known Limitations

- **Retrieval quality is corpus-size dependent** — on very small document sets (a handful of chunks), retrieval can return low-relevance results since there isn't enough content to be selective. Works better with a full-length textbook/course rather than a short test file.
- **Transcription accuracy depends on Whisper's `base` model** — proper nouns and character/technical names can be mis-transcribed, which the RAG pipeline will faithfully (and correctly) cite even if the underlying transcription was wrong.
- **No automated test suite yet** — validation so far has been manual; a `pytest` suite is planned before final submission.
- **Image embeddings (CLIP) are indexed separately from text** and not yet surfaced in citations — that lands in Day 9 (visual citation display).
- **No relevance threshold on retrieval yet** — `hybrid_retrieve()` currently always returns top-5 results even if none are strongly relevant; a minimum score cutoff is planned for Day 10.

---

## 🗺️ Roadmap

- [ ] Bayesian/logistic-regression knowledge tracing with per-topic mastery scores
- [ ] Streamlit dashboard: Study Workspace, Quiz Center, Analytics
- [ ] Clickable video-timestamp jump links in chat
- [ ] Inline diagram rendering next to answers
- [ ] Retrieval precision/recall evaluation on sample queries
- [ ] Config centralization, logging, and basic test coverage

---

## 📄 License

*(add your chosen license here, e.g. MIT)*
