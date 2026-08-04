import os
import json
import fitz  
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

current_script_path = os.path.abspath(__file__)
database_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(database_dir)
project_root = os.path.dirname(src_dir)
chroma_db_path = os.path.join(project_root, "data", "chroma_db")

client = chromadb.PersistentClient(path=chroma_db_path)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(
    name="course_materials",
    embedding_function=embedding_fn
)

# 3. CONFIGURE RECURSIVE CHUNKER
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)


def process_and_index_pdf(pdf_path):
    """Extracts text from PDF, breaks it into chunks, and stores them in ChromaDB."""
    if not os.path.exists(pdf_path):
        print(f" File not found: '{pdf_path}'")
        return

    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    print(f"📄 Indexing PDF: {filename} ({len(doc)} pages)...")

    chunks_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text()
        if not raw_text.strip():
            continue

        page_chunks = text_splitter.split_text(raw_text)
        for chunk_idx, chunk in enumerate(page_chunks):
            chunks_to_add.append(chunk)
            metadatas_to_add.append({
                "source": filename,
                "page": page_num,
                "type": "pdf_text"
            })
            ids_to_add.append(f"{filename}_p{page_num}_c{chunk_idx}")

    if chunks_to_add:
        collection.add(
            documents=chunks_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f" Successfully added {len(chunks_to_add)} PDF chunks to ChromaDB!")


def process_and_index_transcript(transcript_json_path):
    """Loads a Whisper-style transcript JSON and stores each segment in ChromaDB,
    keeping start/end timestamps in metadata so answers can cite [Video @ MM:SS]."""
    if not os.path.exists(transcript_json_path):
        print(f" File not found: '{transcript_json_path}'")
        return

    with open(transcript_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = data.get("file_name", os.path.basename(transcript_json_path))
    segments = data.get("transcript_segments", [])
    print(f"🎬 Indexing video transcript: {filename} ({len(segments)} segments)...")

    chunks_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    buffer_text = ""
    buffer_start = None
    buffer_end = None
    chunk_idx = 0

    def flush_buffer():
        nonlocal buffer_text, buffer_start, buffer_end, chunk_idx
        if buffer_text.strip():
            chunks_to_add.append(buffer_text.strip())
            metadatas_to_add.append({
                "source": filename,
                "start": buffer_start,
                "end": buffer_end,
                "type": "video_transcript"
            })
            ids_to_add.append(f"{filename}_seg{chunk_idx}")
            chunk_idx += 1
        buffer_text = ""
        buffer_start = None
        buffer_end = None

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if buffer_start is None:
            buffer_start = seg["start"]
        buffer_end = seg["end"]
        buffer_text += " " + text

        if len(buffer_text) >= 500:
            flush_buffer()

    flush_buffer()  

    if chunks_to_add:
        collection.add(
            documents=chunks_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f" Successfully added {len(chunks_to_add)} video chunks to ChromaDB!")


if __name__ == "__main__":
    sample_pdf = os.path.join(project_root, "data", "raw_pdfs", "sample_fonts_test.pdf")
    process_and_index_pdf(sample_pdf)

    sample_transcript = os.path.join(
        src_dir, "parsers", "outputs", "Lecture_transcript.json"
    )
    process_and_index_transcript(sample_transcript)

    print("\n Sanity check query:")
    results = collection.query(query_texts=["What is the story about?"], n_results=3)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        location = meta.get("page", meta.get("start"))
        print(f"  - [{meta['type']} @ {location}] {doc[:80]}")