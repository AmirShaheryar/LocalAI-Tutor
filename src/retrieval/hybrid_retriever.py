import os
import re
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

current_script_path = os.path.abspath(__file__)
retrieval_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(retrieval_dir)
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

_bm25_index = None
_bm25_corpus_ids = None
_bm25_corpus_docs = None
_bm25_corpus_metas = None


def _tokenize(text):
    """Lowercase and split into word tokens for BM25 matching."""
    return re.findall(r"\w+", text.lower())


def _build_bm25_index():
    """Pulls every chunk out of ChromaDB once and builds an in-memory BM25 index."""
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_docs, _bm25_corpus_metas

    if _bm25_index is not None:
        return  

    print("🔤 Building BM25 keyword index from ChromaDB contents...")
    all_data = collection.get(include=["documents", "metadatas"])

    _bm25_corpus_ids = all_data["ids"]
    _bm25_corpus_docs = all_data["documents"]
    _bm25_corpus_metas = all_data["metadatas"]

    tokenized_corpus = [_tokenize(doc) for doc in _bm25_corpus_docs]
    _bm25_index = BM25Okapi(tokenized_corpus)

    print(f" BM25 index ready with {len(_bm25_corpus_docs)} chunks.")

def vector_search(query, top_k=20):
    """Semantic search via ChromaDB — finds chunks with similar MEANING."""
    results = collection.query(query_texts=[query], n_results=top_k)

    hits = []
    for doc, meta, doc_id in zip(
        results["documents"][0], results["metadatas"][0], results["ids"][0]
    ):
        hits.append({"id": doc_id, "text": doc, "metadata": meta})
    return hits


def keyword_search(query, top_k=20):
    """Keyword search via BM25 — finds chunks with matching TERMS."""
    _build_bm25_index()

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    hits = []
    for i in ranked_indices:
        if scores[i] <= 0:
            continue  # no keyword overlap at all -- not a real match
        hits.append({
            "id": _bm25_corpus_ids[i],
            "text": _bm25_corpus_docs[i],
            "metadata": _bm25_corpus_metas[i],
        })
    return hits