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

    print(" Building BM25 keyword index from ChromaDB contents...")
    all_data = collection.get(include=["documents", "metadatas"])

    _bm25_corpus_ids = all_data["ids"]
    _bm25_corpus_docs = all_data["documents"]
    _bm25_corpus_metas = all_data["metadatas"]

    tokenized_corpus = [_tokenize(doc) for doc in _bm25_corpus_docs]
    _bm25_index = BM25Okapi(tokenized_corpus)

    print(f" BM25 index ready with {len(_bm25_corpus_docs)} chunks.")

def vector_search(query, top_k=20, source_filter=None):
    """Semantic search via ChromaDB — finds chunks with similar MEANING.
    If source_filter is given, only searches chunks from that exact uploaded file."""
    where_clause = {"source": source_filter} if source_filter else None
    results = collection.query(query_texts=[query], n_results=top_k, where=where_clause)

    hits = []
    for doc, meta, doc_id in zip(
        results["documents"][0], results["metadatas"][0], results["ids"][0]
    ):
        hits.append({"id": doc_id, "text": doc, "metadata": meta})
    return hits


def keyword_search(query, top_k=20, source_filter=None):
    """Keyword search via BM25 — finds chunks with matching TERMS.
    If source_filter is given, only considers chunks from that exact uploaded file."""
    _build_bm25_index()

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    candidate_indices = range(len(scores))
    if source_filter:
        candidate_indices = [
            i for i in candidate_indices
            if _bm25_corpus_metas[i].get("source") == source_filter
        ]

    ranked_indices = sorted(
        candidate_indices, key=lambda i: scores[i], reverse=True
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

def reciprocal_rank_fusion(vector_hits, keyword_hits, k=60, top_n=20):
    """
    Merges two differently-scored ranked lists into one ranking, using
    each result's RANK (position) rather than its raw score -- because
    cosine similarity and BM25 scores are on completely different scales
    and can't be compared directly.
    """
    fused_scores = {}
    doc_lookup = {}

    for rank, hit in enumerate(vector_hits):
        fused_scores[hit["id"]] = fused_scores.get(hit["id"], 0) + 1 / (k + rank + 1)
        doc_lookup[hit["id"]] = hit

    for rank, hit in enumerate(keyword_hits):
        fused_scores[hit["id"]] = fused_scores.get(hit["id"], 0) + 1 / (k + rank + 1)
        doc_lookup[hit["id"]] = hit

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_n]
    return [doc_lookup[doc_id] for doc_id in ranked_ids]

_reranker = None


def _get_reranker():
    """Lazily loads the cross-encoder reranker model (downloads ~1.1GB first run)."""
    global _reranker
    if _reranker is None:
        print(" Loading cross-encoder reranker (BAAI/bge-reranker-base)...")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def rerank(query, candidates, top_k=5):
    """
    Re-scores each candidate chunk by reading it TOGETHER with the query
    (unlike vector/BM25 search, which score query and chunk separately),
    then keeps only the top_k most relevant ones.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    top_candidates = scored_candidates[:top_k]

    results = []
    for hit, score in top_candidates:
        hit_with_score = dict(hit)
        hit_with_score["rerank_score"] = float(score)
        results.append(hit_with_score)
    return results

def get_indexed_sources():
    """Returns the list of distinct source filenames currently indexed --
    used by the UI to build a 'which document?' dropdown."""
    all_data = collection.get(include=["metadatas"])
    sources = {meta.get("source") for meta in all_data["metadatas"] if meta.get("source")}
    return sorted(sources)


def hybrid_retrieve(query, vector_k=20, keyword_k=20, fusion_k=20, final_k=5, source_filter=None):
    """
    Full Day 4 pipeline: vector search + keyword search -> RRF fusion
    -> cross-encoder rerank -> top final_k context chunks.

    source_filter: if given (e.g. "Shaheryar_Amir_CV.pdf"), retrieval is
    scoped to ONLY that uploaded document -- prevents topic drift where
    an unrelated indexed file leaks into results (e.g. asking about your
    CV and getting pirate-novel chunks back because both happen to be indexed).
    """
    vector_hits = vector_search(query, top_k=vector_k, source_filter=source_filter)
    keyword_hits = keyword_search(query, top_k=keyword_k, source_filter=source_filter)

    fused_candidates = reciprocal_rank_fusion(
        vector_hits, keyword_hits, top_n=fusion_k
    )

    final_results = rerank(query, fused_candidates, top_k=final_k)
    return final_results


if __name__ == "__main__":
    test_query = "When the story becomes a trailer fight?"
    print(f"\n Query: {test_query}\n")

    top_chunks = hybrid_retrieve(test_query)

    for i, chunk in enumerate(top_chunks, start=1):
        meta = chunk["metadata"]
        location = meta.get("page", meta.get("start"))
        print(f"{i}. [{meta['type']} @ {location}] score={chunk['rerank_score']:.3f}")
        print(f"   {chunk['text'][:150]}...\n")