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