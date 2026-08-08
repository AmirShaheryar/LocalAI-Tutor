import os
import json
import chromadb
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

current_script_path = os.path.abspath(__file__)
database_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(database_dir)
project_root = os.path.dirname(src_dir)
chroma_db_path = os.path.join(project_root, "data", "chroma_db")

client = chromadb.PersistentClient(path=chroma_db_path)
image_collection = client.get_or_create_collection(name="course_images")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()


def embed_image(image_path):
    """Turns one image file into a CLIP embedding vector."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    # normalize -- keeps distances meaningful for cosine similarity
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()


def process_and_index_images(day1_json_path):
    """Reads day1_structured_output.json, embeds every extracted image with CLIP,
    and stores them in the course_images collection with page/source metadata."""
    if not os.path.exists(day1_json_path):
        print(f" File not found: '{day1_json_path}'")
        return

    with open(day1_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = data.get("document", os.path.basename(day1_json_path))
    print(f" Indexing images from: {filename}...")

    embeddings_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for page in data.get("pages", []):
        page_num = page.get("page_number")
        image_paths = page.get("extracted_images", [])

        for img_idx, image_path in enumerate(image_paths, start=1):
            # image paths in the JSON are relative to project_root
            abs_image_path = os.path.join(project_root, image_path) if not os.path.isabs(image_path) else image_path

            if not os.path.exists(abs_image_path):
                print(f"    Skipping missing image: {abs_image_path}")
                continue

            embedding = embed_image(abs_image_path)
            embeddings_to_add.append(embedding)
            metadatas_to_add.append({
                "source": filename,
                "page": page_num,
                "type": "pdf_image",
                "image_path": abs_image_path
            })
            ids_to_add.append(f"{filename}_p{page_num}_img{img_idx}")

    if embeddings_to_add:
        image_collection.add(
            embeddings=embeddings_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f" Successfully added {len(embeddings_to_add)} image embeddings to ChromaDB!")
    else:
        print("  No images found to index.")


def search_images_by_text(query_text, n_results=3):
    """Search diagrams using a plain-text query, e.g. 'diagram of a neural network'.
    Uses CLIP's TEXT encoder so the query lands in the same space as the image vectors."""
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query_embedding = text_features[0].tolist()

    results = image_collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results


if __name__ == "__main__":
    day1_json = os.path.join(project_root, "outputs", "day1_structured_output.json")
    process_and_index_images(day1_json)

    print("\n Sanity check image search:")
    results = search_images_by_text("a diagram or chart")
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  - [page {meta['page']}] {meta['image_path']}  (distance: {dist:.3f})")