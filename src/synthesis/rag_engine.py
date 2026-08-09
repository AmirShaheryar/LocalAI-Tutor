import os
import ollama
from src.retrieval.hybrid_retriever import hybrid_retrieve

OLLAMA_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a strict, source-grounded tutor assistant.

RULES YOU MUST FOLLOW:
1. Answer ONLY using the provided CONTEXT below. Never use outside knowledge.
2. If the CONTEXT does not contain enough information to answer, say exactly:
   "I don't have enough information in the provided materials to answer that."
3. Every claim you make MUST end with a citation in one of these exact formats:
   - For PDF sources: [PDF Page N]
   - For video sources: [Video @ MM:SS]
4. Do not invent page numbers or timestamps. Only use the ones given in CONTEXT.
5. Keep answers concise and directly address the question.
"""

def format_timestamp(seconds):
    """Converts raw seconds (e.g. 185.4) into MM:SS (e.g. 03:05) for citations."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def build_context_block(chunks):
    """
    Turns retrieved chunks into a numbered, labeled text block the LLM can
    read AND cite from. Each chunk is tagged with the exact citation string
    it must use, so the model doesn't have to compute page/timestamp itself.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]

        if meta["type"] == "pdf_text":
            citation = f"[PDF Page {meta['page']}]"
        elif meta["type"] == "video_transcript":
            citation = f"[Video @ {format_timestamp(meta['start'])}]"
        else:
            citation = "[Unknown Source]"

        lines.append(f"Source {i} {citation}:\n{chunk['text']}\n")

    return "\n".join(lines)

def generate_answer(query, stream_output=True):
    """
    Full Day 5 pipeline: retrieve -> build context -> stream an LLM answer
    with citations. This is the command-line RAG pipeline deliverable.
    """
    print(" Retrieving relevant context...")
    chunks = hybrid_retrieve(query)

    if not chunks:
        print("No relevant context found in the indexed materials.")
        return "I don't have enough information in the provided materials to answer that."

    context_block = build_context_block(chunks)

    user_prompt = f"""CONTEXT:
{context_block}

QUESTION: {query}

Answer the question using only the CONTEXT above, citing every claim."""

    print(f"\n Answer:\n")

    full_answer = ""
    response_stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=stream_output,
    )

    if stream_output:
        for chunk in response_stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_answer += token
        print("\n")
    else:
        full_answer = response_stream["message"]["content"]
        print(full_answer)

    return full_answer

if __name__ == "__main__":
    print(" Local RAG Tutor -- ask a question (type 'quit' to exit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        generate_answer(query)
        print("-" * 60)