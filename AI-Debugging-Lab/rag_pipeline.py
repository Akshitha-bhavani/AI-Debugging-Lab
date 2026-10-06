"""
rag_pipeline.py
---------------
A small, readable RAG pipeline:

  documents -> chunking -> embeddings -> FAISS index -> similarity search -> context

Index files are saved in `vector_store/` so we only embed the knowledge base once.
"""
import json
from pathlib import Path

import numpy as np

import ollama_client

BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"
STORE_DIR = BASE_DIR / "vector_store"
INDEX_FILE = STORE_DIR / "index.faiss"
META_FILE = STORE_DIR / "chunks.json"

# Which knowledge file belongs to which language (concepts are shared by all).
LANGUAGE_FILES = {
    "Python": "python_errors.txt",
    "C": "c_errors.txt",
    "Java": "java_errors.txt",
}
SHARED_FILE = "programming_concepts.txt"


class RAGError(Exception):
    """Any problem with the knowledge base or vector store."""


def _load_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise RAGError("The 'faiss-cpu' package is missing. Run: pip install -r requirements.txt")


# ---------- Step 1: read documents ----------
def load_documents():
    """Return a list of (file_name, text) for every .txt file in knowledge_base/."""
    if not KB_DIR.exists():
        raise RAGError("The 'knowledge_base' folder is missing.")
    documents = []
    for path in sorted(KB_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((path.name, text))
    if not documents:
        raise RAGError("The 'knowledge_base' folder has no readable .txt files.")
    return documents


# ---------- Step 2: chunking ----------
def chunk_text(text, max_chars=800):
    """
    Split text into chunks. Knowledge files are written as short sections
    separated by blank lines, so we split on blank lines and then merge small
    neighbouring sections until a chunk reaches about `max_chars` characters.
    """
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    chunks, current = [], ""
    for section in sections:
        if current and len(current) + len(section) + 2 > max_chars:
            chunks.append(current)
            current = section
        else:
            current = f"{current}\n\n{section}" if current else section
    if current:
        chunks.append(current)
    return chunks


# ---------- Steps 3-4: embeddings + vector database ----------
def _normalize(matrix):
    """Scale vectors to length 1 so inner product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def build_index(embedding_model):
    """Read, chunk, embed and store the whole knowledge base. Returns chunk count."""
    faiss = _load_faiss()
    chunk_records = []
    for file_name, text in load_documents():
        for chunk in chunk_text(text):
            chunk_records.append({"source": file_name, "text": chunk})

    texts = [record["text"] for record in chunk_records]
    vectors = ollama_client.embed(embedding_model, texts)

    matrix = _normalize(np.array(vectors, dtype="float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])  # simple exact-search index
    index.add(matrix)

    try:
        STORE_DIR.mkdir(exist_ok=True)
        faiss.write_index(index, str(INDEX_FILE))
        META_FILE.write_text(
            json.dumps({"embedding_model": embedding_model, "chunks": chunk_records}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RAGError(f"Could not save the vector store: {exc}")
    return len(chunk_records)


def get_status():
    """Small dict used by the sidebar to show the knowledge-base state."""
    if not (INDEX_FILE.exists() and META_FILE.exists()):
        return {"built": False, "chunks": 0, "embedding_model": None}
    try:
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        return {"built": True, "chunks": len(meta["chunks"]),
                "embedding_model": meta.get("embedding_model")}
    except (OSError, ValueError, KeyError):
        return {"built": False, "chunks": 0, "embedding_model": None}


# ---------- Steps 5-7: similarity search ----------
def retrieve(query, language, embedding_model, top_k=4):
    """Return the `top_k` most relevant chunks for the query."""
    faiss = _load_faiss()

    # Build automatically the first time, or if a different embedding model is chosen.
    status = get_status()
    if not status["built"] or status["embedding_model"] != embedding_model:
        build_index(embedding_model)

    try:
        index = faiss.read_index(str(INDEX_FILE))
        chunks = json.loads(META_FILE.read_text(encoding="utf-8"))["chunks"]
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RAGError(f"Vector store is damaged ({exc}). Click 'Rebuild Knowledge Base'.")

    query_vector = _normalize(np.array(ollama_client.embed(embedding_model, [query]), dtype="float32"))
    # Over-fetch, then keep chunks for the chosen language + shared concepts.
    scores, ids = index.search(query_vector, min(top_k * 3, len(chunks)))

    wanted_files = {LANGUAGE_FILES.get(language), SHARED_FILE}
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if chunk["source"] in wanted_files:
            results.append({**chunk, "score": float(score)})
        if len(results) == top_k:
            break
    return results


# ---------- Step 8: format for the prompt ----------
def format_context(results):
    """Turn retrieved chunks into one text block for the prompt."""
    if not results:
        return "No relevant reference material was found."
    return "\n\n---\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
