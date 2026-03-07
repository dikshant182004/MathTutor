import os
import tempfile
from typing import Dict, List, Optional, Any

import cohere
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from backend.logger import get_logger

logger = get_logger(__name__)

COHERE_EMBED_MODEL     = "embed-english-v3.0"
EMBED_INPUT_TYPE_DOC   = "search_document"
EMBED_INPUT_TYPE_QUERY = "search_query"
EMBED_DIM              = 1024

# Minimum cosine similarity to include a chunk in results.
# IndexFlatIP on L2-normalised vectors = cosine similarity in [-1, 1].
# Chunks scoring below 0.30 are almost certainly off-topic.
MIN_SCORE = 0.30

# Number of chunks to retrieve per query
TOP_K = 5

# ── In-memory stores keyed by thread_id ──────────────────────────────────────
# Structure per thread:
#   index     : faiss.IndexFlatIP
#   chunks    : List[str]      — raw text of every chunk
#   metadata  : List[dict]     — {source, page} per chunk
#   filenames : List[str]      — display names of all ingested docs
_STORES: Dict[str, Dict[str, Any]] = {}

_duckduck = DuckDuckGoSearchRun(region="us-en")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cohere_client() -> cohere.Client:
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable is not set.")
    return cohere.Client(api_key)


def _embed_texts(texts: List[str], input_type: str) -> np.ndarray:
    """Embed texts via Cohere and return L2-normalised float32 vectors."""
    client   = _cohere_client()
    BATCH    = 96
    all_vecs: List[List[float]] = []

    for i in range(0, len(texts), BATCH):
        batch    = texts[i : i + BATCH]
        response = client.embed(
            texts=batch,
            model=COHERE_EMBED_MODEL,
            input_type=input_type,
        )
        all_vecs.extend(response.embeddings)

    vecs  = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms   # L2-normalised → inner product == cosine similarity


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC INGESTION API  (called by app.py sidebar)
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(file_bytes: bytes, thread_id: str,
               filename: Optional[str] = None) -> dict:
    """
    Embed and index a PDF for the given thread.

    MULTI-DOC: calling this a second time ADDS to the existing index instead
    of replacing it — all uploaded PDFs are searched together in one index.

    Returns: {filename, pages, chunks, total_chunks}
    """
    if not file_bytes:
        raise ValueError("Empty file bytes — nothing to ingest.")

    fname = filename or "document.pdf"

    # PyPDFLoader needs a real path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not docs:
        raise ValueError("PDF produced no pages — is it a scanned/image PDF?")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks   = splitter.split_documents(docs)
    texts    = [c.page_content for c in chunks]
    metadata = [c.metadata     for c in chunks]

    if not texts:
        raise ValueError("No text chunks produced from PDF.")

    logger.info(f"[RAG] Embedding {len(texts)} chunks | file={fname} | thread={thread_id}")
    vecs = _embed_texts(texts, EMBED_INPUT_TYPE_DOC)

    existing = _STORES.get(thread_id)

    if existing:
        # ── Append to existing index — preserves previously uploaded docs ─────
        existing["index"].add(vecs)
        existing["chunks"].extend(texts)
        existing["metadata"].extend(metadata)
        if fname not in existing["filenames"]:
            existing["filenames"].append(fname)
        total = len(existing["chunks"])
        logger.info(f"[RAG] Appended to existing index | total_chunks={total} | thread={thread_id}")
    else:
        # ── Fresh index for this thread ───────────────────────────────────────
        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(vecs)
        _STORES[thread_id] = {
            "index":     index,
            "chunks":    texts,
            "metadata":  metadata,
            "filenames": [fname],
        }
        total = len(texts)
        logger.info(f"[RAG] New index created | chunks={total} | thread={thread_id}")

    return {
        "filename":     fname,
        "pages":        len(docs),
        "chunks":       len(texts),
        "total_chunks": total,
    }


def get_store_info(thread_id: str) -> Optional[dict]:
    """Return store summary for the thread, or None if nothing is indexed."""
    store = _STORES.get(thread_id)
    if not store:
        return None
    return {
        "filenames": store["filenames"],
        "filename":  ", ".join(store["filenames"]),  # backward-compat for app.py
        "chunks":    len(store["chunks"]),
    }


def has_store(thread_id: str) -> bool:
    """True if at least one document has been indexed for this thread."""
    return thread_id in _STORES


def clear_store(thread_id: str) -> None:
    """Drop the entire vector store for a thread (called on New Chat)."""
    _STORES.pop(thread_id, None)
    logger.info(f"[RAG] Store cleared | thread={thread_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN TOOLS  (bound to solver_agent via bind_tools)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def rag_tool(query: str, thread_id: str) -> str:
    """
    Search the student's uploaded study material for relevant passages.

    WHEN TO CALL:
    - ONLY call this when a PDF has been uploaded for the session.
      If no document is indexed the tool will say so — do not retry it.
    - Call BEFORE web_search_tool for any problem that could be covered
      by the student's notes or textbook.
    - Use a focused query — e.g. "Bayes theorem formula" or
      "probability without replacement" — not the full problem text.
    - If this returns empty context, fall back to web_search_tool.

    Returns a plain-text string with the most relevant passages from the
    uploaded document, ready to use directly in your solution.
    """
    if not thread_id or not has_store(thread_id):
        return (
            "RAG ERROR: No document is indexed for this session. "
            "Do NOT call rag_tool again — use web_search_tool or your own knowledge instead."
        )

    store = _STORES[thread_id]
    logger.info(f"[RAG] query='{query[:80]}' | thread={thread_id} | "
                f"index_size={store['index'].ntotal}")

    q_vec              = _embed_texts([query], EMBED_INPUT_TYPE_QUERY)
    distances, indices = store["index"].search(q_vec, TOP_K)

    results: List[str] = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        score = float(dist)
        if score < MIN_SCORE:
            logger.debug(f"[RAG] Skipping chunk idx={idx} score={score:.3f} < {MIN_SCORE}")
            continue
        meta     = store["metadata"][idx]
        page_num = meta.get("page", "?")
        passage  = store["chunks"][idx].strip()
        results.append(f"[Page {page_num} | score={score:.3f}]\n{passage}")

    if not results:
        logger.info("[RAG] No chunks passed the score threshold")
        return (
            f"RAG: No sufficiently relevant passages found in '{', '.join(store['filenames'])}' "
            f"for query: '{query}'. "
            "Fall back to web_search_tool or your own knowledge."
        )

    header = (
        f"RAG results from '{', '.join(store['filenames'])}' "
        f"— {len(results)} passage(s) found:\n\n"
    )
    logger.info(f"[RAG] Returning {len(results)} chunks")
    return header + "\n\n---\n\n".join(results)


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for math concepts, formulas, theorems, or worked examples.

    Use when:
    - No PDF is uploaded for the session, OR
    - rag_tool returned empty context or an error
    - The problem needs general mathematical knowledge not in the study material
    """
    return _duckduck.run(query)