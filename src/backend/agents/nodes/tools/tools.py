from __future__ import annotations

import re
import tempfile
from backend.agents import Any, Dict, List, Optional, os

import cohere
import faiss
import numpy as np
import sympy as sp
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from backend.agents import logger
from backend.agents.nodes.tools import (
    _get_secret,
    COHERE_EMBED_MODEL,
    EMBED_INPUT_TYPE_DOC,
    EMBED_INPUT_TYPE_QUERY,
    EMBED_DIM,
    MIN_SCORE,
    TOP_K,
)
from backend.agents.nodes.tools.mcp.tavily_mcp_client import tavily_mcp_search


# ══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY VECTOR STORE  (one index per thread_id)
# ══════════════════════════════════════════════════════════════════════════════

_STORES: Dict[str, Dict[str, Any]] = {}


# ── Cohere client ─────────────────────────────────────────────────────────────

def _cohere_client() -> cohere.Client:
    api_key = _get_secret("COHERE_API_KEY")
    if not api_key:
        raise ValueError(
            "COHERE_API_KEY is not set — add it to .env or Streamlit secrets."
        )
    return cohere.Client(api_key)


# ── Embedding helper ──────────────────────────────────────────────────────────

def _embed_texts(texts: List[str], input_type: str) -> np.ndarray:
    """Embed texts via Cohere, return L2-normalised float32 vectors."""
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
    return vecs / norms


def _tokenize(text: str) -> List[str]:
    return re.findall(r"(?u)\b\w+\b", text.lower())


# ══════════════════════════════════════════════════════════════════════════════
#  PDF INGESTION  (called by app.py on upload)
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Embed and index a PDF for the given thread.
    Calling this a second time APPENDS to the existing index — all uploaded
    PDFs are searched together.

    Returns: {filename, pages, chunks, total_chunks}
    """
    if not file_bytes:
        raise ValueError("Empty file bytes — nothing to ingest.")

    fname = filename or "document.pdf"

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
    metadata = [c.metadata for c in chunks]

    if not texts:
        raise ValueError("No text chunks produced from PDF.")

    tokenized_chunks = [_tokenize(t) for t in texts]

    logger.info(
        f"[INGEST] Embedding {len(texts)} chunks | file={fname} | thread={thread_id}"
    )
    vecs = _embed_texts(texts, EMBED_INPUT_TYPE_DOC)

    existing = _STORES.get(thread_id)
    if existing:
        existing["index"].add(vecs)
        existing["doc_vecs"] = (
            np.vstack((existing["doc_vecs"], vecs))
            if len(existing.get("doc_vecs", [])) > 0
            else vecs
        )
        existing["chunks"].extend(texts)
        existing["metadata"].extend(metadata)
        existing["tokenized_chunks"].extend(tokenized_chunks)
        existing["bm25"] = BM25Okapi(existing["tokenized_chunks"])
        if fname not in existing["filenames"]:
            existing["filenames"].append(fname)
        total = len(existing["chunks"])
        logger.info(
            f"[INGEST] Appended | total_chunks={total} | thread={thread_id}"
        )
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(vecs)
        _STORES[thread_id] = {   # passing the sparse and dense vectors 
            "index":            index,
            "chunks":           texts,
            "metadata":         metadata,
            "filenames":        [fname],
            "bm25":             BM25Okapi(tokenized_chunks),
            "tokenized_chunks": tokenized_chunks,
            "doc_vecs":         vecs,
        }
        total = len(texts)
        logger.info(
            f"[INGEST] New index created | chunks={total} | thread={thread_id}"
        )

    return {
        "filename":     fname,
        "pages":        len(docs),
        "chunks":       len(texts),
        "total_chunks": total,
    }


def get_store_info(thread_id: str) -> Optional[dict]:
    store = _STORES.get(thread_id)
    if not store:
        return None
    return {
        "filenames": store["filenames"],
        "filename":  ", ".join(store["filenames"]),
        "chunks":    len(store["chunks"]),
    }


def has_store(thread_id: str) -> bool:
    return thread_id in _STORES


def clear_store(thread_id: str) -> None:
    _STORES.pop(thread_id, None)
    logger.info(f"[CRAG] Store cleared | thread={thread_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — HYBRID CRAG
# ══════════════════════════════════════════════════════════════════════════════

@tool
def rag_tool(query: str, thread_id: str) -> str:
    """
    Hybrid CRAG (Corrective Retrieval-Augmented Generation).

    Pipeline: BM25 sparse + Cohere dense → Reciprocal Rank Fusion
              → Corrective relevance filter (cosine ≥ 0.30)

    ── CALLING RULES ────────────────────────────────────────────────────────
    • ALWAYS call this first when a document is uploaded for the session —
      even if you think the problem is straightforward.
      The student uploaded their notes for a reason.

    • Use a FOCUSED query, not the full problem text.
      Good:  "integration by parts formula"
      Bad:   "find the integral of x²sin(x)dx using any method you know"

    • If this returns "no relevant passages found", do NOT retry it.
      Fall through to web_search_tool or your own knowledge.

    • If no document is indexed, this returns a clear skip message.
      Do NOT call rag_tool again after that message.

    Args:
        query     : Focused retrieval query.
        thread_id : Session thread ID (injected by the graph).

    Returns:
        Ranked passages with page numbers and scores, or a clear
        "no relevant content" message.
    """
    # ── Guard: no store ───────────────────────────────────────────────────────
    if not thread_id or not has_store(thread_id):
        logger.info(f"[CRAG] No store for thread={thread_id} — skipping")
        return (
            "CRAG: No document indexed for this session. "
            "Do NOT call rag_tool again — use web_search_tool or your own knowledge."
        )

    store = _STORES[thread_id]
    logger.info(
        f"[CRAG] query='{query[:80]}' | thread={thread_id} | "
        f"index_size={store['index'].ntotal}"
    )

    # ── Dense retrieval (Cohere) ──────────────────────────────────────────────
    q_vec = _embed_texts([query], EMBED_INPUT_TYPE_QUERY)
    _, indices = store["index"].search(q_vec, 10)
    dense_idx  = indices[0]

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    tokens        = _tokenize(query)
    sparse_scores = store["bm25"].get_scores(tokens)
    sparse_idx    = np.argsort(sparse_scores)[::-1][:10]

    # ── Reciprocal Rank Fusion (LangChain EnsembleRetriever algorithm) ────────
    K          = 60
    rrf: Dict[int, float] = {}
    for rank, idx in enumerate(dense_idx):    ## just using the same yet simple form of RRF as one provided by langchain 
        if 0 <= idx < len(store["chunks"]):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)
    for rank, idx in enumerate(sparse_idx):
        if 0 <= idx < len(store["chunks"]):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)

    fused = sorted(rrf, key=rrf.get, reverse=True)[:TOP_K]

    # ── Corrective filter (the "C" in CRAG) ───────────────────────────────────
    results: List[str] = []
    for idx in fused:
        if not (0 <= idx < len(store["chunks"])):
            continue
        cos_sim = float(np.dot(q_vec[0], store["doc_vecs"][idx]))
        if cos_sim < MIN_SCORE:
            logger.debug(f"[CRAG] Dropped idx={idx} cos={cos_sim:.3f} < {MIN_SCORE}")
            continue
        page    = store["metadata"][idx].get("page", "?")
        passage = store["chunks"][idx].strip()
        results.append(f"[Page {page} | relevance={cos_sim:.3f}]\n{passage}")

    filenames = ", ".join(store["filenames"])

    # ── No relevant content — graceful skip, NOT an error ─────────────────────
    if not results:
        msg = (
            f"CRAG: No relevant passages found in '{filenames}' for query '{query}'. "
            "Document does not appear to cover this topic. "
            "Proceeding with web_search_tool or own knowledge."
        )
        logger.info(f"[CRAG] {msg}")
        return msg

    logger.info(f"[CRAG] Returning {len(results)} passages from '{filenames}'")
    return (
        f"Hybrid CRAG — {len(results)} passage(s) from '{filenames}':\n\n"
        + "\n\n---\n\n".join(results)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 2 — WEB SEARCH  (Tavily MCP — general)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def web_search_tool(query: str) -> str:
    """
    Real-time web search via the Tavily MCP server (remote, no local setup).

    Returns Tavily AI direct answer + top 5 ranked results with snippets.

    ── WHEN TO CALL ──────────────────────────────────────────────────────────
    • Student asks about recent math discoveries or research breakthroughs
    • Student asks for NEW JEE Mains / Advanced questions on a topic
    • Student asks for study resources, textbooks, or video explanations
    • Student asks about Olympiad problems (IMO, Putnam, USAMO, RMO, etc.)
    • CRAG returned empty or insufficient context
    • Any factual question requiring current or up-to-date information

    ── MULTI-QUERY STRATEGY (call up to 3× in the same turn) ─────────────────
      Query 1 → core formula / theorem / concept
      Query 2 → worked example or step-by-step solution
      Query 3 → edge case, common mistake, or application (if needed)

    ── DO NOT USE FOR ─────────────────────────────────────────────────────────
    • Computing math — use your own reasoning or calculator_tool
    • Topics already covered by the student's uploaded notes — use rag_tool first

    Args:
        query: A focused, specific search query.
               Examples:
                 "JEE Mains 2025 coordinate geometry new pattern questions"
                 "recent breakthroughs Riemann hypothesis 2024 2025"
                 "integration by parts tricky JEE Advanced worked example"

    Returns:
        Tavily AI direct answer + top 5 results (title, URL, snippet).
    """
    if not query.strip():
        return "No query provided."

    logger.info(f"[TavilyMCP] web_search_tool | query='{query[:80]}'")

    result = tavily_mcp_search(
        query          = query,
        search_depth   = "advanced",
        topic          = "general",
        max_results    = 5,
    )

    logger.info(f"[TavilyMCP] web_search_tool done | {len(result)} chars returned")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL 3 — SYMBOLIC CALCULATOR  (SymPy)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def calculator_tool(expression: str) -> str:
    """
    Symbolic mathematics calculator (SymPy backend). Use SPARINGLY.

    The solver LLM handles ALL routine JEE-level computation itself.
    Only call this for these three narrow cases:

      1. Very large factorials / combinatorics  e.g. C(50,25), 100!
      2. High-precision decimal results the problem explicitly asks for
      3. Large matrix operations (det, inverse, eigenvalues)

    DO NOT CALL for: basic arithmetic, trig identities, standard integrals,
    derivatives, or probability fractions. These add zero value.

    Valid SymPy expression syntax:
      "binomial(50, 25)"
      "factorial(100)"
      "N(integrate(1/sqrt(1-x**2), x), 50)"    ← 50-digit precision
      "Matrix([[1,2,3],[4,5,6],[7,8,9]]).det()"

    Args:
        expression: A valid SymPy expression string.

    Returns:
        Numeric or symbolic result as a string.
    """
    try:
        expr   = sp.sympify(expression)
        result = sp.N(expr)
        logger.info(
            f"[Calculator] {expression[:60]} → {str(result)[:60]}"
        )
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}. Check SymPy expression syntax."


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    rag_tool,
    web_search_tool,
    calculator_tool,
]