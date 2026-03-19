import re 
import cohere
import faiss
import tempfile
import numpy as np
import sympy as sp 
from backend.agents import *
from backend.agents.nodes.tools import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi
from tavily import TavilyClient

_STORES: Dict[str, Dict[str, Any]] = {}

def _cohere_client() -> cohere.Client:
    api_key = _get_secret("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY is not set — add it to .env (local) or Streamlit Cloud secrets.")
    return cohere.Client(api_key)


def _tavily_client() -> TavilyClient:
    api_key = _get_secret("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set — add it to .env (local) or Streamlit Cloud secrets.")
    return TavilyClient(api_key)


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
    return vecs / norms   

def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25 (removes punctuation, lowercases)."""
    return re.findall(r'(?u)\b\w+\b', text.lower())


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
    metadata = [c.metadata for c in chunks]

    if not texts:
        raise ValueError("No text chunks produced from PDF.")

    tokenized_chunks = [_tokenize(t) for t in texts]

    logger.info(f"[INGEST] Embedding {len(texts)} chunks | file={fname} | thread={thread_id}")
    vecs = _embed_texts(texts, EMBED_INPUT_TYPE_DOC)

    existing = _STORES.get(thread_id)

    if existing:
        existing["index"].add(vecs)

        # Append doc vectors for later relevance scoring
        if "doc_vecs" in existing and len(existing["doc_vecs"]) > 0:
            existing["doc_vecs"] = np.vstack((existing["doc_vecs"], vecs))
        else:
            existing["doc_vecs"] = vecs

        existing["chunks"].extend(texts)
        existing["metadata"].extend(metadata)
        existing["tokenized_chunks"].extend(tokenized_chunks)
        existing["bm25"] = BM25Okapi(existing["tokenized_chunks"])

        if fname not in existing["filenames"]:
            existing["filenames"].append(fname)
        total = len(existing["chunks"])
        logger.info(f"[INGEST] Appended to existing index | total_chunks={total} | thread={thread_id}")
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        index.add(vecs)
        bm25 = BM25Okapi(tokenized_chunks)
        _STORES[thread_id] = {
            "index":           index,
            "chunks":          texts,
            "metadata":        metadata,
            "filenames":       [fname],
            "bm25":            bm25,
            "tokenized_chunks": tokenized_chunks,
            "doc_vecs":        vecs,
        }
        total = len(texts)
        logger.info(f"[INGEST] New index created | chunks={total} | thread={thread_id}")

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
    logger.info(f"[CRAG] Store cleared | thread={thread_id}")

# ══════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN TOOLS  (bound to solver_agent via bind_tools)
# ══════════════════════════════════════════════════════════════════════════════

@tool
def rag_tool(query: str, thread_id: str) -> str:
    """
    Hybrid CRAG (Corrective Retrieval-Augmented Generation) tool:
    • Sparse search (BM25) + Dense search (Cohere embeddings)
    • Reciprocal Rank Fusion (exact same algorithm as LangChain EnsembleRetriever)
    • Final relevance filter (cosine similarity >= 0.30) — this is the "Corrective" part

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
    No extra latency — everything runs in-memory (< 20 ms on typical threads).
    """
    if not thread_id or not has_store(thread_id):
        return (
            "CRAG ERROR: No document is indexed for this session. "
            "Do NOT call rag_tool again — use web_search_tool or your own knowledge instead."
        )

    store = _STORES[thread_id]
    logger.info(f"[CRAG] query='{query[:80]}' | thread={thread_id} | "
                f"index_size={store['index'].ntotal}")

    # ── Dense retrieval (Cohere) ─────────────────────────────────────────────
    q_vec = _embed_texts([query], EMBED_INPUT_TYPE_QUERY)
    distances, indices = store["index"].search(q_vec, 10)
    dense_idx = indices[0]

    # ── Sparse retrieval (BM25) ──────────────────────────────────────────────
    query_tokens = _tokenize(query)
    sparse_scores = store["bm25"].get_scores(query_tokens)
    sparse_idx = np.argsort(sparse_scores)[::-1][:10]

    # ── Reciprocal Rank Fusion (LangChain-style RRF) ─────────────────────────
    rrf_scores: Dict[int, float] = {}
    K = 60
    for rank, idx in enumerate(dense_idx):
        if idx < 0 or idx >= len(store["chunks"]):
            continue
        rrf_scores.setdefault(idx, 0.0)
        rrf_scores[idx] += 1.0 / (K + rank + 1)
    for rank, idx in enumerate(sparse_idx):
        if idx < 0 or idx >= len(store["chunks"]):
            continue
        rrf_scores.setdefault(idx, 0.0)
        rrf_scores[idx] += 1.0 / (K + rank + 1)

    fused_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:TOP_K]

    # ── Relevance decision (corrective step) ─────────────────────────────────
    results: List[str] = []
    for idx in fused_indices:
        if idx < 0 or idx >= len(store["chunks"]):
            continue
        dense_score = float(np.dot(q_vec[0], store["doc_vecs"][idx]))
        if dense_score < MIN_SCORE:
            logger.debug(f"[CRAG] Skipping chunk idx={idx} score={dense_score:.3f} < {MIN_SCORE}")
            continue

        meta     = store["metadata"][idx]
        page_num = meta.get("page", "?")
        passage  = store["chunks"][idx].strip()
        results.append(f"[Page {page_num} | score={dense_score:.3f}]\n{passage}")

    if not results:
        logger.info("[CRAG] No chunks passed the relevance threshold")
        return (
            f"CRAG: No sufficiently relevant passages found in '{', '.join(store['filenames'])}' "
            f"for query: '{query}'. "
            "Fall back to web_search_tool or your own knowledge."
        )

    header = (
        f"Hybrid CRAG results from '{', '.join(store['filenames'])}' "
        f"— {len(results)} passage(s) found:\n\n"
    )
    logger.info(f"[CRAG] Returning {len(results)} chunks")
    return header + "\n\n---\n\n".join(results)


@tool
def web_search_tool(query: str) -> str:
    """
    Tavily AI-powered web search (fastest, most accurate, summarized results).

    HOW THE SOLVER LLM SHOULD USE IT (3-query strategy):
    1. In your reasoning step, decide you need external info.
    2. Generate up to 3 complementary queries (one call per query):
       - Query 1: core concept / formula / proof
       - Query 2: worked example / step-by-step solution
       - Query 3: application / edge case / common mistake (if needed)
    3. Call web_search_tool UP TO THREE TIMES IN PARALLEL in the SAME turn.
       Example:
         web_search_tool("Bayes theorem exact formula and proof")
         web_search_tool("Bayes theorem probability without replacement worked example")
         web_search_tool("Bayes theorem real-world application step by step")
    4. In the next reasoning turn you will receive 3 separate responses.
       Simply combine the best parts (direct answers + top snippets) into your final solution.

    WHEN TO CALL (flexible — not only as RAG fallback):
    - Anytime you want fresh, reliable web information (even if RAG succeeded)
    - When RAG returned empty / insufficient context
    - For confirmation, extra examples, proofs, or real-world context
    - For any general math topic, theorems, or worked examples

    The tool always returns clean, ready-to-use output (Tavily direct answer + top 5 summarized results).

    max_results=7 (Tavily API limit per query) — only top 5 are shown for brevity.
    """

    if not query.strip():
        return "No query provided for web search."

    try:
        client = _tavily_client()
        search_result = client.search(
            query=query,
            search_depth="advanced",
            max_results=7,
            include_answer=True,
            include_images=False,
        )

        parts = []
        if search_result.get("answer"):
            parts.append(f"**Tavily Direct Answer**\n{search_result['answer']}\n")

        for i, res in enumerate(search_result.get("results", [])[:5]):
            parts.append(
                f"[{i+1}] **{res.get('title', 'No title')}**\n"
                f"URL: {res.get('url', '')}\n"
                f"Snippet: {res.get('content', '')[:600]}...\n"
            )

        if not parts:
            return "Tavily: No results found for this query."

        return "Tavily Web Search Results:\n\n" + "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Web search error: {str(e)}. Try a different query or fall back to your knowledge."


@tool
def calculator_tool(expression: str) -> str:
    """
    Symbolic math calculator (SymPy backend) — use ONLY when heavy computation or high precision is REQUIRED.

    The LLM can handle ALL intermediate JEE-level calculations itself (basic arithmetic, trig identities,
    simple integrals, derivatives, etc.). Do NOT call this tool for "stupid" or routine stuff — it wastes
    a tool call and adds zero value.

    The symbolic calculator ONLY adds real value in these three narrow cases (rare in JEE):
    1. Very large factorial / combinatorics numbers (e.g. C(50,25), 100!, permutations)
    2. High-precision decimal results that the problem explicitly asks for
    3. Matrix operations with large dimensions

    Use sparingly and only when one of the above conditions is met.

    Input examples (use these formats):
    - "binomial(50, 25)" or "factorial(100)"
    - "N(integrate(1/sqrt(1-x**2), x), 50)"   # high precision to 50 digits
    - "Matrix([[1,2,3],[4,5,6],[7,8,9]]) * Matrix([[9,8,7],[6,5,4],[3,2,1]])"

    Returns numerical result via SymPy N evaluation.
    """

    try:
        expr = sp.sympify(expression)
        result = sp.N(expr)
        return str(result)
    except Exception as e:
        return f"Error: {e}"