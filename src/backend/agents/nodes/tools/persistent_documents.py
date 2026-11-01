"""Persistent, student-scoped PDF retrieval stored in ordinary Redis.

The implementation deliberately does not require Redis Stack/RediSearch: vectors
and metadata are persisted as Redis hashes, and cosine ranking is computed in
application code. This keeps document storage compatible with ordinary Redis commands.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_CHUNKS_PER_DOCUMENT = 500
MAX_DOCUMENTS_PER_STUDENT = 25
DEFAULT_TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _prefix(student_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", student_id or ""):
        raise ValueError("Invalid student namespace.")
    return f"student-doc:{student_id}"


def _doc_key(student_id: str, document_id: str) -> str:
    return f"{_prefix(student_id)}:{document_id}:meta"


def _chunk_key(student_id: str, document_id: str, index: int) -> str:
    return f"{_prefix(student_id)}:{document_id}:chunk:{index}"


def _split_text(text: str) -> list[str]:
    # Keep overlap so formulas/definitions spanning boundaries remain retrievable.
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(normalized) and len(chunks) < MAX_CHUNKS_PER_DOCUMENT:
        end = min(start + CHUNK_SIZE, len(normalized))
        if end < len(normalized):
            boundary = max(normalized.rfind("\n", start + CHUNK_SIZE // 2, end),
                           normalized.rfind(" ", start + CHUNK_SIZE // 2, end))
            if boundary > start:
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(normalized):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _embed(texts: list[str], *, input_type: str) -> list[list[float]]:
    import os
    import cohere

    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Document retrieval is not configured: COHERE_API_KEY is missing.")
    client = cohere.Client(api_key)
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), 96):
        response = client.embed(
            texts=texts[offset:offset + 96],
            model=os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0"),
            input_type=input_type,
        )
        vectors.extend([[float(value) for value in row] for row in response.embeddings])
    normalized: list[list[float]] = []
    for vector in vectors:
        norm = sum(value * value for value in vector) ** 0.5
        normalized.append([value / norm for value in vector] if norm else vector)
    return normalized


def _read_pdf(file_bytes: bytes) -> tuple[int, str]:
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(file_bytes), strict=False)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(value for value in pages if value.strip())
    if not text.strip():
        raise ValueError("No selectable text was found in this PDF. Scanned PDFs need OCR first.")
    return len(reader.pages), text


def list_documents(redis: Any, student_id: str) -> list[dict[str, Any]]:
    prefix = _prefix(student_id)
    ids = redis.zrevrange(f"{prefix}:index", 0, -1)
    documents = []
    for raw_id in ids:
        document_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
        raw = redis.hgetall(_doc_key(student_id, document_id))
        if not raw:
            continue
        item = {
            (k.decode() if isinstance(k, bytes) else str(k)):
            (v.decode() if isinstance(v, bytes) else str(v))
            for k, v in raw.items()
        }
        documents.append({
            "document_id": document_id,
            "filename": item.get("filename", "document.pdf"),
            "pages": int(item.get("pages", 0)),
            "chunks": int(item.get("chunks", 0)),
            "uploaded_at": float(item.get("uploaded_at", 0)),
        })
    return documents


def ingest_pdf(redis: Any, student_id: str, filename: str, file_bytes: bytes) -> dict[str, Any]:
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")
    if len(file_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 10 MiB upload limit.")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF documents are supported.")
    if len(list_documents(redis, student_id)) >= MAX_DOCUMENTS_PER_STUDENT:
        raise ValueError("A student can keep at most 25 indexed PDFs. Delete an older document first.")
    try:
        pages, text = _read_pdf(file_bytes)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The uploaded PDF could not be parsed.") from exc
    chunks = _split_text(text)
    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")
    max_indexed_chars = CHUNK_SIZE + (MAX_CHUNKS_PER_DOCUMENT - 1) * (CHUNK_SIZE - CHUNK_OVERLAP)
    if len(chunks) >= MAX_CHUNKS_PER_DOCUMENT and len(text) > max_indexed_chars:
        raise ValueError("PDF is too large to index safely (maximum 500 chunks).")
    try:
        vectors = _embed(chunks, input_type="search_document")
    except RuntimeError:
        # Preserve actionable configuration errors, especially a missing API key.
        raise
    except Exception as exc:
        raise RuntimeError("The embedding service could not index this PDF.") from exc
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding service returned an incomplete vector set.")

    document_id = uuid.uuid4().hex
    prefix = _prefix(student_id)
    metadata_key = _doc_key(student_id, document_id)
    stored_keys: list[str] = []
    try:
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            key = _chunk_key(student_id, document_id, index)
            redis.hset(key, mapping={
                "document_id": document_id,
                "student_id": student_id,
                "chunk_index": index,
                "text": chunk,
                "embedding": json.dumps(vector, separators=(",", ":")),
            })
            stored_keys.append(key)
        import time
        redis.hset(metadata_key, mapping={
            "document_id": document_id,
            "student_id": student_id,
            "filename": filename.replace("\\", "/").split("/")[-1][:180],
            "pages": pages,
            "chunks": len(chunks),
            "uploaded_at": time.time(),
            "sha256": hashlib.sha256(file_bytes).hexdigest(),
        })
        redis.zadd(f"{prefix}:index", {document_id: time.time()})
    except Exception:
        if stored_keys:
            redis.delete(*stored_keys)
        redis.delete(metadata_key)
        redis.zrem(f"{prefix}:index", document_id)
        raise
    return {
        "document_id": document_id,
        "filename": filename.replace("\\", "/").split("/")[-1][:180],
        "pages": pages,
        "chunks": len(chunks),
    }


def delete_document(redis: Any, student_id: str, document_id: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{32}", document_id or ""):
        return False
    prefix = _prefix(student_id)
    metadata_key = _doc_key(student_id, document_id)
    if not redis.exists(metadata_key):
        return False
    # Ownership is encoded into the key and checked in the stored metadata too.
    owner = redis.hget(metadata_key, "student_id")
    if isinstance(owner, bytes):
        owner = owner.decode()
    if owner != student_id:
        return False
    raw_count = redis.hget(metadata_key, "chunks")
    if isinstance(raw_count, bytes):
        raw_count = raw_count.decode()
    count = int(raw_count or 0)
    keys = [_chunk_key(student_id, document_id, i) for i in range(count)]
    keys.append(metadata_key)
    redis.delete(*keys)
    redis.zrem(f"{prefix}:index", document_id)
    return True


def search_documents(redis: Any, student_id: str, query: str, limit: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    prefix = _prefix(student_id)
    document_ids = redis.zrange(f"{prefix}:index", 0, -1)
    if not document_ids:
        return []
    vectors = _embed([query], input_type="search_query")
    query_vector = vectors[0]
    ranked: list[dict[str, Any]] = []
    for raw_id in document_ids:
        document_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
        metadata = redis.hgetall(_doc_key(student_id, document_id))
        if not metadata:
            continue
        count_value = metadata.get("chunks", metadata.get(b"chunks", 0))
        count = int(count_value.decode() if isinstance(count_value, bytes) else count_value)
        filename_value = metadata.get("filename", metadata.get(b"filename", "document.pdf"))
        filename = filename_value.decode() if isinstance(filename_value, bytes) else str(filename_value)
        for index in range(count):
            raw_chunk = redis.hgetall(_chunk_key(student_id, document_id, index))
            if not raw_chunk:
                continue
            def field(name: str):
                value = raw_chunk.get(name, raw_chunk.get(name.encode(), ""))
                return value.decode() if isinstance(value, bytes) else value
            # Defence in depth: never trust a key pattern alone for tenant isolation.
            if field("student_id") != student_id:
                continue
            try:
                vector = json.loads(field("embedding"))
                if len(vector) != len(query_vector):
                    continue
                score = sum(float(a) * float(b) for a, b in zip(query_vector, vector))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            ranked.append({
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "text": field("text"),
                "score": score,
            })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:max(1, min(int(limit), 10))]
