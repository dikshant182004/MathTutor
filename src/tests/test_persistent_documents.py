"""Isolation and lifecycle tests for persistent PDF retrieval."""
from __future__ import annotations

import json

from backend.agents.nodes.tools import persistent_documents as documents


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.sorted_sets = {}

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update({str(k): str(v) for k, v in mapping.items()})

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def exists(self, key):
        return int(key in self.hashes)

    def zadd(self, key, mapping):
        self.sorted_sets.setdefault(key, {}).update(mapping)

    def zrange(self, key, start, end):
        values = list(self.sorted_sets.get(key, {}).keys())
        return values[start:] if end == -1 else values[start:end + 1]

    def zrevrange(self, key, start, end):
        values = list(self.sorted_sets.get(key, {}).keys())[::-1]
        return values[start:] if end == -1 else values[start:end + 1]

    def zrem(self, key, member):
        self.sorted_sets.get(key, {}).pop(member, None)

    def delete(self, *keys):
        for key in keys:
            self.hashes.pop(key, None)
            self.sorted_sets.pop(key, None)


def _seed(redis, student_id, document_id, text, vector):
    redis.hset(
        f"student-doc:{student_id}:{document_id}:meta",
        {"student_id": student_id, "filename": "notes.pdf", "chunks": 1, "pages": 1},
    )
    redis.zadd(f"student-doc:{student_id}:index", {document_id: 1})
    redis.hset(
        f"student-doc:{student_id}:{document_id}:chunk:0",
        {
            "student_id": student_id,
            "document_id": document_id,
            "chunk_index": 0,
            "text": text,
            "embedding": json.dumps(vector),
        },
    )


def test_search_is_limited_to_authenticated_students(monkeypatch):
    redis = FakeRedis()
    student_a = "a" * 16
    student_b = "b" * 16
    _seed(redis, student_a, "1" * 32, "Student A's algebra notes", [1.0, 0.0])
    _seed(redis, student_b, "2" * 32, "Student B's private notes", [1.0, 0.0])
    monkeypatch.setattr(documents, "_embed", lambda texts, input_type: [[1.0, 0.0]])

    results = documents.search_documents(redis, student_a, "algebra")

    assert len(results) == 1
    assert results[0]["text"] == "Student A's algebra notes"
    assert all("Student B" not in item["text"] for item in results)


def test_delete_document_removes_only_owned_document():
    redis = FakeRedis()
    student_a = "a" * 16
    student_b = "b" * 16
    doc_a = "1" * 32
    doc_b = "2" * 32
    _seed(redis, student_a, doc_a, "A", [1.0, 0.0])
    _seed(redis, student_b, doc_b, "B", [1.0, 0.0])

    assert documents.delete_document(redis, student_a, doc_b) is False
    assert documents.delete_document(redis, student_a, doc_a) is True
    assert redis.exists(f"student-doc:{student_b}:{doc_b}:meta") == 1
    assert redis.exists(f"student-doc:{student_a}:{doc_a}:meta") == 0


def test_list_documents_returns_only_the_student_index():
    redis = FakeRedis()
    student_a = "a" * 16
    student_b = "b" * 16
    _seed(redis, student_a, "1" * 32, "A", [1.0, 0.0])
    _seed(redis, student_b, "2" * 32, "B", [1.0, 0.0])

    result = documents.list_documents(redis, student_a)

    assert [item["document_id"] for item in result] == ["1" * 32]
    assert result[0]["filename"] == "notes.pdf"



def test_ingestion_persists_chunks_and_returns_relevant_passages(monkeypatch):
    redis = FakeRedis()
    student_id = "c" * 16
    monkeypatch.setattr(documents, "_read_pdf", lambda _data: (2, "Integration by parts formula and worked example."))
    monkeypatch.setattr(
        documents,
        "_embed",
        lambda texts, input_type: [[1.0, 0.0] if input_type == "search_query" or "integration" in text.lower() else [0.0, 1.0] for text in texts],
    )

    saved = documents.ingest_pdf(redis, student_id, "calculus.pdf", b"%PDF-test")
    results = documents.search_documents(redis, student_id, "integration by parts")

    assert saved["document_id"]
    assert saved["chunks"] >= 1
    assert results
    assert results[0]["filename"] == "calculus.pdf"
    assert "Integration by parts" in results[0]["text"]
    assert documents.list_documents(redis, student_id)[0]["document_id"] == saved["document_id"]
