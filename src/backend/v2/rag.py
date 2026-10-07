from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict


class Retriever(Protocol):
    def retrieve(self, query: str, *, student_id: str, top_k: int = 5) -> list[RetrievedChunk]: ...


def reciprocal_rank_fusion(result_lists: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    items: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            items[item.chunk_id] = item
    return sorted(
        (RetrievedChunk(i.chunk_id, i.document_id, i.text, scores[i.chunk_id], i.metadata)
         for i in items.values()),
        key=lambda x: x.score,
        reverse=True,
    )


def build_citation(chunk: RetrievedChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "score": round(chunk.score, 6),
        "source": chunk.metadata.get("source"),
        "page": chunk.metadata.get("page"),
    }
