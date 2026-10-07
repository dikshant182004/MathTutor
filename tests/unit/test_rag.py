from backend.v2.rag import RetrievedChunk, reciprocal_rank_fusion


def test_rrf_combines_ranked_results():
    a = RetrievedChunk("a", "doc", "A", 0.9, {})
    b = RetrievedChunk("b", "doc", "B", 0.8, {})
    result = reciprocal_rank_fusion([[a, b], [b, a]])
    assert {x.chunk_id for x in result} == {"a", "b"}
    assert result[0].score > 0
