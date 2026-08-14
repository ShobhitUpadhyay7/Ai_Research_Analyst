from app.retrieval.rrf import rrf_fuse
from app.retrieval.schema import RetrievedChunk


def test_rrf_prefers_chunk_found_by_both_retrievers():
    bm25_results = [
        RetrievedChunk(
            chunk_id="a",
            text="Hybrid search combines BM25 and vector search.",
            retriever="bm25",
            rank=1,
            score=2.5,
        ),
        RetrievedChunk(
            chunk_id="b",
            text="BM25 is lexical retrieval.",
            retriever="bm25",
            rank=2,
            score=1.8,
        ),
    ]

    vector_results = [
        RetrievedChunk(
            chunk_id="a",
            text="Hybrid search combines BM25 and vector search.",
            retriever="vector",
            rank=1,
        ),
        RetrievedChunk(
            chunk_id="c",
            text="Vector search uses embeddings.",
            retriever="vector",
            rank=2,
        ),
    ]

    fused = rrf_fuse([bm25_results, vector_results])

    assert len(fused) > 0
    assert fused[0].chunk_id == "a"
    assert "bm25" in fused[0].retrievers
    assert "vector" in fused[0].retrievers
    assert fused[0].rrf_score > fused[1].rrf_score