from app.ingest.chunk import chunk_text


def test_chunk_text_returns_multiple_chunks_for_long_text():
    text = (
        "AI Research Analyst uses hybrid retrieval. "
        "BM25 handles lexical search. "
        "Vector search handles semantic search. "
        "RRF combines the results. "
    ) * 100

    chunks = chunk_text(text)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)