import re

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session, joinedload

from app.models import Chunk
from app.retrieval.schema import RetrievedChunk


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    Converts text into lowercase word tokens.
    """
    return re.findall(r"\w+", text.lower())


def bm25_search(
    db: Session,
    query: str,
    k: int = 10,
) -> list[RetrievedChunk]:
    """
    Perform BM25 lexical search over all stored chunks.
    """
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    chunks = (
        db.query(Chunk)
        .options(joinedload(Chunk.source))
        .all()
    )

    if not chunks:
        return []

    chunk_pairs = []

    for chunk in chunks:
        tokens = tokenize(chunk.text)

        if tokens:
            chunk_pairs.append((chunk, tokens))

    if not chunk_pairs:
        return []

    corpus = [tokens for _, tokens in chunk_pairs]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:k]

    results: list[RetrievedChunk] = []

    for rank, index in enumerate(ranked_indices, start=1):
        score = float(scores[index])

        if score <= 0.0:
            continue

        chunk, _ = chunk_pairs[index]
        source = chunk.source

        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                text=chunk.text,
                retriever="bm25",
                rank=rank,
                score=score,
                source_id=chunk.source_id,
                title=source.title if source else None,
                url=source.url if source else None,
                source_type=source.source_type if source else None,
                chunk_index=chunk.chunk_index,
            )
        )

    return results