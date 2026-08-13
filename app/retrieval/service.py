import logging

from sqlalchemy.orm import Session

from app.retrieval.bm25 import bm25_search
from app.retrieval.rrf import rrf_fuse
from app.retrieval.schema import FusedEvidence
from app.retrieval.vector import vector_search

logger = logging.getLogger(__name__)


def hybrid_search(
    db: Session,
    query: str,
    top_k: int = 5,
    retrieval_k: int = 10,
    source_types: list[str] | None = None,
) -> list[FusedEvidence]:
    """
    Hybrid retrieval pipeline:

    BM25 search
    +
    Vector search
    +
    RRF fusion

    Optionally filters results by source_types.
    """

    source_types = source_types or None

    # 1. BM25 retrieval
    bm25_results = bm25_search(
        db=db,
        query=query,
        k=retrieval_k,
        source_types=source_types,
    )

    # 2. Vector retrieval
    try:
        vector_results = vector_search(
            query=query,
            k=retrieval_k,
            source_types=source_types,
        )

    except Exception as error:
        logger.warning(
            "Vector search failed, falling back to BM25 only: %s",
            error,
            exc_info=True,
        )

        vector_results = []

    # 3. Reciprocal Rank Fusion
    fused_results = rrf_fuse(
        result_lists=[
            bm25_results,
            vector_results,
        ]
    )

    # 4. Return top results
    return fused_results[:top_k]