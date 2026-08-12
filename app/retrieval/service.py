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
) -> list[FusedEvidence]:
    """
    Hybrid retrieval pipeline:

    BM25 search
    +
    Vector search
    +
    RRF fusion
    """
    bm25_results = bm25_search(
        db=db,
        query=query,
        k=retrieval_k,
    )

    try:
        vector_results = vector_search(
            query=query,
            k=retrieval_k,
        )
    except Exception as error:
        # If vector search fails, log the exception and fallback to BM25.
        logger.warning("Vector search failed, falling back to BM25 only: %s", error, exc_info=True)
        vector_results = []

    fused_results = rrf_fuse(
        result_lists=[
            bm25_results,
            vector_results,
        ]
    )

    return fused_results[:top_k]