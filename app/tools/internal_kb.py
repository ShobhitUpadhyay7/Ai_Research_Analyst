from sqlalchemy.orm import Session

from app.retrieval.schema import FusedEvidence
from app.retrieval.service import hybrid_search


def search_internal_kb(
    db: Session,
    query: str,
    top_k: int = 5,
    retrieval_k: int = 10,
) -> list[FusedEvidence]:
    """
    Searches internal knowledge base using hybrid retrieval.
    """
    return hybrid_search(
        db=db,
        query=query,
        top_k=top_k,
        retrieval_k=retrieval_k,
        source_types=["internal"],
    )