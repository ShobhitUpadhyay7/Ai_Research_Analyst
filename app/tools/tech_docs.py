from sqlalchemy.orm import Session

from app.retrieval.schema import FusedEvidence
from app.retrieval.service import hybrid_search


def search_tech_docs(
    db: Session,
    query: str,
    top_k: int = 5,
    retrieval_k: int = 10,
) -> list[FusedEvidence]:
    """
    Searches technical documentation sources using hybrid retrieval.
    """
    return hybrid_search(
        db=db,
        query=query,
        top_k=top_k,
        retrieval_k=retrieval_k,
        source_types=["tech_doc"],
    )