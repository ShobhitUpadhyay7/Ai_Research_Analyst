from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.retrieval.service import hybrid_search
from app.schemas import RetrieveRequest, RetrieveResponse


router = APIRouter(
    tags=["retrieve"],
)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(
    payload: RetrieveRequest,
    db: Session = Depends(get_db),
):
    try:
        results = hybrid_search(
            db=db,
            query=payload.query,
            top_k=payload.top_k,
            retrieval_k=payload.retrieval_k,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Retrieval failed: {error}",
        )

    return RetrieveResponse(
        query=payload.query,
        count=len(results),
        results=results,
    )