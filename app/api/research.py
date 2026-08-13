from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.router import route_query
from app.agent.schemas import RoutePlan, TransformedQuery
from app.agent.transformation import transform_query
from app.config import settings
from app.db import get_db
from app.retrieval.schema import FusedEvidence
from app.tools.internal_kb import search_internal_kb
from app.tools.schema import WebSearchResult
from app.tools.tech_docs import search_tech_docs
from app.tools.web_search import search_web


router = APIRouter(
    prefix="/research",
    tags=["research"],
)


class ResearchPlanRequest(BaseModel):
    query: str = Field(min_length=1)


class ResearchPlanResponse(BaseModel):
    query: str
    route_plan: RoutePlan
    transformed_query: TransformedQuery


class ResearchSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_k: int = Field(default=10, ge=1, le=50)


class ResearchSearchResponse(BaseModel):
    query: str
    search_query: str
    route_plan: RoutePlan
    transformed_query: TransformedQuery

    internal_results: list[FusedEvidence]
    tech_docs_results: list[FusedEvidence]
    web_results: list[WebSearchResult]


@router.post("/plan", response_model=ResearchPlanResponse)
def research_plan(payload: ResearchPlanRequest):
    """
    Returns routing and query transformation only.
    Does not perform retrieval.
    """
    route_plan = route_query(payload.query)
    transformed_query = transform_query(payload.query)

    return ResearchPlanResponse(
        query=payload.query,
        route_plan=route_plan,
        transformed_query=transformed_query,
    )


@router.post("/search", response_model=ResearchSearchResponse)
def research_search(
    payload: ResearchSearchRequest,
    db: Session = Depends(get_db),
):
    """
    Runs routing, query transformation, and selected tools.
    """
    route_plan = route_query(payload.query)
    transformed_query = transform_query(payload.query)

    search_query = transformed_query.rewritten_query or payload.query

    internal_results: list[FusedEvidence] = []
    tech_docs_results: list[FusedEvidence] = []
    web_results: list[WebSearchResult] = []

    if "internal_kb" in route_plan.routes:
        internal_results = search_internal_kb(
            db=db,
            query=search_query,
            top_k=payload.top_k,
            retrieval_k=payload.retrieval_k,
        )

    if "tech_docs" in route_plan.routes:
        tech_docs_results = search_tech_docs(
            db=db,
            query=search_query,
            top_k=payload.top_k,
            retrieval_k=payload.retrieval_k,
        )

    if "web" in route_plan.routes:
        web_results = search_web(
            query=search_query,
            max_results=settings.max_web_results,
        )

    return ResearchSearchResponse(
        query=payload.query,
        search_query=search_query,
        route_plan=route_plan,
        transformed_query=transformed_query,
        internal_results=internal_results,
        tech_docs_results=tech_docs_results,
        web_results=web_results,
    )