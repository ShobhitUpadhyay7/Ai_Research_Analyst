from sqlalchemy.orm import Session

from app.agent.router import route_query
from app.agent.transformation import transform_query
from app.config import settings
from app.generation.compression import compress_context
from app.generation.evidence import build_evidence_list
from app.generation.rerank import rerank_evidence
from app.generation.report import generate_report
from app.generation.schema import ReportEvidence
from app.tools.internal_kb import search_internal_kb
from app.tools.tech_docs import search_tech_docs
from app.tools.web_search import search_web
from app.verification.citations import verify_citations
from app.verification.conflicts import detect_conflicts
from app.verification.report_sections import (
    build_citation_verification_section,
    build_verified_conflicts_section,
)


def generate_research_report(
    db: Session,
    query: str,
    top_k: int = 5,
    retrieval_k: int = 10,
    max_evidence: int = 8,
    token_budget: int = 6000,
    verify: bool = True,
    detect_source_conflicts: bool = True,
    check_urls: bool = False,
) -> dict:
    """
    Full research report pipeline:

    Query
    ↓
    Router
    ↓
    Query Transformation
    ↓
    Tools
    ↓
    Evidence Collection
    ↓
    Reranking
    ↓
    Context Compression
    ↓
    Report Generation
    ↓
    Citation Verification
    ↓
    Conflict Detection
    """
    route_plan = route_query(query)
    transformed_query = transform_query(query)

    search_query = transformed_query.rewritten_query or query

    internal_results = []
    tech_docs_results = []
    web_results = []

    if "internal_kb" in route_plan.routes:
        internal_results = search_internal_kb(
            db=db,
            query=search_query,
            top_k=top_k,
            retrieval_k=retrieval_k,
        )

    if "tech_docs" in route_plan.routes:
        tech_docs_results = search_tech_docs(
            db=db,
            query=search_query,
            top_k=top_k,
            retrieval_k=retrieval_k,
        )

    if "web" in route_plan.routes:
        web_results = search_web(
            query=search_query,
            max_results=settings.max_web_results,
        )

    evidence = build_evidence_list(
        internal_results=internal_results,
        tech_docs_results=tech_docs_results,
        web_results=web_results,
    )

    reranked_evidence = rerank_evidence(
        query=search_query,
        evidence=evidence,
        top_n=max_evidence * 2,
    )

    compressed_evidence = compress_context(
        query=search_query,
        evidence=reranked_evidence,
        max_evidence=max_evidence,
        token_budget=token_budget,
    )

    for index, item in enumerate(compressed_evidence, start=1):
        item.citation_key = f"S{index}"

    report_markdown = generate_report(
        query=search_query,
        evidence=compressed_evidence,
    )

    citation_verification = []
    conflicts = []

    if verify:
        citation_verification = verify_citations(
            report_markdown=report_markdown,
            evidence=compressed_evidence,
            check_urls=check_urls,
        )

    if detect_source_conflicts:
        conflicts = detect_conflicts(
            query=search_query,
            evidence=compressed_evidence,
        )

    if verify:
        report_markdown += "\n\n" + build_citation_verification_section(
            citation_verification
        )

    if detect_source_conflicts:
        report_markdown += "\n\n" + build_verified_conflicts_section(
            conflicts
        )

    report_evidence = [
        ReportEvidence(
            citation_key=item.citation_key,
            title=item.title,
            url=item.url,
            source_type=item.source_type,
            text=item.text,
        )
        for item in compressed_evidence
    ]

    return {
        "query": query,
        "search_query": search_query,
        "route_plan": route_plan,
        "transformed_query": transformed_query,
        "report_markdown": report_markdown,
        "evidence": report_evidence,
        "citation_verification": citation_verification,
        "conflicts": conflicts,
    }