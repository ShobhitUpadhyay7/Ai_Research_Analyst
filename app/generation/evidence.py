import hashlib

from app.generation.schema import EvidenceItem
from app.retrieval.schema import FusedEvidence
from app.tools.schema import WebSearchResult


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build_evidence_list(
    internal_results: list[FusedEvidence],
    tech_docs_results: list[FusedEvidence],
    web_results: list[WebSearchResult],
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []

    for item in internal_results:
        evidence.append(
            EvidenceItem(
                id=item.chunk_id or f"internal-{_short_hash(item.text)}",
                text=item.text,
                title=item.title,
                url=item.url,
                source_type=item.source_type or "internal",
                tool="internal_kb",
                score=item.rrf_score,
                retrievers=item.retrievers,
            )
        )

    for item in tech_docs_results:
        evidence.append(
            EvidenceItem(
                id=item.chunk_id or f"tech-{_short_hash(item.text)}",
                text=item.text,
                title=item.title,
                url=item.url,
                source_type=item.source_type or "tech_doc",
                tool="tech_docs",
                score=item.rrf_score,
                retrievers=item.retrievers,
            )
        )

    for item in web_results:
        text = item.text or item.snippet or item.title or ""

        if not text:
            continue

        evidence.append(
            EvidenceItem(
                id=f"web-{_short_hash(text)}",
                text=text,
                title=item.title,
                url=item.url,
                source_type="web",
                tool="web",
                score=None,
                retrievers=[],
            )
        )

    return evidence