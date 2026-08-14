from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.agent.llm import get_chat_model
from app.generation.schema import EvidenceItem


class RerankOutput(BaseModel):
    ranked_indices: list[int] = Field(default_factory=list)


RERANK_SYSTEM_PROMPT = """
You are a strict relevance reranker for an AI research analyst system.

You will receive:
- a user query
- a numbered list of evidence passages

Your job:
- return the indices of the most relevant passages
- sort them from most relevant to least relevant
- only include indices that are actually relevant
- do not invent new indices
- do not explain your answer

Example output:
{
  "ranked_indices": [3, 1, 5]
}
"""


rerank_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RERANK_SYSTEM_PROMPT),
        (
            "human",
            "Query:\n{query}\n\n"
            "Return the top {top_n} most relevant passage indices.\n\n"
            "Evidence:\n{evidence}",
        ),
    ]
)


def _fallback_rank(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    """
    Simple fallback ranking if LLM reranking fails.
    """
    def score_item(item: EvidenceItem) -> float:
        score = item.score or 0.0

        if item.source_type == "internal":
            score += 0.02

        if item.source_type == "tech_doc":
            score += 0.015

        if item.source_type == "web":
            score += 0.005

        if len(item.retrievers) > 1:
            score += 0.02

        return score

    return sorted(
        evidence,
        key=score_item,
        reverse=True,
    )


def rerank_evidence(
    query: str,
    evidence: list[EvidenceItem],
    top_n: int = 8,
) -> list[EvidenceItem]:
    """
    Reranks evidence using LLM.
    Falls back to simple ranking if LLM fails.
    """
    if not evidence:
        return []

    candidates = evidence[:15]

    evidence_block_parts = []

    for index, item in enumerate(candidates, start=1):
        title = item.title or "Untitled"
        text = item.text[:1200]

        evidence_block_parts.append(
            f"[{index}] Title: {title}\n"
            f"Source Type: {item.source_type}\n"
            f"Tool: {item.tool}\n"
            f"Evidence:\n{text}"
        )

    evidence_block = "\n\n".join(evidence_block_parts)

    try:
        llm = get_chat_model()
        chain = rerank_prompt | llm.with_structured_output(RerankOutput)

        output = chain.invoke(
            {
                "query": query,
                "top_n": top_n,
                "evidence": evidence_block,
            }
        )

        valid_indices = []
        seen = set()

        for idx in output.ranked_indices:
            if 1 <= idx <= len(candidates) and idx not in seen:
                valid_indices.append(idx)
                seen.add(idx)

        ranked = [
            candidates[idx - 1]
            for idx in valid_indices
        ]

        if ranked:
            return ranked[:top_n]

    except Exception:
        pass

    return _fallback_rank(candidates)[:top_n]