from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agent.llm import get_chat_model
from app.generation.schema import EvidenceItem
from app.verification.schema import Conflict


class ConflictOutput(BaseModel):
    conflicts: list[Conflict] = Field(default_factory=list)


CONFLICT_SYSTEM_PROMPT = """
You are a conflict detection engine for an AI research analyst system.

You will receive:
- a user query
- multiple evidence passages with citation keys

Your job:
- identify only true contradictions or material disagreements between sources
- do not report differences in focus or wording as conflicts
- if there are no conflicts, return an empty list
- citation_keys must refer to evidence passage keys like S1, S2, etc.
- severity must be low, medium, or high
"""


conflict_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONFLICT_SYSTEM_PROMPT),
        (
            "human",
            "User Query:\n{query}\n\n"
            "Evidence:\n{evidence}",
        ),
    ]
)


def detect_conflicts(
    query: str,
    evidence: list[EvidenceItem],
) -> list[Conflict]:
    """
    Detects conflicts between evidence passages.
    """
    if len(evidence) < 2:
        return []

    evidence_block_parts = []

    for item in evidence:
        evidence_block_parts.append(
            f"[{item.citation_key}] Title: {item.title or 'Untitled'}\n"
            f"Source Type: {item.source_type}\n"
            f"Evidence:\n{item.text[:1000]}"
        )

    evidence_block = "\n\n".join(evidence_block_parts)

    try:
        llm = get_chat_model()
        chain = conflict_prompt | llm.with_structured_output(ConflictOutput)

        output = chain.invoke(
            {
                "query": query,
                "evidence": evidence_block,
            }
        )

        return output.conflicts
    except Exception:
        return []