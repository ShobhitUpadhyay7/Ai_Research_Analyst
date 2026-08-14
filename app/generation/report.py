import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.llm import get_chat_model
from app.generation.schema import EvidenceItem


REPORT_SYSTEM_PROMPT = """
You are an AI Research Analyst.

Your job is to write a grounded research report using ONLY the provided evidence.

Rules:
1. Use only the provided evidence.
2. Cite every factual claim using citation keys like [S1], [S2], etc.
3. Do not invent facts.
4. Do not invent citations.
5. If evidence conflicts, clearly mention the conflict.
6. If evidence is insufficient, say so clearly.
7. Do not write a Citations section. Citations will be appended automatically.
8. Write in Markdown.
9. Keep the report concise and professional.

Use this structure:

# Research Report

## Executive Summary

## Key Findings

## Evidence Comparison

## Conflicts

## Recommendation
"""


report_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REPORT_SYSTEM_PROMPT),
        (
            "human",
            "User Query:\n{query}\n\n"
            "Evidence:\n{evidence}",
        ),
    ]
)


def _remove_citations_section(markdown: str) -> str:
    """
    Removes a generated Citations section if the LLM accidentally adds one.
    """
    return re.split(
        r"\n##\s*Citations\s*\n",
        markdown,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].rstrip()


def build_citations_section(evidence: list[EvidenceItem]) -> str:
    lines = [
        "## Citations",
        "",
    ]

    for item in evidence:
        title = item.title or "Untitled"
        url = item.url or "No URL"

        lines.append(
            f"[{item.citation_key}] {title} — {url} "
            f"(`{item.source_type}` via `{item.tool}`)"
        )

    return "\n".join(lines)


def generate_report(
    query: str,
    evidence: list[EvidenceItem],
) -> str:
    """
    Generates a Markdown research report from evidence.
    """
    if not evidence:
        return (
            "# Research Report\n\n"
            "## Executive Summary\n\n"
            "Insufficient evidence was retrieved to answer this query.\n\n"
            "## Recommendation\n\n"
            "Try ingesting more internal documents, adding technical documentation, "
            "or enabling web search.\n"
        )

    evidence_block_parts = []

    for item in evidence:
        evidence_block_parts.append(
            f"[{item.citation_key}] Title: {item.title or 'Untitled'}\n"
            f"Source Type: {item.source_type}\n"
            f"URL: {item.url or 'No URL'}\n"
            f"Evidence:\n{item.text}"
        )

    evidence_block = "\n\n".join(evidence_block_parts)

    chain = (
        report_prompt
        | get_chat_model(temperature=0.2)
        | StrOutputParser()
    )

    report = chain.invoke(
        {
            "query": query,
            "evidence": evidence_block,
        }
    )

    report = _remove_citations_section(report)

    citations = build_citations_section(evidence)

    return report.rstrip() + "\n\n" + citations