import re
from typing import Literal

import httpx
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agent.llm import get_chat_model
from app.generation.schema import EvidenceItem
from app.verification.schema import CitationVerification, Claim


CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class CitationCheck(BaseModel):
    citation_key: str
    supported: bool
    confidence: Literal["high", "medium", "low"]
    reason: str


class CitationVerificationOutput(BaseModel):
    checks: list[CitationCheck] = Field(default_factory=list)


VERIFICATION_SYSTEM_PROMPT = """
You are a citation verification engine for an AI research analyst system.

You will receive:
- citation keys
- evidence passages
- claims that cite those evidence passages

Your job:
- decide whether each cited evidence passage supports the claims
- supported=true if the evidence directly or partially supports the claims
- supported=false if the evidence does not support the claims
- confidence should be high, medium, or low
- reason must be short and clear
- return checks only for citation keys present in the input
"""


verification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VERIFICATION_SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


def extract_claims(report_markdown: str) -> list[Claim]:
    """
    Extracts sentences/bullets from the report that contain citations like [S1].
    """
    claims: list[Claim] = []
    in_citations_section = False

    for line in report_markdown.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.lower().startswith("## citations"):
            in_citations_section = True
            continue

        if in_citations_section:
            continue

        # Skip headings because they usually do not contain claim sentences.
        if stripped.startswith("#"):
            continue

        # Skip citation list rows like:
        # [S1] Title — URL
        if stripped.startswith("[S") and ("—" in stripped or "http" in stripped):
            continue

        sentences = re.split(r"(?<=[.!?])\s+", stripped)

        for sentence in sentences:
            matches = CITATION_PATTERN.findall(sentence)

            if not matches:
                continue

            citation_keys = [f"S{match}" for match in matches]
            clean_sentence = CITATION_PATTERN.sub("", sentence).strip()

            if len(clean_sentence) >= 8:
                claims.append(
                    Claim(
                        text=clean_sentence,
                        citation_keys=citation_keys,
                    )
                )

    return claims


def check_url(url: str | None) -> bool | None:
    """
    Optional URL reachability check.
    Returns:
        True if reachable
        False if unreachable
        None if no URL exists
    """
    if not url:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AIResearchAnalystBot/0.1)"
    }

    try:
        response = httpx.head(
            url,
            timeout=5.0,
            follow_redirects=True,
            headers=headers,
        )

        if response.status_code >= 400:
            response = httpx.get(
                url,
                timeout=5.0,
                follow_redirects=True,
                headers=headers,
            )

        return response.status_code < 400
    except Exception:
        return False


def verify_citations(
    report_markdown: str,
    evidence: list[EvidenceItem],
    check_urls: bool = False,
) -> list[CitationVerification]:
    """
    Verifies whether cited evidence supports claims in the generated report.
    """
    if not evidence:
        return []

    evidence_map = {
        item.citation_key: item
        for item in evidence
        if item.citation_key
    }

    claims = extract_claims(report_markdown)

    citation_claims: dict[str, list[str]] = {}

    for claim in claims:
        for citation_key in claim.citation_keys:
            if citation_key in evidence_map:
                citation_claims.setdefault(citation_key, []).append(claim.text)

    results: list[CitationVerification] = []

    cited_blocks: list[str] = []

    for citation_key, item in evidence_map.items():
        claims_for_citation = citation_claims.get(citation_key, [])

        if not claims_for_citation:
            continue

        claims_text = "\n".join(
            f"- {claim}"
            for claim in claims_for_citation[:5]
        )

        cited_blocks.append(
            f"Citation Key: [{citation_key}]\n"
            f"Evidence:\n{item.text[:1200]}\n\n"
            f"Claims citing this source:\n{claims_text}"
        )

    checks_by_key: dict[str, CitationCheck] = {}

    if cited_blocks:
        input_block = "\n\n---\n\n".join(cited_blocks)

        try:
         llm = get_chat_model()
         chain = (
        verification_prompt
        | llm.with_structured_output(CitationVerificationOutput)
         )

         output = chain.invoke({"input": input_block})

         print("=== CITATION VERIFICATION OUTPUT ===")
         print(output)

         print("=== OUTPUT CHECKS ===")
         print(output.checks)

         print("=== EVIDENCE KEYS ===")
         print(list(evidence_map.keys()))

         for check in output.checks:
          citation_key = check.citation_key.strip("[]")

          print(
           f"CHECK: raw_key={check.citation_key} "
           f"normalized_key={citation_key} "
           f"supported={check.supported} "
           f"confidence={check.confidence}"
          )

          if citation_key in evidence_map:
           checks_by_key[citation_key] = check

         print("=== CHECKS BY KEY ===")
         print(checks_by_key)

        except Exception as error:
         print(f"CITATION VERIFICATION ERROR: {error}")
         raise

    for citation_key, item in evidence_map.items():
        url_reachable = None

        if check_urls:
            url_reachable = check_url(item.url)

        if citation_key not in citation_claims:
            results.append(
                CitationVerification(
                    citation_key=citation_key,
                    supported=False,
                    status="not_cited",
                    confidence="low",
                    reason="No report claim was found citing this source.",
                    url_reachable=url_reachable,
                )
            )
            continue

        check = checks_by_key.get(citation_key)

        if check:
            status = "supported" if check.supported else "unsupported"

            results.append(
                CitationVerification(
                    citation_key=citation_key,
                    supported=check.supported,
                    status=status,
                    confidence=check.confidence,
                    reason=check.reason,
                    url_reachable=url_reachable,
                )
            )
        else:
            results.append(
                CitationVerification(
                    citation_key=citation_key,
                    supported=False,
                    status="error",
                    confidence="low",
                    reason="Citation verification failed.",
                    url_reachable=url_reachable,
                )
            )

    return results