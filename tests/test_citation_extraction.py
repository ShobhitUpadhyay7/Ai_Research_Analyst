from app.verification.citations import extract_claims


def test_extract_claims_finds_citations():
    report = """
# Research Report

## Key Findings

Hybrid search improves retrieval quality [S1].
It combines BM25 and vector search [S1][S2].

## Citations

[S1] Hybrid Search Overview — No URL
[S2] Vector Database Notes — No URL
"""

    claims = extract_claims(report)

    assert len(claims) >= 2

    first_claim_keys = claims[0].citation_keys

    assert "S1" in first_claim_keys