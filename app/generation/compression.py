import hashlib

from app.generation.schema import EvidenceItem
from app.generation.tokens import count_tokens, truncate_text_to_tokens


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def compress_context(
    query: str,
    evidence: list[EvidenceItem],
    max_evidence: int = 8,
    token_budget: int = 6000,
) -> list[EvidenceItem]:
    """
    Compresses evidence by:
    - removing duplicates
    - limiting number of evidence items
    - enforcing token budget
    - truncating final evidence if needed
    """
    deduped: list[EvidenceItem] = []
    seen_hashes: set[str] = set()

    for item in evidence:
        if not item.text:
            continue

        normalized = _normalize_text(item.text)
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if text_hash in seen_hashes:
            continue

        seen_hashes.add(text_hash)
        deduped.append(item)

    selected: list[EvidenceItem] = []
    used_tokens = 0

    for item in deduped:
        if len(selected) >= max_evidence:
            break

        title = item.title or ""
        title_tokens = count_tokens(title)
        text_tokens = count_tokens(item.text)

        # Extra overhead for labels, citation key, source metadata, etc.
        overhead_tokens = 30

        total_tokens = title_tokens + text_tokens + overhead_tokens

        if used_tokens + total_tokens <= token_budget:
            selected.append(item)
            used_tokens += total_tokens
            continue

        remaining_tokens = (
            token_budget
            - used_tokens
            - title_tokens
            - overhead_tokens
        )

        if remaining_tokens > 80:
            shortened_text = truncate_text_to_tokens(
                item.text,
                remaining_tokens,
            )

            shortened_item = item.model_copy(
                update={
                    "text": shortened_text,
                }
            )

            selected.append(shortened_item)
            used_tokens += (
                title_tokens
                + count_tokens(shortened_text)
                + overhead_tokens
            )

        break

    return selected[:max_evidence]