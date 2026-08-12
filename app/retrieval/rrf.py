from app.retrieval.schema import FusedEvidence, RetrievedChunk


RRF_K = 60


def rrf_fuse(
    result_lists: list[list[RetrievedChunk]],
    k: int = RRF_K,
) -> list[FusedEvidence]:
    """
    Fuse multiple ranked retrieval lists using Reciprocal Rank Fusion.

    RRF score:
        score(document) = sum(1 / (k + rank))
    """
    fused: dict[str, dict] = {}

    for result_list in result_lists:
        for item in result_list:
            if not item.chunk_id:
                continue

            if item.chunk_id not in fused:
                fused[item.chunk_id] = {
                    "chunk_id": item.chunk_id,
                    "text": item.text,
                    "source_id": item.source_id,
                    "title": item.title,
                    "url": item.url,
                    "source_type": item.source_type,
                    "chunk_index": item.chunk_index,
                    "retrievers": [],
                    "bm25_rank": None,
                    "vector_rank": None,
                    "bm25_score": None,
                    "vector_score": None,
                    "rrf_score": 0.0,
                }

            entry = fused[item.chunk_id]

            entry["rrf_score"] += 1.0 / (k + item.rank)

            if item.retriever not in entry["retrievers"]:
                entry["retrievers"].append(item.retriever)

            if item.retriever == "bm25":
                entry["bm25_rank"] = item.rank
                entry["bm25_score"] = item.score

            if item.retriever == "vector":
                entry["vector_rank"] = item.rank
                entry["vector_score"] = item.score

            if entry["text"] is None and item.text:
                entry["text"] = item.text

            if entry["source_id"] is None and item.source_id:
                entry["source_id"] = item.source_id

            if entry["title"] is None and item.title:
                entry["title"] = item.title

            if entry["url"] is None and item.url:
                entry["url"] = item.url

            if entry["source_type"] is None and item.source_type:
                entry["source_type"] = item.source_type

            if entry["chunk_index"] is None and item.chunk_index is not None:
                entry["chunk_index"] = item.chunk_index

    fused_results = [
        FusedEvidence(**entry)
        for entry in fused.values()
    ]

    fused_results.sort(
        key=lambda item: item.rrf_score,
        reverse=True,
    )

    return fused_results