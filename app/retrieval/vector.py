from app.ingest.vectorstore import get_vectorstore
from app.retrieval.schema import RetrievedChunk


def vector_search(
    query: str,
    k: int = 10,
    source_types: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Perform semantic vector search using ChromaDB.

    Optionally filters by source_type metadata.
    """

    vectorstore = get_vectorstore()

    where_filter = None

    if source_types:
        if len(source_types) == 1:
            where_filter = {
                "source_type": source_types[0],
            }
        else:
            where_filter = {
                "source_type": {
                    "$in": source_types,
                },
            }

    try:
        if where_filter:
            documents_with_scores = (
                vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=where_filter,
                )
            )
        else:
            documents_with_scores = (
                vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                )
            )

    except Exception:
        # Fallback to unfiltered search
        try:
            documents_with_scores = (
                vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                )
            )
        except Exception:
            documents_with_scores = []

    results: list[RetrievedChunk] = []

    for rank, (document, score) in enumerate(
        documents_with_scores,
        start=1,
    ):
        metadata = document.metadata or {}

        chunk_id = metadata.get("chunk_id")

        if not chunk_id:
            continue

        chunk_index = metadata.get("chunk_index")

        if chunk_index is not None:
            chunk_index = int(chunk_index)

        results.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                text=document.page_content,
                retriever="vector",
                rank=rank,
                score=round(float(score), 4),
                source_id=metadata.get("source_id"),
                title=metadata.get("title"),
                url=metadata.get("url"),
                source_type=metadata.get("source_type"),
                chunk_index=chunk_index,
            )
        )

    return results