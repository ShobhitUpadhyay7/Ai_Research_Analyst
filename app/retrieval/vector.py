from app.ingest.vectorstore import get_vectorstore
from app.retrieval.schema import RetrievedChunk


def vector_search(
    query: str,
    k: int = 10,
) -> list[RetrievedChunk]:
    """
    Perform semantic vector search using ChromaDB.
    """
    vectorstore = get_vectorstore()

    documents_with_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
    )

    results: list[RetrievedChunk] = []

    for rank, (document, score) in enumerate(documents_with_scores, start=1):
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