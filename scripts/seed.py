import textwrap

from app.db import SessionLocal, init_db
from app.ingest.service import ingest_text, reindex_chunks
from app.models import Source

INTERNAL_DOCS = [
    (
        "Hybrid Search Overview",
        textwrap.dedent("""\
            Hybrid search combines lexical retrieval and semantic retrieval.
            BM25 is useful for exact keyword matching, product names, error codes,
            and rare terms. Vector search is useful for semantic similarity and
            paraphrased queries.

            In our AI Research Analyst system, we use BM25 and vector search together.
            The results are fused using Reciprocal Rank Fusion, also known as RRF.
            RRF is useful because it combines ranked lists without requiring raw
            score normalization.
        """),
    ),
    (
        "Research Report Guidelines",
        textwrap.dedent("""\
            A research report should contain an executive summary, key findings,
            evidence comparison, conflicts, recommendation, and citations.

            Every factual claim should cite at least one source.
            If sources disagree, the report should identify the conflict and
            explain confidence level.

            Citation verification should check whether the cited evidence actually
            supports the generated claim.
        """),
    ),
    (
        "Vector Database Notes",
        textwrap.dedent("""\
            ChromaDB is an easy-to-use vector database for prototyping RAG systems.
            It integrates well with LangChain and can be run using Docker.

            For production scale, alternatives include pgvector, Qdrant,
            Weaviate, OpenSearch, and Elasticsearch.
        """),
    ),
]



def main():
    init_db()
    db = SessionLocal()

    try:
        for title, text in INTERNAL_DOCS:
            existing = (
                db.query(Source)
                .filter(Source.title == title, Source.status == "active")
                .first()
            )

            if existing:
                print(f"Skipping '{title}' | source_id={existing.id} already exists.")
                continue

            source, chunks_count = ingest_text(
                db=db,
                title=title,
                text=text,
                source_type="internal",
            )

            print(
                f"Ingested '{title}' | "
                f"source_id={source.id} | "
                f"chunks={chunks_count}"
            )

        reindexed_count = reindex_chunks(db)
        if reindexed_count > 0:
            print(f"Re-indexed {reindexed_count} missing chunk(s) into ChromaDB.")

    finally:
        db.close()


if __name__ == "__main__":
    main()