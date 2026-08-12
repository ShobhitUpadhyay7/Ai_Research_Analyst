import re
import hashlib
import uuid

import httpx
from sqlalchemy.orm import Session

from app.ingest.chunk import chunk_text
from app.ingest.extract import extract_html
from app.ingest.vectorstore import get_vectorstore
from app.models import Chunk, Source

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AIResearchAnalystBot/0.1)"
}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def clean_text(text: str) -> str:
    """
    Cleans messy text by normalizing whitespaces and newlines.
    This ensures the LLM receives clean context later.
    """
    # 1. Strip trailing/leading whitespace from every line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    
    # 2. Replace 3 or more consecutive newlines with just 2 (one empty line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # 3. Replace multiple spaces/tabs with a single space (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    
    return text.strip()


def ingest_text(
    db: Session,
    *,
    title: str,
    text: str,
    url: str | None = None,
    source_type: str = "internal",
) -> tuple[Source, int]:
    text = clean_text(text)

    if not text:
        raise ValueError("Text is empty")

    source = Source(
        title=title,
        url=url,
        source_type=source_type,
        content_hash=_hash_text(text),
        status="active",
    )

    db.add(source)
    db.flush()

    chunks = chunk_text(text)

    chunk_rows = []
    texts = []
    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())

        chunk_row = Chunk(
            id=chunk_id,
            source_id=source.id,
            text=chunk,
            chunk_index=index,
            token_count=_approx_tokens(chunk),
            chroma_id=chunk_id,
        )

        chunk_rows.append(chunk_row)

        texts.append(chunk)
        ids.append(chunk_id)

        metadatas.append(
            {
                "chunk_id": chunk_id,
                "source_id": source.id,
                "title": title or "",
                "url": url or "",
                "source_type": source_type,
                "chunk_index": index,
            }
        )

    db.add_all(chunk_rows)
    db.commit()

    try:
        if texts:
            vectorstore = get_vectorstore()
            vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )
    except Exception:
        source.status = "failed"
        db.commit()
        raise

    db.refresh(source)

    return source, len(chunk_rows)


def ingest_url(
    db: Session,
    url: str,
    source_type: str = "web",
) -> tuple[Source, int]:
    response = httpx.get(
        url,
        headers=HEADERS,
        timeout=20.0,
        follow_redirects=True,
    )

    response.raise_for_status()

    title, text = extract_html(response.text)

    if not title:
        title = url

    if not text:
        raise ValueError("No readable text extracted from URL")

    return ingest_text(
        db=db,
        title=title,
        text=text,
        url=url,
        source_type=source_type,
    )

def reindex_chunks(db: Session) -> int:
    """
    Ensure all active chunks from Postgres DB are present in ChromaDB.
    Returns the count of chunks re-indexed into ChromaDB.
    """
    from sqlalchemy.orm import joinedload

    active_chunks = (
        db.query(Chunk)
        .join(Source)
        .filter(Source.status == "active")
        .options(joinedload(Chunk.source))
        .all()
    )

    if not active_chunks:
        return 0

    vectorstore = get_vectorstore()
    existing_ids = set(vectorstore._collection.get()["ids"])

    missing_chunks = [c for c in active_chunks if c.id not in existing_ids]

    if not missing_chunks:
        return 0

    texts = []
    metadatas = []
    ids = []

    for chunk in missing_chunks:
        source = chunk.source
        texts.append(chunk.text)
        ids.append(chunk.id)
        metadatas.append(
            {
                "chunk_id": chunk.id,
                "source_id": chunk.source_id,
                "title": source.title if source else "",
                "url": source.url if source else "",
                "source_type": source.source_type if source else "internal",
                "chunk_index": chunk.chunk_index,
            }
        )

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    return len(missing_chunks)