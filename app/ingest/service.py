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


def ingest_text(
    db: Session,
    *,
    title: str,
    text: str,
    url: str | None = None,
    source_type: str = "internal",
) -> tuple[Source, int]:
    text = text.strip()

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