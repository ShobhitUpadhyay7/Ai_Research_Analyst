from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingest.service import ingest_text, ingest_url
from app.ingest.vectorstore import get_chroma_count
from app.models import Chunk, Source
from app.schemas import (
    IngestResponse,
    IngestStats,
    IngestTextRequest,
    IngestUrlRequest,
)


router = APIRouter(
    prefix="/ingest",
    tags=["ingest"],
)


@router.post("/url", response_model=IngestResponse)
def ingest_url_endpoint(
    payload: IngestUrlRequest,
    db: Session = Depends(get_db),
):
    try:
        source, chunks_count = ingest_url(
            db=db,
            url=str(payload.url),
            source_type=payload.source_type,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"URL ingestion failed: {error}",
        )

    return IngestResponse(
        source_id=source.id,
        title=source.title,
        url=source.url,
        source_type=source.source_type,
        chunks_count=chunks_count,
        status=source.status,
    )


@router.post("/text", response_model=IngestResponse)
def ingest_text_endpoint(
    payload: IngestTextRequest,
    db: Session = Depends(get_db),
):
    try:
        source, chunks_count = ingest_text(
            db=db,
            title=payload.title,
            text=payload.text,
            url=payload.url,
            source_type=payload.source_type,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Text ingestion failed: {error}",
        )

    return IngestResponse(
        source_id=source.id,
        title=source.title,
        url=source.url,
        source_type=source.source_type,
        chunks_count=chunks_count,
        status=source.status,
    )


@router.get("/stats", response_model=IngestStats)
def ingest_stats(db: Session = Depends(get_db)):
    sources_count = db.query(func.count(Source.id)).scalar() or 0
    chunks_count = db.query(func.count(Chunk.id)).scalar() or 0

    try:
        chroma_vectors = get_chroma_count()
    except Exception:
        chroma_vectors = -1

    return IngestStats(
        sources=sources_count,
        chunks=chunks_count,
        chroma_vectors=chroma_vectors,
    )