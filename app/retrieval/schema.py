from typing import Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    retriever: str
    rank: int

    score: Optional[float] = None

    source_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    source_type: Optional[str] = None
    chunk_index: Optional[int] = None


class FusedEvidence(BaseModel):
    chunk_id: str
    text: str

    source_id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    source_type: Optional[str] = None
    chunk_index: Optional[int] = None

    retrievers: list[str] = Field(default_factory=list)

    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None

    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None

    rrf_score: float = 0.0