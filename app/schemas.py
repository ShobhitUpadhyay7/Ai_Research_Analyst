from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

from app.retrieval.schema import FusedEvidence


SourceType = Literal["internal", "web", "tech_doc"]


class IngestUrlRequest(BaseModel):
    url: HttpUrl
    source_type: SourceType = "web"


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=1000)
    text: str = Field(min_length=1)
    url: Optional[str] = None
    source_type: SourceType = "internal"


class IngestResponse(BaseModel):
    source_id: str
    title: Optional[str]
    url: Optional[str]
    source_type: str
    chunks_count: int
    status: str


class IngestStats(BaseModel):
    sources: int
    chunks: int
    chroma_vectors: int

class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_k: int = Field(default=10, ge=1, le=50)


class RetrieveResponse(BaseModel):
    query: str
    count: int
    results: list[FusedEvidence]