from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


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