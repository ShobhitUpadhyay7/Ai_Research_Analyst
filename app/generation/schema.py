from typing import Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    id: str
    text: str

    title: Optional[str] = None
    url: Optional[str] = None

    source_type: str = "unknown"
    tool: str = "unknown"

    score: Optional[float] = None
    retrievers: list[str] = Field(default_factory=list)

    citation_key: Optional[str] = None


class ReportEvidence(BaseModel):
    citation_key: str

    title: Optional[str] = None
    url: Optional[str] = None

    source_type: str
    text: str