from typing import Literal, Optional

from pydantic import BaseModel, Field


Confidence = Literal["high", "medium", "low"]
Severity = Literal["low", "medium", "high"]


class Claim(BaseModel):
    text: str
    citation_keys: list[str] = Field(default_factory=list)


class CitationVerification(BaseModel):
    citation_key: str
    supported: bool

    # supported, unsupported, not_cited, error
    status: str

    confidence: Confidence = "low"
    reason: str = ""

    url_reachable: Optional[bool] = None


class Conflict(BaseModel):
    description: str
    citation_keys: list[str] = Field(default_factory=list)
    severity: Severity = "medium"