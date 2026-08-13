from typing import Literal

from pydantic import BaseModel, Field


RouteName = Literal["internal_kb", "web", "tech_docs"]


class RoutePlan(BaseModel):
    routes: list[RouteName] = Field(default_factory=list)
    reason: str = ""


class TransformedQuery(BaseModel):
    rewritten_query: str
    subqueries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)