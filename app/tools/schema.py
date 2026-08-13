from typing import Optional

from pydantic import BaseModel


class WebSearchResult(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    text: str
    source_type: str = "web"