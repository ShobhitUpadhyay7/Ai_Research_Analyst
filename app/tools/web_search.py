import logging
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from app.config import settings
from app.tools.schema import WebSearchResult
from app.tools.utility import clean_snippet


logger = logging.getLogger(__name__)


try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def search_web(
    query: str,
    max_results: int | None = None,
) -> list[WebSearchResult]:
    """
    Performs web search.

    Primary:
        DuckDuckGo via ddgs

    Fallback:
        Wikipedia search API, which is more stable for demos.
    """
    max_results = max_results or settings.max_web_results

    results = _search_duckduckgo(
        query=query,
        max_results=max_results,
    )

    if results:
        return results

    logger.info(
        "DuckDuckGo returned no results. Falling back to Wikipedia search."
    )

    return _search_wikipedia(
        query=query,
        max_results=max_results,
    )


def _search_duckduckgo(
    query: str,
    max_results: int,
) -> list[WebSearchResult]:
    """
    DuckDuckGo search using the new ddgs package.
    """
    if DDGS is None:
        logger.warning("DDGS package is not installed.")
        return []

    try:
        ddgs = DDGS()

        raw_results = list(
            ddgs.text(
                query,
                max_results=max_results,
            )
        )
    except Exception:
        logger.exception("DuckDuckGo search failed")
        return []

    results: list[WebSearchResult] = []

    for item in raw_results:
        title = item.get("title")
        url = item.get("href") or item.get("url")
        snippet = item.get("body") or item.get("snippet")

        text_parts = []

        if title:
            text_parts.append(title)

        if snippet:
            text_parts.append(snippet)

        text = clean_snippet("\n".join(text_parts))

        if not text:
            continue

        results.append(
            WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                text=text,
                source_type="web",
            )
        )

    return results


def _search_wikipedia(
    query: str,
    max_results: int,
) -> list[WebSearchResult]:
    """
    Stable fallback web search using Wikipedia's public API.
    """
    try:
        response = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": max_results,
            },
            timeout=10.0,
        )

        response.raise_for_status()

        data = response.json()

        raw_results = data.get("query", {}).get("search", [])
    except Exception:
        logger.exception("Wikipedia fallback search failed")
        return []

    results: list[WebSearchResult] = []

    for item in raw_results:
        title = item.get("title")
        snippet_html = item.get("snippet", "")

        if not title:
            continue

        snippet = BeautifulSoup(
            snippet_html,
            "html.parser",
        ).get_text()

        url = (
            "https://en.wikipedia.org/wiki/"
            + quote(title.replace(" ", "_"))
        )

        text_parts = []

        if title:
            text_parts.append(title)

        if snippet:
            text_parts.append(snippet)

        text = clean_snippet("\n".join(text_parts))

        if not text:
            continue

        results.append(
            WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                text=text,
                source_type="web",
            )
        )

    return results