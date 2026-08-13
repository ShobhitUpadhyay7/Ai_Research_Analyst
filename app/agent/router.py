from langchain_core.prompts import ChatPromptTemplate

from app.agent.llm import get_chat_model
from app.agent.schemas import RoutePlan


DEFAULT_ROUTES = ["internal_kb", "web", "tech_docs"]


ROUTER_SYSTEM_PROMPT = """
You are a query router for an AI research analyst system.

Your job is to decide which retrieval tools should be used for a user query.

Available routes:

1. internal_kb
   - Searches the application's internal knowledge base.
   - Use this as the default route for research, technical, analytical,
     comparison, and knowledge-based queries.
   - It searches already ingested documents and internal notes.

2. web
   - Use for public information, recent events, comparisons, opinions,
     current information, and external articles.

3. tech_docs
   - Use for official technical documentation, libraries, frameworks,
     databases, APIs, SDKs, and engineering documentation.

Routing rules:

- internal_kb should normally be included for research, technical,
  analytical, and comparison queries.
- Use web when the query benefits from current, public, or external
  information.
- Use tech_docs when the query involves technical technologies,
  libraries, frameworks, databases, APIs, or engineering concepts.
- Multiple routes can be selected for the same query.
- For broad technical research or comparison questions, prefer:
  internal_kb + web + tech_docs.
- Do not exclude internal_kb simply because web or tech_docs are also
  relevant.
- If unsure, include all relevant routes.
- Keep the reason short and clear.
"""


router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "{query}"),
    ]
)


def route_query(query: str) -> RoutePlan:
    """
    Uses LLM to decide which tools should handle the query.
    Falls back to all routes if LLM fails.
    """
    try:
        llm = get_chat_model()
        chain = router_prompt | llm.with_structured_output(RoutePlan)

        plan = chain.invoke({"query": query})

        if not plan.routes:
            plan.routes = DEFAULT_ROUTES

        return plan

    except Exception:
        return RoutePlan(
            routes=DEFAULT_ROUTES,
            reason="Fallback routing because LLM was unavailable or returned invalid output.",
        )