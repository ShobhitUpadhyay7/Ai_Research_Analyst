from langchain_core.prompts import ChatPromptTemplate

from app.agent.llm import get_chat_model
from app.agent.schemas import TransformedQuery


TRANSFORMATION_SYSTEM_PROMPT = """
You are a query transformation engine for an AI research analyst system.

Your job is to rewrite the user query into a better search query.

You must produce:
1. rewritten_query
   - A clearer, search-friendly version of the original query.
2. subqueries
   - Up to 3 focused subqueries that help retrieve better evidence.
3. keywords
   - Up to 8 important keywords.

Rules:
- Do not answer the question.
- Do not invent facts.
- Keep the output concise.
- Focus on retrieval quality.
"""


transformation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TRANSFORMATION_SYSTEM_PROMPT),
        ("human", "{query}"),
    ]
)


def transform_query(query: str) -> TransformedQuery:
    """
    Uses LLM to rewrite the query and generate subqueries/keywords.
    Falls back to the original query if LLM fails.
    """
    try:
        llm = get_chat_model()
        chain = transformation_prompt | llm.with_structured_output(TransformedQuery)

        transformed = chain.invoke({"query": query})

        if not transformed.rewritten_query:
            transformed.rewritten_query = query

        if not transformed.subqueries:
            transformed.subqueries = [query]

        return transformed

    except Exception:
        return TransformedQuery(
            rewritten_query=query,
            subqueries=[query],
            keywords=[],
        )