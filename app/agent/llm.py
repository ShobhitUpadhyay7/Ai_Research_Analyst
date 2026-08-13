from langchain_ollama import ChatOllama

from app.config import settings


def get_chat_model(temperature: float = 0.0):
    return ChatOllama(
        model=settings.llm_model,
        temperature=temperature,
    )