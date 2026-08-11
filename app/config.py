from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Research Analyst"
    environment: str = "development"

    database_url: str = "postgresql+psycopg://app:app@db:5432/research"

    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection: str = "research_chunks"

    ollama_host: str = "http://host.docker.internal:11434"
    llm_model: str = "qwen3:1.7b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    max_web_results: int = 5
    max_evidence: int = 12
    context_token_budget: int = 6000

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()