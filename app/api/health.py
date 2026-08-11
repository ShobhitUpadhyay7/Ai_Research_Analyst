import chromadb
from fastapi import APIRouter
from sqlalchemy import create_engine, text

from app.config import settings

router = APIRouter(tags=["health"])


def check_postgres() -> dict:
    try:
        engine = create_engine(settings.database_url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database": "postgres",
        }
    except Exception as error:
        return {
            "status": "error",
            "database": "postgres",
            "detail": str(error),
        }


def check_chroma() -> dict:
    try:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        heartbeat = client.heartbeat()

        return {
            "status": "ok",
            "vector_db": "chroma",
            "heartbeat": str(heartbeat),
        }
    except Exception as error:
        return {
            "status": "error",
            "vector_db": "chroma",
            "detail": str(error),
        }


@router.get("/health")
def health() -> dict:
    postgres_status = check_postgres()
    chroma_status = check_chroma()

    overall_status = "ok"

    if postgres_status["status"] != "ok":
        overall_status = "degraded"

    if chroma_status["status"] != "ok":
        overall_status = "degraded"

    return {
        "service": settings.app_name,
        "status": overall_status,
        "postgres": postgres_status,
        "chroma": chroma_status,
    }