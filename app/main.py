import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.api.retrieve import router as retrieve_router
from app.api.research import router as research_router
from app.config import settings
from app.db import SessionLocal, init_db
from app.ingest.service import reindex_chunks

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        db = SessionLocal()
        reindexed = reindex_chunks(db)
        if reindexed > 0:
            logger.info("Re-indexed %d missing chunks into ChromaDB at startup.", reindexed)
        db.close()
    except Exception as error:
        logger.warning("Startup vector re-index skipped/failed: %s", error)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.3.0",
        description="AI Research Analyst API",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(retrieve_router)
    app.include_router(research_router)

    @app.get("/", tags=["root"])
    def root() -> dict:
        return {
            "service": settings.app_name,
            "docs": "/docs",
            "health": "/health",
            "ingest_stats": "/ingest/stats",
            "retrieve": "/retrieve",
            "research_plan": "/research/plan",
            "research_search": "/research/search",
            "research_report": "/research/report",
        }

    return app


app = create_app()