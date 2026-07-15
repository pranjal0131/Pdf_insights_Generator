"""FastAPI application entrypoint.

Run locally:
    uvicorn backend.main:app --reload
"""
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import analysis, documents, health
from backend.core.config import get_settings
from backend.core.exceptions import AppError
from backend.core.logging import setup_logging

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "REST API for AI-powered financial report analysis: PDF ingestion, "
            "LLM analysis pipelines, and retrieval-augmented Q&A."
        ),
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        logger.warning("%s: %s", type(exc).__name__, exc.message)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rid=%s %s %s -> %d (%.1f ms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(analysis.router, prefix=API_PREFIX)
    return app


app = create_app()
