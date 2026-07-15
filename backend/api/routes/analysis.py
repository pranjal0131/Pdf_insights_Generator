"""Analysis and retrieval-augmented Q&A endpoints."""
import logging
import time

from fastapi import APIRouter

from backend.api.dependencies import AnalysisServiceDep, DocumentStoreDep, EmbeddingsDep
from backend.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    QARequest,
    QAResponse,
    SourceChunk,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents/{document_id}", tags=["analysis"])

_SNIPPET_CHARS = 300


@router.post(
    "/analysis",
    response_model=AnalysisResponse,
    summary="Run AI analyses on a document",
)
async def analyze_document(
    document_id: str,
    request: AnalysisRequest,
    store: DocumentStoreDep,
    service: AnalysisServiceDep,
) -> AnalysisResponse:
    record = store.get(document_id)
    analyses = [a.value for a in request.analyses]

    start = time.perf_counter()
    results, cached = await service.analyze(record, analyses)
    elapsed = time.perf_counter() - start

    logger.info(
        "Analysis for %s: %s (%d cached) in %.2fs",
        document_id,
        analyses,
        len(cached),
        elapsed,
    )
    return AnalysisResponse(
        document_id=document_id,
        results=results,
        served_from_cache=cached,
        elapsed_seconds=round(elapsed, 2),
    )


@router.post(
    "/qa",
    response_model=QAResponse,
    summary="Ask a question about a document (RAG)",
)
async def ask_question(
    document_id: str,
    request: QARequest,
    store: DocumentStoreDep,
    service: AnalysisServiceDep,
    embeddings: EmbeddingsDep,
) -> QAResponse:
    record = store.get(document_id)
    answer, sources = await service.answer_question(record, request.question, embeddings)

    return QAResponse(
        document_id=document_id,
        question=request.question,
        answer=answer,
        sources=[
            SourceChunk(
                page=doc.metadata.get("page"),
                snippet=doc.page_content[:_SNIPPET_CHARS],
            )
            for doc in sources
        ],
    )
