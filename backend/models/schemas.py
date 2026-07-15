"""Pydantic request/response schemas for the REST API."""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class AnalysisType(StrEnum):
    summary = "summary"
    key_insights = "key_insights"
    trend_analysis = "trend_analysis"
    risk_assessment = "risk_assessment"
    recommendations = "recommendations"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    llm_model: str
    llm_configured: bool


class DocumentMetadata(BaseModel):
    total_pages: int
    title: str | None = None
    author: str | None = None
    subject: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    token_count: int
    chunk_count: int
    deduplicated: bool = Field(
        description="True if an identical document was already uploaded and reused."
    )
    metadata: DocumentMetadata


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    num_pages: int
    token_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class AnalysisRequest(BaseModel):
    analyses: list[AnalysisType] = Field(
        default=list(AnalysisType),
        min_length=1,
        description="Which analyses to run. Defaults to all.",
    )


class AnalysisResponse(BaseModel):
    document_id: str
    results: dict[AnalysisType, str]
    served_from_cache: list[AnalysisType]
    elapsed_seconds: float


class QARequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class SourceChunk(BaseModel):
    page: int | None
    snippet: str


class QAResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    sources: list[SourceChunk]


class ErrorResponse(BaseModel):
    detail: str
