"""Document upload and lifecycle endpoints."""
import logging

from fastapi import APIRouter, UploadFile, status

from backend.api.dependencies import DocumentStoreDep, SettingsDep
from backend.core.exceptions import FileTooLargeError, InvalidFileTypeError
from backend.models.schemas import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentSummary,
    DocumentUploadResponse,
)
from backend.services.pdf_service import parse_pdf
from backend.services.text_service import chunk_document, count_tokens

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF financial report",
)
async def upload_document(
    file: UploadFile,
    store: DocumentStoreDep,
    settings: SettingsDep,
) -> DocumentUploadResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise InvalidFileTypeError("Only PDF files are supported.")

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File exceeds the {settings.max_upload_mb} MB upload limit."
        )

    parsed = parse_pdf(data, max_pages=settings.max_pdf_pages)
    chunks = chunk_document(
        parsed, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    text = parsed.text

    record, deduplicated = store.add(
        filename=file.filename or "report.pdf",
        raw_bytes=data,
        text=text,
        num_pages=parsed.num_pages,
        token_count=count_tokens(text),
        chunks=chunks,
        metadata=parsed.metadata,
    )

    return DocumentUploadResponse(
        document_id=record.id,
        filename=record.filename,
        num_pages=record.num_pages,
        token_count=record.token_count,
        chunk_count=len(record.chunks),
        deduplicated=deduplicated,
        metadata=DocumentMetadata(**record.metadata),
    )


@router.get("", response_model=DocumentListResponse, summary="List uploaded documents")
def list_documents(store: DocumentStoreDep) -> DocumentListResponse:
    return DocumentListResponse(
        documents=[
            DocumentSummary(
                document_id=r.id,
                filename=r.filename,
                num_pages=r.num_pages,
                token_count=r.token_count,
                created_at=r.created_at,
            )
            for r in store.list()
        ]
    )


@router.get("/{document_id}", response_model=DocumentSummary, summary="Get document details")
def get_document(document_id: str, store: DocumentStoreDep) -> DocumentSummary:
    record = store.get(document_id)
    return DocumentSummary(
        document_id=record.id,
        filename=record.filename,
        num_pages=record.num_pages,
        token_count=record.token_count,
        created_at=record.created_at,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its index",
)
def delete_document(document_id: str, store: DocumentStoreDep) -> None:
    store.delete(document_id)
