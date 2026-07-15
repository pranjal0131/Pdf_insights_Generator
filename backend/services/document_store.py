"""In-memory document registry.

Holds parsed documents, their chunks, and their vector indexes for the lifetime
of the process. Thread-safe, deduplicates by content hash, and evicts the
least-recently-used document when the capacity limit is reached.

For a single-instance deployment this is intentionally simple; swapping in
Redis/Postgres + a persistent vector store only requires re-implementing this
interface.
"""
import hashlib
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from backend.core.exceptions import DocumentNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    id: str
    filename: str
    content_hash: str
    text: str
    num_pages: int
    token_count: int
    chunks: list[Document]
    metadata: dict
    created_at: datetime
    vector_store: VectorStore | None = None
    # Map-reduce condensed version of the document, computed once and reused
    # by every analysis type on large documents.
    condensed_text: str | None = None
    # Analysis results cached per analysis type, keyed by type name.
    analysis_cache: dict[str, str] = field(default_factory=dict)


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DocumentStore:
    def __init__(self, max_documents: int = 50):
        self._max_documents = max_documents
        self._lock = threading.Lock()
        self._docs: OrderedDict[str, DocumentRecord] = OrderedDict()

    def add(
        self,
        *,
        filename: str,
        raw_bytes: bytes,
        text: str,
        num_pages: int,
        token_count: int,
        chunks: list[Document],
        metadata: dict,
        vector_store: VectorStore | None = None,
    ) -> tuple[DocumentRecord, bool]:
        """Register a document. Returns (record, deduplicated).

        If a document with identical content already exists, the existing
        record is returned and no new entry is created.
        """
        digest = content_hash(raw_bytes)
        with self._lock:
            for record in self._docs.values():
                if record.content_hash == digest:
                    self._docs.move_to_end(record.id)
                    logger.info("Duplicate upload detected; reusing document %s", record.id)
                    return record, True

            record = DocumentRecord(
                id=uuid.uuid4().hex,
                filename=filename,
                content_hash=digest,
                text=text,
                num_pages=num_pages,
                token_count=token_count,
                chunks=chunks,
                metadata=metadata,
                created_at=datetime.now(UTC),
                vector_store=vector_store,
            )
            self._docs[record.id] = record

            while len(self._docs) > self._max_documents:
                evicted_id, _ = self._docs.popitem(last=False)
                logger.info("Evicted least-recently-used document %s", evicted_id)

            return record, False

    def get(self, document_id: str) -> DocumentRecord:
        with self._lock:
            record = self._docs.get(document_id)
            if record is None:
                raise DocumentNotFoundError(f"Document '{document_id}' not found")
            self._docs.move_to_end(document_id)
            return record

    def delete(self, document_id: str) -> None:
        with self._lock:
            if self._docs.pop(document_id, None) is None:
                raise DocumentNotFoundError(f"Document '{document_id}' not found")

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._docs.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)
