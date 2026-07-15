"""Orchestrates document analysis and retrieval-augmented Q&A.

Responsibilities:
- pick the stuff vs map-reduce strategy based on document size,
- run requested analyses concurrently and cache results per document,
- lazily build the vector index on first Q&A request,
- return source citations alongside answers.
"""
import asyncio
import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from backend.chains.analysis_chains import build_analysis_chain, build_qa_chain, condense_document
from backend.core.config import Settings
from backend.services import rag_service
from backend.services.document_store import DocumentRecord

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, llm: BaseChatModel, settings: Settings):
        self._llm = llm
        self._settings = settings

    async def analyze(
        self, record: DocumentRecord, analyses: list[str]
    ) -> tuple[dict[str, str], list[str]]:
        """Run the requested analyses. Returns (results, served_from_cache)."""
        cached = [a for a in analyses if a in record.analysis_cache]
        pending = [a for a in analyses if a not in record.analysis_cache]

        if pending:
            text = await self._prepare_input(record)
            outputs = await asyncio.gather(
                *(build_analysis_chain(self._llm, a).ainvoke({"text": text}) for a in pending)
            )
            for analysis_type, output in zip(pending, outputs, strict=True):
                record.analysis_cache[analysis_type] = output.strip()

        results = {a: record.analysis_cache[a] for a in analyses}
        return results, cached

    async def _prepare_input(self, record: DocumentRecord) -> str:
        """Return the text each analysis chain should see for this document."""
        if record.token_count <= self._settings.stuff_threshold_tokens:
            return record.text

        if record.condensed_text is None:
            logger.info(
                "Document %s (%d tokens) exceeds stuff threshold; running map-reduce",
                record.id,
                record.token_count,
            )
            record.condensed_text = await condense_document(
                self._llm, record.chunks, self._settings
            )
        return record.condensed_text

    async def answer_question(
        self, record: DocumentRecord, question: str, embeddings: Embeddings
    ) -> tuple[str, list[Document]]:
        """Answer a question via RAG. Returns (answer, source_chunks)."""
        if record.vector_store is None:
            logger.info("Building vector index for document %s on first Q&A", record.id)
            record.vector_store = await asyncio.to_thread(
                rag_service.build_index, record.chunks, embeddings
            )

        sources = await asyncio.to_thread(
            rag_service.retrieve, record.vector_store, question, self._settings.retrieval_k
        )
        context = rag_service.format_context(sources)
        answer = await build_qa_chain(self._llm).ainvoke(
            {"context": context, "question": question}
        )
        return answer.strip(), sources
