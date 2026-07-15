import pytest
from langchain_core.documents import Document

from backend.chains.analysis_chains import batch_chunks, condense_document
from backend.core.config import Settings
from backend.services.analysis_service import AnalysisService
from backend.services.document_store import DocumentStore
from backend.services.text_service import count_tokens
from tests.conftest import FAKE_ANSWER


def _chunks(n: int, sentence: str = "Revenue grew twenty percent this year. ") -> list[Document]:
    return [Document(page_content=sentence * 5, metadata={"page": i + 1}) for i in range(n)]


def test_batch_chunks_respects_budget():
    chunks = _chunks(10)
    per_chunk = count_tokens(chunks[0].page_content)
    budget = per_chunk * 3
    batches = batch_chunks(chunks, budget)
    assert len(batches) >= 3
    assert all(count_tokens(b) <= budget + per_chunk for b in batches)


def test_batch_chunks_single_batch_when_small():
    batches = batch_chunks(_chunks(2), batch_token_budget=10_000)
    assert len(batches) == 1


@pytest.mark.asyncio
async def test_condense_document_map_reduce(fake_llm):
    settings = Settings(openai_api_key="test", stuff_threshold_tokens=50)
    condensed = await condense_document(fake_llm, _chunks(6), settings)
    assert FAKE_ANSWER in condensed


@pytest.mark.asyncio
async def test_analysis_results_are_cached(fake_llm):
    settings = Settings(openai_api_key="test")
    service = AnalysisService(llm=fake_llm, settings=settings)

    store = DocumentStore()
    record, _ = store.add(
        filename="r.pdf",
        raw_bytes=b"pdf-bytes",
        text="Revenue grew 20% to $5.2M.",
        num_pages=1,
        token_count=12,
        chunks=_chunks(1),
        metadata={},
    )

    results, cached = await service.analyze(record, ["summary", "key_insights"])
    assert set(results) == {"summary", "key_insights"}
    assert results["summary"] == FAKE_ANSWER
    assert cached == []

    _, cached_second = await service.analyze(record, ["summary", "key_insights"])
    assert set(cached_second) == {"summary", "key_insights"}
