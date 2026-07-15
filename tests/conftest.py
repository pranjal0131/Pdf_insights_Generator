"""Shared fixtures: fake LLM/embeddings, PDF builders, and a wired test app."""
import itertools

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.api.dependencies import get_document_store, get_embeddings, get_llm
from backend.main import create_app
from backend.services.document_store import DocumentStore

FAKE_ANSWER = "FAKE_ANALYSIS_OUTPUT: revenue grew 20% year over year."


def make_pdf(pages: list[str]) -> bytes:
    """Build an in-memory text PDF with one entry per page."""
    pdf = FPDF()
    pdf.set_font("helvetica", size=12)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return make_pdf(
        [
            "Annual Report 2024. Revenue increased 20% to $5.2 million. "
            "Operating profit reached $1.1 million, up from $0.8 million.",
            "Risk factors: currency volatility and supply chain pressure remain "
            "key operational risks for the coming fiscal year.",
        ]
    )


@pytest.fixture
def fake_llm() -> GenericFakeChatModel:
    return GenericFakeChatModel(
        messages=itertools.cycle([AIMessage(content=FAKE_ANSWER)])
    )


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def client(fake_llm, fake_embeddings) -> TestClient:
    """Test client with fake models and a fresh document store per test."""
    app = create_app()
    store = DocumentStore(max_documents=10)
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_embeddings] = lambda: fake_embeddings
    app.dependency_overrides[get_document_store] = lambda: store
    return TestClient(app)
