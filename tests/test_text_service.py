from backend.services.pdf_service import parse_pdf
from backend.services.text_service import chunk_document, clean_text, count_tokens


def test_clean_text_normalizes_whitespace():
    raw = "Revenue\t\t grew   fast\n\n\n\nProfit  followed\x00"
    cleaned = clean_text(raw)
    assert cleaned == "Revenue grew fast\n\nProfit followed"


def test_count_tokens_positive():
    assert count_tokens("Revenue increased 20% to $5.2 million") > 0


def test_chunk_document_tags_pages(sample_pdf_bytes):
    parsed = parse_pdf(sample_pdf_bytes)
    chunks = chunk_document(parsed, chunk_size=100, chunk_overlap=10)
    assert chunks, "expected at least one chunk"
    pages = {c.metadata["page"] for c in chunks}
    assert pages == {1, 2}


def test_chunk_document_respects_token_budget(sample_pdf_bytes):
    parsed = parse_pdf(sample_pdf_bytes)
    chunks = chunk_document(parsed, chunk_size=30, chunk_overlap=5)
    assert all(count_tokens(c.page_content) <= 30 for c in chunks)
