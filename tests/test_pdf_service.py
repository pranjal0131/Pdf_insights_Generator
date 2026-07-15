import pytest

from backend.core.exceptions import EmptyDocumentError, PDFExtractionError
from backend.services.pdf_service import parse_pdf


def test_parse_pdf_extracts_pages(sample_pdf_bytes):
    parsed = parse_pdf(sample_pdf_bytes)
    assert parsed.num_pages == 2
    assert "Revenue increased 20%" in parsed.pages[0].text
    assert parsed.pages[0].page_number == 1
    assert parsed.metadata["total_pages"] == 2


def test_parse_pdf_text_property_joins_pages(sample_pdf_bytes):
    parsed = parse_pdf(sample_pdf_bytes)
    assert "Revenue increased" in parsed.text
    assert "Risk factors" in parsed.text


def test_parse_pdf_respects_max_pages(sample_pdf_bytes):
    parsed = parse_pdf(sample_pdf_bytes, max_pages=1)
    assert parsed.num_pages == 1


def test_parse_pdf_rejects_garbage():
    with pytest.raises(PDFExtractionError):
        parse_pdf(b"this is definitely not a pdf")


def test_parse_pdf_rejects_textless_document():
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()  # blank page, no text
    blank = bytes(pdf.output())

    with pytest.raises(EmptyDocumentError):
        parse_pdf(blank)
