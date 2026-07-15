"""PDF parsing: extract per-page text and document metadata from raw bytes.

Parsing works on in-memory bytes so uploaded files never touch disk.
"""
import io
import logging
from dataclasses import dataclass, field

from pypdf import PdfReader

from backend.core.exceptions import EmptyDocumentError, PDFExtractionError

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    page_number: int  # 1-indexed
    text: str


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def num_pages(self) -> int:
        return len(self.pages)


def parse_pdf(data: bytes, max_pages: int = 500) -> ParsedDocument:
    """Parse PDF bytes into per-page text plus metadata.

    Raises:
        PDFExtractionError: the bytes are not a readable PDF.
        EmptyDocumentError: the PDF has no extractable text (e.g. scanned images).
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages)

        pages: list[ParsedPage] = []
        for i in range(pages_to_read):
            page_text = (reader.pages[i].extract_text() or "").strip()
            if page_text:
                pages.append(ParsedPage(page_number=i + 1, text=page_text))

        metadata = _extract_metadata(reader, total_pages)
    except (PDFExtractionError, EmptyDocumentError):
        raise
    except Exception as exc:
        logger.exception("Failed to parse PDF")
        raise PDFExtractionError(f"Could not read PDF: {exc}") from exc

    if not pages:
        raise EmptyDocumentError(
            "No extractable text found in the PDF. "
            "Scanned/image-only documents are not supported."
        )

    if total_pages > max_pages:
        logger.warning("PDF has %d pages; truncated to %d", total_pages, max_pages)

    return ParsedDocument(pages=pages, metadata=metadata)


def _extract_metadata(reader: PdfReader, total_pages: int) -> dict:
    meta = reader.metadata
    return {
        "total_pages": total_pages,
        "title": (meta.title if meta else None) or None,
        "author": (meta.author if meta else None) or None,
        "subject": (meta.subject if meta else None) or None,
    }
