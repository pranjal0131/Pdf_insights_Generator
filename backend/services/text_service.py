"""Text cleaning, token counting, and page-aware chunking."""
import logging
import re

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.services.pdf_service import ParsedDocument

logger = logging.getLogger(__name__)

# cl100k_base is the tokenizer family used by current OpenAI chat/embedding models.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters, preserving line structure."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse blank-line runs
    return text.strip()


def chunk_document(
    parsed: ParsedDocument,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split a parsed PDF into chunks, tagging each chunk with its source page.

    Chunk sizes are measured in tokens (not characters) so downstream context
    budgets are predictable. Page numbers ride along in metadata, which lets
    the Q&A endpoint cite sources.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for page in parsed.pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        for piece in splitter.split_text(cleaned):
            chunks.append(
                Document(page_content=piece, metadata={"page": page.page_number})
            )

    logger.info("Document split into %d chunks from %d pages", len(chunks), parsed.num_pages)
    return chunks
