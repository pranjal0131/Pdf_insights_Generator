"""Vector indexing and retrieval for document Q&A.

Uses FAISS for similarity search when available, falling back to LangChain's
in-memory vector store (useful in environments without native wheels).
"""
import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

logger = logging.getLogger(__name__)

try:
    from langchain_community.vectorstores import FAISS

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - depends on environment
    _HAS_FAISS = False


def build_index(chunks: list[Document], embeddings: Embeddings) -> VectorStore:
    """Embed chunks and build a similarity-search index over them."""
    if _HAS_FAISS:
        try:
            return FAISS.from_documents(chunks, embeddings)
        except ImportError:
            logger.warning("faiss package unavailable; falling back to in-memory store")
    return InMemoryVectorStore.from_documents(chunks, embeddings)


def retrieve(store: VectorStore, query: str, k: int = 6) -> list[Document]:
    """Return the k chunks most semantically similar to the query."""
    return store.similarity_search(query, k=k)


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into a prompt-ready context block with page tags."""
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
