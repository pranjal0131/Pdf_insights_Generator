import pytest

from backend.core.exceptions import DocumentNotFoundError
from backend.services.document_store import DocumentStore


def _add(store: DocumentStore, raw: bytes, filename: str = "r.pdf"):
    return store.add(
        filename=filename,
        raw_bytes=raw,
        text="text",
        num_pages=1,
        token_count=10,
        chunks=[],
        metadata={},
    )


def test_add_and_get_roundtrip():
    store = DocumentStore()
    record, dedup = _add(store, b"content-1")
    assert not dedup
    assert store.get(record.id).id == record.id


def test_identical_content_is_deduplicated():
    store = DocumentStore()
    first, _ = _add(store, b"same-bytes")
    second, dedup = _add(store, b"same-bytes", filename="other-name.pdf")
    assert dedup
    assert second.id == first.id
    assert len(store) == 1


def test_lru_eviction_at_capacity():
    store = DocumentStore(max_documents=2)
    a, _ = _add(store, b"a")
    b, _ = _add(store, b"b")
    store.get(a.id)  # touch a so b becomes least recently used
    c, _ = _add(store, b"c")

    assert len(store) == 2
    store.get(a.id)
    store.get(c.id)
    with pytest.raises(DocumentNotFoundError):
        store.get(b.id)


def test_delete_and_missing_lookup():
    store = DocumentStore()
    record, _ = _add(store, b"x")
    store.delete(record.id)
    with pytest.raises(DocumentNotFoundError):
        store.get(record.id)
    with pytest.raises(DocumentNotFoundError):
        store.delete("nope")
