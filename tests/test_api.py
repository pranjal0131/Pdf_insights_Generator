"""End-to-end API tests against the wired app with fake LLM/embeddings."""
from tests.conftest import FAKE_ANSWER


def _upload(client, pdf_bytes, name="report.pdf"):
    return client.post(
        "/api/v1/documents",
        files={"file": (name, pdf_bytes, "application/pdf")},
    )


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_upload_document(client, sample_pdf_bytes):
    response = _upload(client, sample_pdf_bytes)
    assert response.status_code == 201
    body = response.json()
    assert body["num_pages"] == 2
    assert body["chunk_count"] >= 2
    assert body["token_count"] > 0
    assert body["deduplicated"] is False
    assert body["metadata"]["total_pages"] == 2


def test_upload_duplicate_is_deduplicated(client, sample_pdf_bytes):
    first = _upload(client, sample_pdf_bytes).json()
    second = _upload(client, sample_pdf_bytes, name="copy.pdf").json()
    assert second["deduplicated"] is True
    assert second["document_id"] == first["document_id"]


def test_upload_rejects_non_pdf_extension(client):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_rejects_unreadable_pdf(client):
    response = _upload(client, b"not really a pdf")
    assert response.status_code == 422


def test_list_and_delete_document(client, sample_pdf_bytes):
    doc_id = _upload(client, sample_pdf_bytes).json()["document_id"]

    listed = client.get("/api/v1/documents").json()["documents"]
    assert [d["document_id"] for d in listed] == [doc_id]

    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404


def test_analysis_returns_results_and_caches(client, sample_pdf_bytes):
    doc_id = _upload(client, sample_pdf_bytes).json()["document_id"]

    response = client.post(
        f"/api/v1/documents/{doc_id}/analysis",
        json={"analyses": ["summary", "risk_assessment"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"]["summary"] == FAKE_ANSWER
    assert body["results"]["risk_assessment"] == FAKE_ANSWER
    assert body["served_from_cache"] == []

    repeat = client.post(
        f"/api/v1/documents/{doc_id}/analysis",
        json={"analyses": ["summary"]},
    ).json()
    assert repeat["served_from_cache"] == ["summary"]


def test_analysis_rejects_unknown_type(client, sample_pdf_bytes):
    doc_id = _upload(client, sample_pdf_bytes).json()["document_id"]
    response = client.post(
        f"/api/v1/documents/{doc_id}/analysis",
        json={"analyses": ["astrology"]},
    )
    assert response.status_code == 422


def test_analysis_unknown_document_404(client):
    response = client.post(
        "/api/v1/documents/does-not-exist/analysis",
        json={"analyses": ["summary"]},
    )
    assert response.status_code == 404


def test_qa_returns_answer_with_sources(client, sample_pdf_bytes):
    doc_id = _upload(client, sample_pdf_bytes).json()["document_id"]

    response = client.post(
        f"/api/v1/documents/{doc_id}/qa",
        json={"question": "What were the main risks?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == FAKE_ANSWER
    assert body["sources"], "expected retrieved source chunks"
    assert all("snippet" in s for s in body["sources"])
    assert all(isinstance(s["page"], int) for s in body["sources"])


def test_qa_validates_question_length(client, sample_pdf_bytes):
    doc_id = _upload(client, sample_pdf_bytes).json()["document_id"]
    response = client.post(f"/api/v1/documents/{doc_id}/qa", json={"question": "a"})
    assert response.status_code == 422
