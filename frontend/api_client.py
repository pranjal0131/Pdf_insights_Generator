"""Thin HTTP client for the Insights API used by the Streamlit frontend."""
import os
from typing import Any

import httpx

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"


class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class InsightsAPIClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 600.0):
        self._client = httpx.Client(base_url=base_url + API_PREFIX, timeout=timeout)

    def _handle(self, response: httpx.Response) -> Any:
        if response.is_success:
            return response.json() if response.content else None
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(response.status_code, str(detail))

    def health(self) -> dict:
        return self._handle(self._client.get("/health"))

    def upload_document(self, filename: str, data: bytes) -> dict:
        return self._handle(
            self._client.post(
                "/documents",
                files={"file": (filename, data, "application/pdf")},
            )
        )

    def list_documents(self) -> dict:
        return self._handle(self._client.get("/documents"))

    def delete_document(self, document_id: str) -> None:
        self._handle(self._client.delete(f"/documents/{document_id}"))

    def analyze(self, document_id: str, analyses: list[str]) -> dict:
        return self._handle(
            self._client.post(
                f"/documents/{document_id}/analysis",
                json={"analyses": analyses},
            )
        )

    def ask(self, document_id: str, question: str) -> dict:
        return self._handle(
            self._client.post(
                f"/documents/{document_id}/qa",
                json={"question": question},
            )
        )
