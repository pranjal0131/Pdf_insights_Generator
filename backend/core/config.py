"""Application configuration via pydantic-settings.

All values can be overridden through environment variables or a `.env` file.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Financial Report Insights API"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- LLM ---
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    embedding_model: str = "text-embedding-3-small"
    llm_max_retries: int = 3
    llm_request_timeout: float = 120.0

    # --- Document processing ---
    max_pdf_pages: int = 500
    max_upload_mb: int = 25
    chunk_size: int = 1200
    chunk_overlap: int = 150

    # --- RAG ---
    retrieval_k: int = 6

    # Documents under this token count are summarized in a single call ("stuff");
    # larger ones go through map-reduce over chunk batches.
    stuff_threshold_tokens: int = 12_000
    # Hard cap on tokens sent in a single LLM call.
    max_context_tokens: int = 100_000

    # --- Document store ---
    max_documents: int = 50

    # --- CORS (frontend origins) ---
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
