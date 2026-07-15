"""Factories for chat and embedding models.

Kept behind functions so tests can substitute fakes via FastAPI dependency
overrides, and so provider choice stays in one place.
"""
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.core.config import Settings
from backend.core.exceptions import LLMConfigurationError


def _require_api_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not configured. Set it in your environment or .env file."
        )
    return settings.openai_api_key


def create_chat_model(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=_require_api_key(settings),
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_request_timeout,
    )


def create_embeddings(settings: Settings) -> Embeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=_require_api_key(settings),
    )
