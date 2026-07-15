"""FastAPI dependency providers.

Tests override these via `app.dependency_overrides` to inject fake models,
so no production code needs test hooks.
"""
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from backend.core.config import Settings, get_settings
from backend.services.analysis_service import AnalysisService
from backend.services.document_store import DocumentStore
from backend.services.llm_factory import create_chat_model, create_embeddings

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def get_document_store() -> DocumentStore:
    return DocumentStore(max_documents=get_settings().max_documents)


DocumentStoreDep = Annotated[DocumentStore, Depends(get_document_store)]


def get_llm(settings: SettingsDep) -> BaseChatModel:
    return create_chat_model(settings)


def get_embeddings(settings: SettingsDep) -> Embeddings:
    return create_embeddings(settings)


LLMDep = Annotated[BaseChatModel, Depends(get_llm)]
EmbeddingsDep = Annotated[Embeddings, Depends(get_embeddings)]


def get_analysis_service(llm: LLMDep, settings: SettingsDep) -> AnalysisService:
    return AnalysisService(llm=llm, settings=settings)


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
