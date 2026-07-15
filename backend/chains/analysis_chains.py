"""LCEL chains for document analysis.

Two strategies, chosen by document size:

- "stuff": documents under the token threshold are analyzed in a single call.
- "map-reduce": larger documents are condensed first — chunks are grouped into
  token-budgeted batches, each batch is condensed concurrently (map), and the
  condensed notes are merged (reduce, recursively if needed). Every analysis
  type then runs against the condensed notes, which are computed only once
  per document.
"""
import asyncio
import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from backend.chains.prompts import ANALYSIS_PROMPTS, MAP_PROMPT, QA_PROMPT, REDUCE_PROMPT
from backend.core.config import Settings
from backend.services.text_service import count_tokens

logger = logging.getLogger(__name__)

# Cap on concurrent LLM calls during the map phase, to stay under rate limits.
MAP_CONCURRENCY = 5


def build_chain(llm: BaseChatModel, prompt: ChatPromptTemplate) -> Runnable:
    return prompt | llm | StrOutputParser()


def build_analysis_chain(llm: BaseChatModel, analysis_type: str) -> Runnable:
    return build_chain(llm, ANALYSIS_PROMPTS[analysis_type])


def build_qa_chain(llm: BaseChatModel) -> Runnable:
    return build_chain(llm, QA_PROMPT)


def batch_chunks(chunks: list[Document], batch_token_budget: int) -> list[str]:
    """Group consecutive chunks into batches that fit the token budget."""
    batches: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk.page_content)
        if current and current_tokens + chunk_tokens > batch_token_budget:
            batches.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(chunk.page_content)
        current_tokens += chunk_tokens

    if current:
        batches.append("\n\n".join(current))
    return batches


async def condense_document(
    llm: BaseChatModel,
    chunks: list[Document],
    settings: Settings,
) -> str:
    """Map-reduce a large document into condensed analyst notes."""
    map_chain = build_chain(llm, MAP_PROMPT)
    reduce_chain = build_chain(llm, REDUCE_PROMPT)

    batches = batch_chunks(chunks, settings.stuff_threshold_tokens)
    logger.info("Map phase: condensing %d batches", len(batches))

    semaphore = asyncio.Semaphore(MAP_CONCURRENCY)

    async def condense_batch(batch: str) -> str:
        async with semaphore:
            return await map_chain.ainvoke({"text": batch})

    notes = await asyncio.gather(*(condense_batch(b) for b in batches))

    # Reduce phase: merge notes; collapse recursively while over budget.
    merged = "\n\n".join(notes)
    rounds = 0
    while count_tokens(merged) > settings.stuff_threshold_tokens and rounds < 3:
        rounds += 1
        logger.info("Reduce phase round %d: collapsing %d tokens", rounds, count_tokens(merged))
        note_docs = [Document(page_content=n) for n in notes]
        sub_batches = batch_chunks(note_docs, settings.stuff_threshold_tokens)
        notes = await asyncio.gather(
            *(reduce_chain.ainvoke({"text": b}) for b in sub_batches)
        )
        merged = "\n\n".join(notes)

    if len(notes) > 1:
        merged = await reduce_chain.ainvoke({"text": merged})

    return merged
