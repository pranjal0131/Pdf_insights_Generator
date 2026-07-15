"""Prompt templates for every analysis chain.

All prompts are ChatPromptTemplates with a system role establishing the
analyst persona, which measurably improves consistency over bare user prompts.
"""
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = (
    "You are a senior financial analyst. You write precise, factual analyses "
    "grounded strictly in the provided report text. When the text does not "
    "support a claim, you say so instead of speculating."
)


def _analysis_prompt(instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            ("human", instruction + "\n\nFinancial report text:\n```\n{text}\n```"),
        ]
    )


ANALYSIS_PROMPTS: dict[str, ChatPromptTemplate] = {
    "summary": _analysis_prompt(
        "Write a concise executive summary of the following financial report. "
        "Cover key metrics, performance highlights, and overall financial health "
        "in 3-5 short paragraphs."
    ),
    "key_insights": _analysis_prompt(
        "Extract the key financial insights from the following report. "
        "Present 5-10 bullet points covering major findings, anomalies, and "
        "notable figures. Quote concrete numbers where available."
    ),
    "trend_analysis": _analysis_prompt(
        "Analyze the trends in the following financial report. Identify upward "
        "and downward trends, quantify them where possible, and explain likely "
        "drivers mentioned in the text."
    ),
    "risk_assessment": _analysis_prompt(
        "Identify and assess the risks stated or implied in the following "
        "financial report. Categorize each risk (market, operational, financial, "
        "regulatory) and rate its severity as low/medium/high with a one-line "
        "justification."
    ),
    "recommendations": _analysis_prompt(
        "Based on the following financial report, provide 4-8 actionable "
        "recommendations for analysts and management. Address both opportunities "
        "and challenges, and tie each recommendation to evidence from the text."
    ),
}

# --- Map-reduce condensation for documents that exceed the context budget ---

MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        (
            "human",
            "Condense the following excerpt of a financial report into dense "
            "analyst notes. Preserve every concrete figure, metric, trend, risk, "
            "and forward-looking statement. Omit boilerplate.\n\n"
            "Excerpt:\n```\n{text}\n```",
        ),
    ]
)

REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        (
            "human",
            "The following are analyst notes covering consecutive sections of one "
            "financial report. Merge them into a single coherent set of notes, "
            "removing duplication but preserving all concrete figures and findings."
            "\n\nNotes:\n```\n{text}\n```",
        ),
    ]
)

# --- Retrieval-augmented Q&A ---

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            _SYSTEM
            + " Answer using ONLY the provided context excerpts. If the answer "
            "is not in the context, state that the document does not contain "
            "this information. Cite page numbers like (p. 12) when you use a fact.",
        ),
        (
            "human",
            "Context excerpts from the report:\n{context}\n\nQuestion: {question}",
        ),
    ]
)
