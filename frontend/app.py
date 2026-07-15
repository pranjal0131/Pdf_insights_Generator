"""Streamlit frontend for the Financial Report Insights API.

Run (with the API already up):
    streamlit run frontend/app.py
"""
import contextlib

import streamlit as st
from api_client import APIError, InsightsAPIClient

ANALYSIS_LABELS = {
    "summary": "📋 Executive Summary",
    "key_insights": "💡 Key Insights",
    "trend_analysis": "📊 Trend Analysis",
    "risk_assessment": "⚠️ Risk Assessment",
    "recommendations": "🎯 Recommendations",
}

st.set_page_config(
    page_title="Financial Report Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_client() -> InsightsAPIClient:
    return InsightsAPIClient()


def init_state() -> None:
    defaults = {
        "document": None,       # upload response dict
        "results": None,        # analysis results dict
        "qa_history": [],       # list of {question, answer, sources}
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def check_api() -> dict | None:
    try:
        return get_client().health()
    except Exception:
        return None


def render_sidebar(health: dict | None) -> None:
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "AI-powered financial report analysis: upload a PDF, run analysis "
            "pipelines, and ask questions answered via retrieval-augmented "
            "generation with page-level citations."
        )
        st.header("🔌 Backend")
        if health is None:
            st.error(
                "API is not reachable. Start it with:\n\n"
                "`uvicorn backend.main:app --reload`"
            )
        else:
            st.success(f"Connected · v{health['version']} · {health['llm_model']}")
            if not health["llm_configured"]:
                st.warning("OPENAI_API_KEY is not set on the backend.")

        if st.session_state.document:
            st.header("📄 Current document")
            doc = st.session_state.document
            st.markdown(
                f"**{doc['filename']}**\n\n"
                f"- Pages: {doc['num_pages']}\n"
                f"- Tokens: {doc['token_count']:,}\n"
                f"- Chunks: {doc['chunk_count']}"
            )
            if st.button("🗑️ Remove document", use_container_width=True):
                with contextlib.suppress(APIError):  # may already be evicted server-side
                    get_client().delete_document(doc["document_id"])
                st.session_state.document = None
                st.session_state.results = None
                st.session_state.qa_history = []
                st.rerun()


def render_upload() -> None:
    st.header("📄 Upload Financial Report")
    uploaded = st.file_uploader(
        "Choose a PDF file", type=["pdf"], help="Text-based PDF, max 25 MB"
    )
    if uploaded is None:
        return

    current = st.session_state.document
    if current and current.get("filename") == uploaded.name:
        return  # already uploaded this file in this session

    try:
        with st.spinner("Uploading and indexing PDF..."):
            doc = get_client().upload_document(uploaded.name, uploaded.getvalue())
        st.session_state.document = doc
        st.session_state.results = None
        st.session_state.qa_history = []
        if doc["deduplicated"]:
            st.info("This document was already uploaded — reusing the existing index.")
        st.success("✅ PDF processed successfully!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", doc["num_pages"])
        col2.metric("Tokens", f"{doc['token_count']:,}")
        col3.metric("Chunks", doc["chunk_count"])
    except APIError as exc:
        st.error(f"❌ {exc.detail}")


def render_analysis() -> None:
    doc = st.session_state.document
    if not doc:
        st.info("📌 Upload a PDF first.")
        return

    st.header("🔍 Analysis")
    selected = st.multiselect(
        "Select analyses to run:",
        options=list(ANALYSIS_LABELS),
        default=["summary", "key_insights"],
        format_func=lambda key: ANALYSIS_LABELS[key],
    )

    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not selected:
            st.warning("Select at least one analysis type.")
        else:
            try:
                with st.spinner("Analyzing... large documents are condensed first."):
                    response = get_client().analyze(doc["document_id"], selected)
                st.session_state.results = response["results"]
                cached = response["served_from_cache"]
                note = f" ({len(cached)} from cache)" if cached else ""
                st.success(f"✅ Done in {response['elapsed_seconds']}s{note}")
            except APIError as exc:
                st.error(f"❌ {exc.detail}")

    st.divider()
    st.subheader("❓ Ask a question")
    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What were the main revenue drivers this quarter?",
        )
        submitted = st.form_submit_button("🔍 Ask")

    if submitted and question.strip():
        try:
            with st.spinner("Retrieving relevant passages and answering..."):
                qa = get_client().ask(doc["document_id"], question.strip())
            st.session_state.qa_history.append(qa)
        except APIError as exc:
            st.error(f"❌ {exc.detail}")

    for i, qa in enumerate(reversed(st.session_state.qa_history), 1):
        is_latest = i == 1
        with st.expander(f"Q: {qa['question'][:80]}", expanded=is_latest):
            st.markdown(qa["answer"])
            if qa["sources"]:
                st.caption("Sources")
                for src in qa["sources"]:
                    page = src["page"] if src["page"] is not None else "?"
                    st.markdown(f"> **p. {page}** — {src['snippet']}…")

    if st.session_state.qa_history and st.button("🗑️ Clear Q&A history"):
        st.session_state.qa_history = []
        st.rerun()


def render_results() -> None:
    results = st.session_state.results
    if not results:
        st.info("📌 Run an analysis to see results here.")
        return

    st.header("📈 Results")
    ordered = [key for key in ANALYSIS_LABELS if key in results]
    tabs = st.tabs([ANALYSIS_LABELS[key] for key in ordered])
    for tab, key in zip(tabs, ordered, strict=True):
        with tab:
            st.markdown(results[key])

    st.divider()
    export = "\n\n".join(
        f"## {ANALYSIS_LABELS[key].split(' ', 1)[1]}\n\n{results[key]}" for key in ordered
    )
    st.download_button(
        "📥 Download report (Markdown)",
        data=f"# Financial Analysis Report\n\n{export}",
        file_name="financial_analysis_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def main() -> None:
    init_state()
    st.title("📊 Financial Report Insights")
    st.markdown("*AI-powered analysis of financial reports — FastAPI · LangChain · RAG*")
    st.divider()

    health = check_api()
    render_sidebar(health)

    tab_upload, tab_analysis, tab_results = st.tabs(
        ["📤 Upload", "🔍 Analyze & Ask", "📈 Results"]
    )
    with tab_upload:
        render_upload()
    with tab_analysis:
        render_analysis()
    with tab_results:
        render_results()


if __name__ == "__main__":
    main()
