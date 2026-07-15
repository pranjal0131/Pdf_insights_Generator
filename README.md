# Financial Report Insights Generator 📊

AI-powered financial report analysis platform: upload a PDF report, run LLM analysis pipelines (summary, insights, trends, risks, recommendations), and ask free-form questions answered via **retrieval-augmented generation (RAG)** with page-level source citations.

Built as a production-style service: a **FastAPI** REST backend with a layered architecture, a **Streamlit** frontend that consumes it over HTTP, a fully mocked **pytest** suite, **Docker** packaging, and **GitHub Actions** CI.

## Architecture

```
┌──────────────┐   HTTP/JSON    ┌─────────────────────────────────────────────┐
│  Streamlit   │ ─────────────► │                FastAPI (api/v1)             │
│  frontend    │                │  routes → dependencies → services → chains  │
└──────────────┘                └─────────────────────────────────────────────┘
                                       │                    │
                          ┌────────────┴─────┐   ┌──────────┴───────────┐
                          │  Document store  │   │   LangChain (LCEL)   │
                          │  in-memory, LRU, │   │  stuff / map-reduce  │
                          │  dedup by hash   │   │  analysis chains     │
                          └────────┬─────────┘   └──────────┬───────────┘
                                   │                        │
                          ┌────────┴─────────┐   ┌──────────┴───────────┐
                          │  FAISS vector    │   │   OpenAI chat +      │
                          │  index (per doc) │   │   embedding models   │
                          └──────────────────┘   └──────────────────────┘
```

```
backend/
├── main.py               # App factory, CORS, error handlers, request logging
├── api/
│   ├── dependencies.py   # DI providers (overridable in tests)
│   └── routes/           # health, documents, analysis endpoints
├── core/                 # Settings (pydantic-settings), exceptions, logging
├── models/schemas.py     # Pydantic request/response contracts
├── services/             # PDF parsing, chunking, document store, RAG, orchestration
└── chains/               # Prompts + LCEL chains (incl. map-reduce condensation)
frontend/                 # Streamlit UI + typed HTTP client
tests/                    # 28 tests, fake LLM/embeddings — no API key needed
```

### Key design decisions

- **Token-aware processing** — chunk sizes and context budgets are measured in tokens (tiktoken), not characters, so LLM context limits are respected predictably.
- **Stuff vs map-reduce strategy** — small documents are analyzed in one call; large ones are condensed via concurrent map-reduce (bounded by a semaphore to respect rate limits), and the condensed notes are computed **once** and reused by every analysis type.
- **RAG Q&A with citations** — chunks carry their source page in metadata; answers cite pages, and the API returns the retrieved snippets as evidence.
- **Lazy vector indexing** — embeddings are only computed on the first question, so uploads stay fast and analysis-only usage never pays embedding cost.
- **Caching & dedup** — analysis results are cached per document + analysis type; re-uploading identical bytes (SHA-256) reuses the existing document and its caches.
- **Bounded memory** — the in-memory store is thread-safe and evicts least-recently-used documents at capacity. Swapping in Redis/Postgres only requires re-implementing one class.
- **Typed error handling** — domain exceptions carry HTTP status codes; one exception handler maps them to consistent JSON errors (404, 413, 415, 422, 503...).
- **Testability by design** — LLM and embeddings are injected via FastAPI dependencies, so the entire suite runs against fake models with zero network calls.

## API

Interactive docs at `http://localhost:8000/docs` (OpenAPI).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health + LLM configuration status |
| `POST` | `/api/v1/documents` | Upload & index a PDF (multipart) |
| `GET` | `/api/v1/documents` | List uploaded documents |
| `GET` | `/api/v1/documents/{id}` | Document details |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and its index |
| `POST` | `/api/v1/documents/{id}/analysis` | Run selected analyses (concurrent, cached) |
| `POST` | `/api/v1/documents/{id}/qa` | RAG Q&A with page-cited sources |

## Getting started

### Prerequisites

- Python 3.11+ (3.12/3.13 tested in CI)
- An OpenAI API key

### Local development

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt

# 2. Configure
copy .env.example .env         # then set OPENAI_API_KEY

# 3. Run the API
uvicorn backend.main:app --reload

# 4. Run the frontend (second terminal)
streamlit run frontend/app.py
```

Frontend: `http://localhost:8501` · API docs: `http://localhost:8000/docs`

### Docker

```bash
OPENAI_API_KEY=sk-... docker compose up --build
```

### Tests & lint

```bash
pytest --cov=backend    # 28 tests, no API key required (fake LLM/embeddings)
ruff check .
```

## Configuration

All settings are environment variables (see `.env.example`), validated at startup by pydantic-settings:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for real analysis |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for RAG |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `150` | Chunking, in **tokens** |
| `STUFF_THRESHOLD_TOKENS` | `12000` | Above this, map-reduce condensation kicks in |
| `RETRIEVAL_K` | `6` | Chunks retrieved per question |
| `MAX_UPLOAD_MB` / `MAX_PDF_PAGES` | `25` / `500` | Upload guardrails |
| `MAX_DOCUMENTS` | `50` | LRU capacity of the document store |

## Tech stack

FastAPI · LangChain (LCEL) · OpenAI · FAISS · tiktoken · Streamlit · Pydantic v2 · pytest · ruff · Docker · GitHub Actions

## Limitations & roadmap

- Documents live in process memory — a Redis/Postgres-backed store would enable horizontal scaling and persistence across restarts.
- Scanned (image-only) PDFs are rejected; OCR (e.g. Tesseract) is a natural extension.
- No authentication — add API keys/JWT before exposing publicly.
- Streaming responses (SSE) for long analyses would improve perceived latency.

## Disclaimer

This tool assists financial analysts; it is not financial advice. Outputs should be reviewed by qualified professionals.
