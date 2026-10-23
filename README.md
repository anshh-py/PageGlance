# PageGlance

**Ask questions about webpages and documents, backed by retrieval-augmented generation.**

PageGlance is a Chrome extension with a FastAPI backend that lets users ask questions about webpage content and uploaded PDF or text documents. It combines local semantic embeddings, PostgreSQL-based retrieval, and LLM-generated responses with source citations.

## Architecture

```mermaid
flowchart TD
    A[Chrome Extension] --> B[FastAPI Backend]
    B --> C[Content Extraction]
    B --> D[Document Ingestion]
    C --> E[Chunking and Normalization]
    D --> E
    E --> F[Sentence Transformers]
    F --> G[(PostgreSQL + pgvector)]
    G --> H[Hybrid Retrieval]
    H --> I[Optional Cross-Encoder Reranking]
    I --> J[LLM Response Generation]
    J --> K[Source Validation and Citations]
    K --> A
```

## Core Capabilities

- **Context-aware chat:** Ask questions about the active webpage or selected documents, with support for follow-up questions and streaming responses.
- **Retrieval-augmented generation:** Retrieve relevant chunks using semantic vector search and PostgreSQL full-text search, combined through Reciprocal Rank Fusion.
- **Local embeddings:** Generate 384-dimensional embeddings using Sentence Transformers without relying on a hosted embedding API.
- **Document ingestion:** Process PDF and TXT uploads, including OCR for scanned PDF pages, with bounded processing limits.
- **Source grounding:** Associate answers with retrieved sources and reject unsupported or invalid citations.
- **Persistent retrieval:** Store document chunks, embedding metadata, and retrieval data in PostgreSQL with pgvector.
- **Retrieval evaluation:** Compare retrieval strategies using Recall@K, MRR, nDCG, citation recall, scope checks, and latency.

## Technology Stack

| Layer | Technologies |
|---|---|
| Extension | React, TypeScript, Vite, Chrome Extension Manifest V3 |
| API | Python, FastAPI, Pydantic |
| LLM | Groq API |
| Embeddings | Sentence Transformers, all-MiniLM-L6-v2 |
| Retrieval | Vector search, PostgreSQL full-text search, RRF |
| Reranking | Sentence Transformers CrossEncoder |
| Persistence | PostgreSQL, SQLAlchemy, pgvector, Alembic |
| Quality | Pytest, ESLint, retrieval evaluation |

## Engineering Design

### Retrieval pipeline

Documents and webpages are normalized, split into chunks, and embedded using a shared embedding pipeline. Retrieval combines vector similarity with lexical search to improve coverage across natural-language questions and exact terms.

An optional cross-encoder reranks candidate passages. Retrieved evidence is scoped to the active webpage or explicitly selected documents before being passed to the generation layer.

### Reliability and safety

- Bounded input sizes, document processing, and retrieval candidates.
- URL normalization and page-scoped retrieval to reduce cross-page contamination.
- Validation of generated source references against retrieved evidence.
- Fallback behavior when retrieval is unavailable or evidence is insufficient.
- Database migrations and automated tests for ingestion, retrieval, API behavior, and persistence.

### Evaluation

The repository includes a versioned retrieval evaluation dataset and a reproducible evaluation runner. It measures retrieval quality and latency across vector-only, lexical-only, and fused retrieval.

The evaluation is intended as a regression baseline; results on a small labeled dataset should not be interpreted as proof of production-wide answer accuracy.

## Local Development

### Prerequisites

- Python 3.11 or a compatible version supported by the backend dependencies
- Node.js and npm
- PostgreSQL with pgvector
- A Groq API key
- Tesseract OCR for scanned PDF support

### Backend

Install the backend and development dependencies in a virtual environment, configure the required runtime settings, apply the database migrations, and start the API:

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`, with interactive documentation at `/docs`.

### Chrome Extension

```powershell
cd extension
npm install
npm run lint
npm test
npm run build
```

Open `chrome://extensions`, enable Developer mode, and load the generated `extension/dist` directory using **Load unpacked**.

### Testing

Run the backend test suite from the backend directory:

```powershell
python -m pytest -q
```

PostgreSQL integration tests are opt-in and require a dedicated disposable test database.

## Project Structure

```text
PageGlance/
├── backend/
│   ├── app/             # API, ingestion, embeddings and retrieval
│   ├── migrations/      # Database schema migrations
│   ├── tests/           # Unit and integration tests
│   └── evals/           # Retrieval evaluation dataset and runner
├── extension/
│   └── src/             # Extension UI and client-side logic
└── README.md
```

## Current Scope

PageGlance supports webpage question answering and retrieval over uploaded text and PDF documents. Retrieval evaluation currently focuses on a small labeled dataset. Production deployment, authentication, multi-user data isolation, and broader answer-quality evaluation remain separate engineering concerns.

## License

<<<<<<< HEAD
No license has been specified yet.
=======
```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"
```

`PAGEGLANCE_EMBEDDING_BATCH_SIZE` defaults to `64`. Document uploads, automatic webpage indexing, and query retrieval all use the same cached provider factory and model profile. Model name, provider, dimensions, and exact chunk-text hash are persisted; vectors are reused only when the profile and hash match. Tests inject mock models and do not download model weights or make paid requests.

For backward compatibility, explicitly setting `PAGEGLANCE_EMBEDDING_PROVIDER=openai` still selects the optional hosted adapter and requires `OPENAI_API_KEY`; it is never selected automatically. The example environment defaults to local inference and does not include an OpenAI key placeholder.

### Safe 1536-to-384 vector transition

The existing flexible `embedding vector` column and its model metadata are left intact. Alembic revision `20261023_01` additively creates `local_embedding vector(384)` and separate local profile/hash metadata. It does not rewrite or drop existing OpenAI vectors. Local retrieval uses only the local column and exact local model/dimension profile; the optional legacy adapter uses only the original column. The two profiles cannot be mixed in retrieval or embedding reuse.

Apply the additive migration from `backend\` after backing up the database:

```powershell
alembic upgrade head
```

Existing documents are not bulk re-embedded by the migration. New uploads and pages are indexed locally when they are ingested; re-upload a document or revisit a webpage to generate its local vectors. Keep a restorable database backup before re-indexing. Legacy vectors remain stored alongside local vectors for unchanged chunks. To roll back, first export or back up local vectors: downgrading revision `20261023_01` removes only the newly added local-vector columns and their data. No bulk delete/re-index command is included in this phase.

### Vector retrieval (Phase 8D)

`VectorRetriever` combines local-vector candidates and PostgreSQL full-text candidates with Reciprocal Rank Fusion (RRF). Vector retrieval embeds each non-empty query with the configured embedding service and searches only chunks whose stored model profile and dimensions match. Lexical retrieval uses `to_tsvector('simple', text)` and a GIN expression index; query terms are safely tokenized and OR-combined so natural-language wording can still match exact identifiers and numeric terms. Both candidate queries apply `document_id`, `source_type`, and `source_url` filters before ranking and limiting. The `/ask` and `/ask/stream` orchestration calls the hybrid retriever only inside the active page URL or selected document scope.

`PAGEGLANCE_RETRIEVAL_TOP_K` sets the default fused result count (5), and `PAGEGLANCE_RETRIEVAL_MAX_TOP_K` bounds it (50). `PAGEGLANCE_RETRIEVAL_CANDIDATE_K` controls the per-search candidate count (20), bounded by `PAGEGLANCE_RETRIEVAL_MAX_CANDIDATE_K` (200); `PAGEGLANCE_RETRIEVAL_RRF_K` sets the RRF rank constant (60). A blank query or no candidates returns an empty result. Embedding and database failures keep the existing safe webpage-context fallback and document-chat error behavior. The GIN index accelerates lexical matches; vector ranking remains exact cosine with no approximate vector index in this phase. Local MiniLM vectors use the existing fixed `vector(384)` column; legacy vectors remain unchanged.

Optional advanced retrieval is configured with `PAGEGLANCE_RERANKER_ENABLED=true`, `PAGEGLANCE_RERANKER_MODEL`, and `PAGEGLANCE_RERANKER_CANDIDATE_K` (maximum 20 candidates per rerank). It uses Sentence Transformers `CrossEncoder` locally on CPU with 512-token inputs; the first use may download the model. If the model cannot load or score, the request falls back to the RRF ranking. Keep it disabled until the model is cached if the backend must never download weights at request time. No embedding provider, API key, or database migration is needed for reranking.

Adaptive retrieval keeps simple questions on the normal single retrieval path. It can deterministically clarify a leading pronoun from the active source title or split a multi-part ask, check whether retrieved text covers each facet, and perform one bounded retry round of at most three scoped queries. The merged evidence remains capped at five chunks. This step does not call another LLM or send retrieved content to a query-rewrite service. RAG answers with retrieved sources must include known `[S#]` or applicable `[P]` references; unknown or missing references are replaced with a safe abstention. The response and SSE `sources` event include webpage and document citations; streamed grounding rejection is communicated with an additive `replace` event.

`POST /ask` also accepts `document_ids` instead of `page` or `document_id` for a bounded, explicit selection of up to five uploaded TXT/PDF documents. The backend verifies every selected ID is an uploaded-document type, applies the ID list to vector and lexical retrieval, and rejects any returned source outside that set. Existing single-document requests are unchanged. There is no per-user authentication in the current API, so this is selection scoping rather than tenant authorization; do not expose the endpoint to untrusted networks without adding an identity/ownership layer. PDF retrieval retains extracted text and page references. The current `pypdf` extraction does not build structured table cells or reliably preserve column alignment, so table-specific header/row retrieval still needs a parser-aware follow-up; OCR remains unchanged.

### Retrieval evaluation baseline

The versioned retrieval dataset is `backend/evals/rag_eval_v1.json`. It includes exact model identifiers, numeric settings, paraphrases, unsupported questions, expected chunk citations, and a webpage URL-isolation case. The runner compares vector-only top-5, lexical-only top-5, and fused top-5 using Recall@5, MRR@5, nDCG@5, expected-citation recall, unsupported candidate counts, scoped-result violations, and latency. When `PAGEGLANCE_RERANKER_ENABLED=true`, it also compares fused retrieval with the local cross-encoder reranker. It warms the existing local MiniLM model before timing, calls no Groq or hosted embedding API, and only accepts a local database named `pageglance_test`. It temporarily creates only clearly marked evaluation fixtures and removes only those fixtures it created. Results and PostgreSQL query plans are written to `backend/evals/reports/rag_eval_v1.md`.

After applying migrations, run from `backend\` in PowerShell with a test-only local connection URL:

```powershell
$env:PAGEGLANCE_TEST_DATABASE_URL = 'postgresql+psycopg://pageglance_test:replace_me@127.0.0.1:5432/pageglance_test'
python -m evals.run_retrieval_eval
Remove-Item Env:\PAGEGLANCE_TEST_DATABASE_URL
```

The evaluation suite measures retrieval, not generated-answer correctness. Its small hand-labeled corpus is a regression baseline, not a production-wide quality claim.

### Chat retrieval orchestration (Phase 8E)

`POST /ask` and `POST /ask/stream` use `RagOrchestrator` to retrieve chunks within the active page URL, or an optional `page.document_id` scope when supplied. Retrieved source IDs and metadata are included in the backend prompt, which directs the model to cite supplied sources and acknowledge insufficient evidence. Conversation history and selected active-page context remain available. If retrieval is unavailable or returns no indexed chunks, the endpoints use the existing selected-page context prompt without presenting the answer as retrieval-grounded. The API response and SSE event formats are unchanged.

### Automatic webpage indexing (Phase 9A)

Before retrieval, both chat endpoints pass the supplied page context through the existing `ingest_text` and `embed_and_persist` pipeline. Unchanged chunks reuse vectors by text hash and embedding-model profile; pages with the same normalized text at different URLs retain separate URL-scoped document records. URL normalization lowercases the scheme/host and removes default ports while preserving path, query, and fragment. Only HTTP and HTTPS URLs are accepted. Page text remains limited to 100,000 characters, indexing is capped at 128 chunks per request, and embedding batches are capped at 64. Indexing failures use the existing selected-page fallback and skip query embedding/retrieval. The opt-in PostgreSQL + pgvector integration suite verifies this against a disposable local database when configured; mocked/unit tests do not establish live database compatibility.

### Uploaded document ingestion (Phase 9C)

`POST /documents/upload` accepts one multipart field named `file` containing a UTF-8 `.txt` file or a `.pdf`. The backend validates the extension and file signature. `PAGEGLANCE_MAX_UPLOAD_MB` in the repository-root `.env` controls the per-file limit in MiB; the default is 100 MiB and accepted values are 1-512 MiB. Each multipart request is limited to that file allowance plus 256 KiB of framing overhead. The request middleware enforces the total body cap for both declared and streamed lengths, and FastAPI reads at most the configured file cap plus one byte before rejecting it. Extracted text remains limited to 100,000 characters, PDFs to 200 pages, and output to 128 chunks. It normalizes and chunks text, generates vectors with the configured embedding provider, and persists through the same PostgreSQL/pgvector repository used for webpage indexing. Duplicate non-webpage content reuses the canonical stored document and matching chunk embeddings. The response includes the document ID, safe filename, source type, content hash, chunk count, duplicate flag, and PDF page count where applicable. No original file is retained. The extension leaves size enforcement to the backend so a stale bundled limit cannot prevent uploads permitted by this setting.

Uploaded documents have `text` or `pdf` source types and no webpage URL. PDF chunk metadata includes page references for both extracted and OCR-recognized text. Text extraction remains the first choice on every page; local OCR runs only on pages without meaningful selectable text, so mixed PDFs preserve both sources and citations. OCR uses PyMuPDF, Pillow, `pytesseract`, and a locally installed Tesseract executable. It does not use a paid OCR service. OCR renders and recognizes one page at a time, reports page-batch progress in backend logs, and bounds each rendered page to 20 megapixels and 20 seconds. `PAGEGLANCE_MAX_OCR_PAGES` configures the scanned-page allowance (default 200, hard maximum 200) and `PAGEGLANCE_MAX_OCR_SECONDS` configures the total OCR time per PDF (default 900 seconds, hard maximum 3600). The existing 200-page PDF, extracted-text, and 128-chunk limits still apply. PDF uploads are read from FastAPI's spooled upload file; OCR uses a temporary disk copy made in bounded blocks, not a full PDF byte buffer in application memory. An unreadable scanned page fails explicitly instead of being omitted silently. If a PDF has no readable text after OCR, the upload is rejected.

Install the Python packages with the normal backend setup (`pip install -e ".[dev]"` from `backend\`). On Windows, install Tesseract 5 using the [Windows binaries referenced by the Tesseract project](https://tesseract-ocr.github.io/tessdoc/Downloads.html) (the project points to the [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki)). Ensure `C:\Program Files\Tesseract-OCR` is on `PATH`, then open a new PowerShell/backend process and verify it with:

```powershell
tesseract --version
```

If Tesseract is installed elsewhere, set `PAGEGLANCE_TESSERACT_CMD` in the repository-root `.env` to the full path of `tesseract.exe` (the optional setting is shown in `.env.example`) and restart FastAPI. Text-based PDFs do not require the executable; OCR support is activated only when a page needs it. Retrieval remains a separate service; callers must pass an uploaded `document_id` explicitly. Webpage chat retrieval continues to use the active normalized page URL/document scope and does not include uploaded documents.

### PostgreSQL + pgvector integration checks (Phase 8F)

The opt-in `backend/tests/test_postgres_integration.py` suite checks the Alembic schema, pgvector extension, persisted ingestion chunks and embeddings, and cosine retrieval against a real local PostgreSQL server. Embeddings are deterministic test vectors; the suite never calls OpenAI or Groq. Tests require `PAGEGLANCE_TEST_DATABASE_URL` in the process environment, and accept only `postgresql+psycopg` on localhost with a database named exactly `pageglance_test`. They do not read the application's root `.env`, change or downgrade schemas, or connect to a differently named database. A fresh database will be migrated to the current Alembic head. Tests skip when the test URL is unset or the server does not advertise pgvector.

On Windows, install the pgvector server extension matching the installed PostgreSQL major version. For the current PostgreSQL 18 installation, use an administrator **x64 Native Tools PowerShell for Visual Studio** with the C++ build tools installed:

```powershell
$env:PGROOT = 'C:\Program Files\PostgreSQL\18'
Set-Location $env:TEMP
git clone --branch v0.8.7 https://github.com/pgvector/pgvector.git
Set-Location .\pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

Restart the PostgreSQL service if needed. In pgAdmin, create a disposable database named `pageglance_test` owned by the test role, connect to it, then enable the extension as a database administrator with `CREATE EXTENSION vector;`. In a regular backend PowerShell session, set a test-only connection URL and run:

```powershell
$env:PAGEGLANCE_TEST_DATABASE_URL = 'postgresql+psycopg://pageglance_test:replace_me@127.0.0.1:5432/pageglance_test'
Set-Location backend
python -m pytest tests/test_postgres_integration.py -v
Remove-Item Env:\PAGEGLANCE_TEST_DATABASE_URL
```

Replace the placeholder credentials locally; do not commit them. The migrations also issue `CREATE EXTENSION IF NOT EXISTS vector`, so the database owner must be permitted to use the already-enabled extension. From `backend\`, `python -m pytest` includes these opt-in tests and reports them as skipped unless the test-only environment variable is set.

Follow-up context is limited by `PAGEGLANCE_HISTORY_MAX_MESSAGES` (default `10`, allowed range `0`–`20`) and `PAGEGLANCE_HISTORY_MAX_CHARS` (default `12000`, allowed range `1000`–`40000`). `POST /ask` and `POST /ask/stream` accept optional recent `history` messages; existing callers can omit the field.

`GET /health` reports service status. `POST /ask` accepts a question and bounded page context (`title`, `url`, and readable `text`), then returns `{"answer": "..."}` generated from the supplied page. `POST /ask/stream` is used by the popup to stream answer deltas. Both accept optional bounded `history` messages for follow-up questions; existing callers can omit that field. Page text is treated as untrusted data, and the model is instructed to ignore any directives embedded in it and acknowledge when the page does not support an answer. Provider credentials remain on the backend.

The popup saves conversations and drafts locally per normalized page URL. A restored answer marked as interrupted was still streaming when its popup closed; reopening does not repeat the request. Welcome suggestions submit ordinary questions through the same extraction and streaming flow.

### Automatic Windows development startup

The optional per-user Windows Task Scheduler task starts PageGlance after you sign in. From the repository root, run this once in PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-pageglance-startup.ps1
```

The installer reuses a working backend virtual environment or the currently available Python runtime when its dependencies are installed; otherwise it creates `backend\.venv` and installs the dependencies declared in `backend\pyproject.toml`. It runs `npm ci` only if the local Vite installation is missing. The task starts FastAPI, waits up to 45 seconds for `GET /health` to return `{"status":"ok"}`, then starts Vite on `127.0.0.1:5173`, waits up to 30 seconds for its development page, and opens that page in the default browser. Existing healthy services are reused. Unexpected port occupants, URL mismatches, missing dependencies, and timeouts are logged and stop startup without retrying forever.

The startup script reads `PAGEGLANCE_HOST` and `PAGEGLANCE_PORT` from the root `.env` (defaults `127.0.0.1` and `8000`). It resolves `VITE_PAGEGLANCE_API_URL` from the process environment or the extension's Vite environment files and checks that it targets the same loopback backend. When the setting is not explicitly present, it supplies the computed local API URL to that Vite process without changing any `.env` file. The Chrome extension build keeps its existing Vite configuration and URL behavior.

To start both services manually, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-pageglance.ps1`. To stop only processes started by this script, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop-pageglance.ps1`; processes without a matching recorded PID and executable are left alone. To disable automatic startup while keeping the services running, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\uninstall-pageglance-startup.ps1`. Add `-StopServices` to that command to also stop the tracked services. You can also disable or remove **PageGlance Development Services** in Windows Task Scheduler. Logs and PID records are stored under `%LOCALAPPDATA%\PageGlance\`, with separate backend/frontend output logs in its `logs` subdirectory.
>>>>>>> afd2000 (Update document ingestion and OCR handling)
