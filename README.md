# 🧠 PageGlance

### An AI-Powered Browser Assistant for Context-Aware Webpage & Document Understanding

<p align="center">
  <a href="https://addons.mozilla.org/en-GB/firefox/addon/pageglance/"><strong>Firefox Add-on</strong></a> •
  <a href="https://github.com/anshh-py/PageGlance"><strong>GitHub Repository</strong></a> •
  <a href="https://ansh9057-pageglance-backend.hf.space"><strong>Backend API</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React"/>
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript"/>
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/pgvector-336791?style=for-the-badge&logo=postgresql&logoColor=white" alt="pgvector"/>
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/LangSmith-1C3C3C?style=for-the-badge" alt="LangSmith"/>
</p>

---

## Overview

**PageGlance** is an AI-powered browser assistant that enables users to understand, question, and explore information from webpages and documents through natural-language conversations.

Instead of manually searching through lengthy articles, navigating between sections, or repeatedly reading documents, users can ask questions directly about the content they are viewing.

PageGlance combines a browser extension with a Retrieval-Augmented Generation (RAG) backend to retrieve relevant context and generate answers grounded in the available source material.

The platform supports webpage question answering, PDF and TXT document ingestion, OCR-based text extraction, transcript ingestion, multi-document retrieval, and source-aware responses.

At its core, PageGlance is built around a **hybrid retrieval architecture** that combines semantic vector search with lexical full-text search, Reciprocal Rank Fusion (RRF), and optional reranking to improve the relevance of retrieved context before answer generation.

The ingestion architecture has also been extended beyond a basic webpage-and-document workflow. A standardized content model connects source-specific extraction pipelines, including webpage extraction, PDF processing, OCR, transcript parsing, and URL imports, to a common downstream indexing and retrieval workflow.

The project brings together browser-extension engineering, backend API development, document processing, vector databases, information retrieval, security hardening, and LLM-powered response generation in one integrated application.

## Live Application & Resources

| Component | Resource |
|---|---|
| Firefox Add-on | [PageGlance on Mozilla Add-ons](https://addons.mozilla.org/en-GB/firefox/addon/pageglance/) |
| Backend API | [PageGlance Backend](https://ansh9057-pageglance-backend.hf.space) |
| GitHub Repository | [anshh-py/PageGlance](https://github.com/anshh-py/PageGlance) |
| GitHub Releases | [Download Releases](https://github.com/anshh-py/PageGlance/releases) |

> **Firefox availability:** PageGlance v0.1.1 is approved, published, and available on Mozilla Add-ons.

> **Chrome distribution:** Chrome v0.1.1 has a packaged build for manual installation. It is not published on the Chrome Web Store.

> **Backend deployment:** The production FastAPI backend is deployed on Hugging Face Spaces.

---

## Key Features

### 🌐 Context-Aware Webpage Question Answering

- Ask natural-language questions about the currently active webpage.
- Extract relevant webpage content through the browser extension.
- Use page context to generate relevant, source-grounded responses.
- Avoid repeatedly copying and pasting article content into external AI tools.
- Maintain a conversational interface for exploring webpage information.
- Support webpage extraction through the common ingestion and normalization workflow.

### 📄 PDF & TXT Document Intelligence

- Upload PDF and TXT documents for AI-assisted analysis.
- Extract text from supported document formats.
- OCR-based text extraction for scanned PDF content.
- Process and chunk documents for retrieval.
- Ask questions across multiple uploaded documents.
- Retrieve relevant passages instead of sending entire documents directly to the language model.
- Preserve document and source metadata through processing.

### 🖼️ Image Understanding & OCR

- Extract readable text from image-based content through OCR.
- Process OCR output through the common content-normalization workflow.
- Retain source-related metadata for downstream retrieval.
- Support OCR-assisted processing of scanned document content.

### 🎬 Transcript & Subtitle Ingestion

- Ingest transcript-oriented content.
- Support SRT and WebVTT subtitle formats.
- Process TXT-based transcripts.
- Preserve relevant transcript timing and source information.
- Convert extracted transcript content into retrievable chunks.

### 🔗 URL Import & Extraction Fallback

- Support server-side URL ingestion.
- Validate external URLs before fetching content.
- Handle extraction failures through classified error and fallback paths.
- Keep source-specific extraction separate from common downstream indexing.
- Avoid treating every extraction failure as the same generic error.

### 🔎 Hybrid Information Retrieval

PageGlance uses multiple retrieval strategies to improve the relevance of information supplied to the generation model.

- **Semantic vector search:** Retrieves contextually similar passages using embeddings.
- **Lexical search:** Uses PostgreSQL full-text search to identify keyword-relevant content.
- **Reciprocal Rank Fusion (RRF):** Combines rankings from multiple retrieval strategies.
- **Optional reranking:** Refines retrieved candidates before context construction.
- **Context-aware retrieval:** Selects relevant document passages for answer generation.
- **Scoped retrieval:** Restricts retrieval to the selected webpage or uploaded document collection.

### 📚 Source-Grounded Answer Generation

- Generate answers using retrieved source material.
- Provide source references to help users trace relevant information.
- Keep retrieval and answer generation as separate stages.
- Support evidence-based responses rather than relying exclusively on the model's parametric knowledge.
- Validate citation references against retrieved sources.
- Preserve citation metadata for retained context.

### 🧠 Retrieval-Augmented Generation (RAG)

- Document ingestion and preprocessing pipeline.
- Standardized content representation.
- Text chunking and embedding generation.
- Persistent vector storage using PostgreSQL and pgvector.
- Hybrid retrieval and context assembly.
- Token-aware context budgeting.
- LLM-powered answer generation using retrieved context.
- Source-aware citation handling.

### 📏 Token-Aware Context Management

- Account for system instructions, conversation history, the user question, retrieved context, and reserved output tokens.
- Keep retrieved context within a bounded input budget.
- Prioritize relevant chunks instead of blindly including every retrieval result.
- Preserve complete chunk boundaries when reducing context.
- Retain source metadata and citation references for selected chunks.
- Avoid generating citations for discarded content.
- Return a controlled error when the remaining context cannot fit within the available budget.

### 🔐 Security & Reliability

- Installation authentication and owner-scoped document access.
- Shared-database rate limiting.
- SSRF protections for URL ingestion.
- Citation validation and grounded-answer checks.
- Consistent ingestion status and error handling.
- Document deletion and ownership protections.
- Duplicate-submission safeguards.
- Focused regression and compatibility coverage.

### 📊 Observability & Tracing

- LangSmith integration for tracing supported LLM workflows.
- Visibility into retrieval and generation execution.
- Easier inspection of intermediate pipeline behavior.
- Support for debugging and iterative RAG development.

### ⚡ Browser Extension Architecture

- React-based extension interface.
- TypeScript application logic.
- Vite-based development and production builds.
- Dedicated browser-specific production packaging.
- Chrome and Firefox build targets.
- Backend communication through a deployed FastAPI service.

---

## System Architecture

PageGlance follows a modular architecture that separates the browser interface, API layer, document-processing pipeline, retrieval system, and language-model generation.

```mermaid
flowchart TD
    A["User"] --> B["PageGlance Browser Extension"]
    B --> C["FastAPI Backend"]

    C --> D{"Input Type"}

    D --> E["Active Webpage"]
    D --> F["PDF / TXT Upload"]
    D --> G["Image / Scanned Content"]
    D --> H["SRT / WebVTT Transcript"]
    D --> I["URL Import"]

    E --> J["Website Extraction"]
    F --> K["Document Processing"]
    G --> L["OCR Processing"]
    H --> M["Transcript Parsing"]
    I --> N["URL Validation & Extraction Fallback"]

    J --> O["Universal Content Model"]
    K --> O
    L --> O
    M --> O
    N --> O

    O --> P["Normalization & Metadata"]
    P --> Q["Chunking"]

    Q --> R["Embedding Generation"]
    Q --> S["Lexical Indexing"]

    R --> T["PostgreSQL + pgvector"]
    S --> U["PostgreSQL Full-Text Search"]

    V["User Question"] --> W["Scoped Retrieval Pipeline"]

    T --> W
    U --> W

    W --> X["Reciprocal Rank Fusion"]
    X --> Y["Optional Reranking"]
    Y --> Z["Relevant Chunk Selection"]

    Z --> AA["Token-Aware Context Builder"]
    AA --> AB["LLM Response Generation"]
    AB --> AC["Citation Validation"]
    AC --> AD["Answer & Source References"]

    AD --> B
```

### Architecture Highlights

- **Browser Extension:** Captures webpage context and provides the conversational user interface.
- **FastAPI Backend:** Handles requests, document workflows, retrieval orchestration, and response generation.
- **Universal Content Model:** Provides a common representation for different extracted content types.
- **Document Processing:** Extracts and prepares content from webpages, PDFs, TXT files, images, and supported transcript formats.
- **OCR Processing:** Converts image-based text into content that can participate in retrieval.
- **Transcript Processing:** Parses supported subtitle and transcript files while retaining relevant timing information.
- **URL Import & Fallback:** Handles external URL ingestion, validation, and extraction failure paths.
- **Embedding Pipeline:** Converts text chunks into vector representations for semantic retrieval.
- **PostgreSQL + pgvector:** Stores document data, chunks, and vector embeddings.
- **Lexical Retrieval:** Uses PostgreSQL full-text search to retrieve keyword-relevant passages.
- **RRF:** Combines results from semantic and lexical retrieval.
- **Reranking:** Optionally improves the ordering of retrieved candidates.
- **Context Builder:** Selects relevant complete chunks within a token-aware context budget.
- **LLM Generation:** Produces responses using the retrieved context.
- **Citation Validation:** Checks generated source references against available retrieved sources.
- **LangSmith:** Supports tracing and inspection of AI workflows.

---

## Retrieval-Augmented Generation Pipeline

The RAG pipeline is the central component of PageGlance.

Rather than passing an entire webpage or document directly to a language model, PageGlance processes the source material, retrieves relevant passages, and constructs bounded context for answer generation.

```mermaid
flowchart TD
    A["Webpage / PDF / TXT / OCR / Transcript"] --> B["Source-Specific Extraction"]
    B --> C["Universal Content Model"]
    C --> D["Preprocessing & Metadata"]
    D --> E["Text Chunking"]

    E --> F["Embedding Generation"]
    E --> G["Lexical Indexing"]

    F --> H["Vector Storage"]
    G --> I["Full-Text Index"]

    J["User Query"] --> K["Query Processing"]
    K --> L["Scoped Retrieval"]

    H --> L
    I --> L

    L --> M["Reciprocal Rank Fusion"]
    M --> N["Optional Reranking"]
    N --> O["Relevant Candidate Selection"]

    O --> P["Token-Aware Context Budget"]
    P --> Q["Complete Chunk Selection"]
    Q --> R["Prompt Construction"]

    R --> S["LLM"]
    S --> T["Citation Validation"]
    T --> U["Grounded Answer"]
    U --> V["Source References"]
```

### Retrieval Strategy

| Stage | Technique | Purpose |
|---|---|---|
| Preprocessing | Text extraction and normalization | Prepare source content |
| Content Representation | Universal Content Model | Standardize extracted content |
| Chunking | Document segmentation | Create retrievable passages |
| Embeddings | Dense vector representations | Capture semantic similarity |
| Semantic Retrieval | pgvector similarity search | Find contextually relevant chunks |
| Lexical Retrieval | PostgreSQL full-text search | Find keyword-relevant chunks |
| Fusion | Reciprocal Rank Fusion | Combine retrieval rankings |
| Reranking | Optional reranker | Refine candidate relevance |
| Context Selection | Relevance-based chunk selection | Select useful source passages |
| Context Budgeting | Token-aware budget management | Keep the prompt within model limits |
| Generation | LLM-based synthesis | Produce an answer from retrieved context |
| Validation | Citation validation | Check references against retrieved sources |

### Why Hybrid Retrieval?

Vector search and lexical search solve different retrieval problems.

Semantic search can retrieve conceptually related passages even when the wording differs. Lexical search can identify exact terminology, names, technical phrases, and keywords.

Combining both strategies through RRF provides a retrieval architecture that uses complementary relevance signals instead of depending on a single search method.

### Context Budgeting & Citation Preservation

Retrieval results must fit within the language model's available input budget.

PageGlance accounts for the prompt instructions, conversation history, user question, retrieved context, and reserved output tokens when constructing the final model input.

When the available budget is insufficient for all retrieved content, the context builder prioritizes relevance and selects complete chunks rather than silently truncating arbitrary portions of the context.

Source metadata and citation references remain associated with retained chunks. Content excluded from the final context must not produce citations in the generated answer.

---

## Multi-Document Retrieval

PageGlance supports querying information across multiple uploaded documents.

The document workflow includes:

1. Uploading supported PDF and TXT files.
2. Extracting document text, including OCR-based extraction where applicable.
3. Normalizing extracted content and retaining source metadata.
4. Splitting content into manageable chunks.
5. Generating embeddings for retrieval.
6. Persisting document and chunk information.
7. Retrieving relevant passages across the available document collection.
8. Selecting relevant context within the model's token budget.
9. Generating answers from retrieved context.
10. Returning source references associated with the selected evidence.

This makes PageGlance useful for exploring collections of notes, reports, technical documents, and other text-based resources.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Languages | Python, TypeScript, JavaScript |
| Browser Extension | React, TypeScript |
| Frontend Build Tool | Vite |
| Backend Framework | FastAPI |
| API Architecture | REST APIs |
| Database | PostgreSQL |
| Vector Search | pgvector |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Semantic Retrieval | Dense vector retrieval |
| Lexical Retrieval | PostgreSQL Full-Text Search |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Optional reranking pipeline |
| Document Processing | PDF and TXT extraction, OCR |
| Transcript Processing | SRT, WebVTT and TXT |
| AI Integration | Groq LLM API |
| Observability | LangSmith |
| RAG Evaluation | Ragas |
| Database Migrations | Alembic |
| Backend Hosting | Hugging Face Spaces |
| Database Hosting | Neon PostgreSQL |
| Version Control | Git, GitHub |

---

## Database & Vector Storage

PageGlance uses PostgreSQL as its persistent data layer, with pgvector providing vector similarity search.

The database supports the storage and retrieval of document-related information, including text chunks and their corresponding embeddings.

### Database Responsibilities

- Persistent document storage.
- Text chunk management.
- Vector embedding storage.
- Semantic similarity retrieval.
- PostgreSQL full-text search.
- Source and metadata associations.
- Document ownership and lifecycle operations.
- Schema evolution through Alembic migrations.

### Vector Retrieval

Embeddings represent text chunks as numerical vectors. When a user submits a question, the retrieval pipeline uses vector similarity to identify passages that are semantically related to the query.

The vector retrieval results are combined with lexical retrieval results to create a more comprehensive candidate set.

---

## Backend API

PageGlance uses FastAPI to expose backend functionality through HTTP APIs.

The backend handles communication between the browser extension and the RAG pipeline, including content processing, document ingestion, retrieval, and answer generation.

**Backend:** [PageGlance FastAPI Service](https://ansh9057-pageglance-backend.hf.space)

API documentation, where enabled:

- `/docs` — Swagger UI
- `/openapi.json` — OpenAPI schema

### Backend Responsibilities

- Browser extension communication.
- Webpage content processing.
- PDF and TXT ingestion.
- OCR-based document processing.
- Transcript parsing.
- URL import and extraction fallback.
- Embedding generation.
- Hybrid retrieval orchestration.
- Token-aware context construction.
- LLM response generation.
- Source-aware response handling.
- Authentication and owner isolation.
- Document lifecycle management.

---

## Project Structure

PageGlance separates browser-extension code, backend services, and supporting infrastructure.

```text
PageGlance/
│
├── extension/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── ...
│   │
│   ├── public/
│   ├── manifests/
│   │   ├── chrome.json
│   │   └── firefox.json
│   │
│   ├── scripts/
│   │   ├── pack-firefox.mjs
│   │   ├── verify-firefox-zip.mjs
│   │   ├── pack-chrome.mjs
│   │   └── verify-chrome-zip.mjs
│   │
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── retrieval/
│   │   ├── ingestion/
│   │   ├── database/
│   │   └── ...
│   │
│   ├── alembic/
│   ├── evals/
│   ├── tests/
│   └── requirements.txt
│
├── deployment/
│   └── huggingface/
│
├── pageglance_backend/
│
├── README.md
└── ...
```

> The directory tree is a high-level representation. Refer to the repository for the current implementation and exact module names.

---

## Getting Started

Follow these steps to set up PageGlance locally.

### Prerequisites

- Python 3.10+
- Node.js and npm
- PostgreSQL with pgvector
- Groq API key
- Tesseract OCR (for scanned PDFs)
- A Chromium-based browser or Firefox for extension testing

### 1. Clone the Repository

```bash
git clone https://github.com/anshh-py/PageGlance.git
cd PageGlance
```

### 2. Set Up the Backend

Create a virtual environment:

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Configure Backend Environment Variables

Create a local environment file using the repository's example configuration, if provided.

Configure the required values for:

- PostgreSQL database connection.
- LLM provider credentials.
- Embedding model/provider configuration.
- Any optional observability integrations.

Use the actual variable names documented in the backend configuration and environment example.

Never commit `.env` files, API keys, database passwords, or other secrets.

### 4. Configure the Database

Ensure PostgreSQL and pgvector are available.

Run the project's Alembic migrations using the repository's configured migration workflow.

```bash
cd backend
alembic upgrade head
```

### 5. Start the Backend

From the backend directory, use the application's configured FastAPI entry point:

```bash
uvicorn app.main:app --reload
```

The local API will be available at:

```text
http://127.0.0.1:8000
```

### 6. Set Up the Browser Extension

Open a separate terminal:

```bash
cd extension
npm ci
```

Configure the extension's production or development API URL using the documented Vite environment variable:

```text
VITE_PAGEGLANCE_API_URL
```

For local backend development, the value should point to:

```text
http://127.0.0.1:8000
```

Keep private environment files out of version control.

### 7. Build the Extension

Use the browser-specific build scripts defined in `extension/package.json`.

For Firefox:

```bash
npm run build:firefox
npm run pack:firefox
npm run verify:firefox
```

For Chrome:

```bash
npm run build:chrome
npm run pack:chrome
npm run verify:chrome
```

### 8. Load the Extension Locally

**Chrome / Chromium**

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Select Load unpacked.
4. Select the generated Chrome build directory.

**Firefox**

1. Open `about:debugging#/runtime/this-firefox`.
2. Select Load Temporary Add-on.
3. Select the generated Firefox `manifest.json`.

---

## Deployment Architecture

PageGlance uses a separated deployment architecture for the browser client, API service, and persistent database.

| Component | Platform | Responsibility |
|---|---|---|
| Browser Extension | Chrome / Firefox | User interface and webpage context |
| Backend API | Hugging Face Spaces | FastAPI application and RAG orchestration |
| Database | Neon PostgreSQL | Persistent data and vector storage |
| Vector Search | pgvector | Semantic retrieval |
| Observability | LangSmith | LLM workflow tracing |

**Production backend:**  
https://ansh9057-pageglance-backend.hf.space

The browser extension communicates with the deployed backend through its configured API endpoint. Database operations and retrieval orchestration remain on the backend.

---

## Engineering Highlights

PageGlance demonstrates practical implementation of several software engineering and AI concepts:

- Designing a browser extension with React and TypeScript.
- Building REST APIs using FastAPI.
- Designing a universal content model for heterogeneous source formats.
- Implementing source-specific webpage and document extraction.
- Building a multi-format document ingestion and preprocessing pipeline.
- Working with PDF/TXT extraction and OCR.
- Supporting transcript-oriented ingestion using SRT and WebVTT.
- Implementing URL import and extraction fallback handling.
- Designing a Retrieval-Augmented Generation architecture.
- Implementing semantic vector retrieval using pgvector.
- Integrating PostgreSQL full-text search.
- Combining retrieval rankings using Reciprocal Rank Fusion.
- Integrating optional reranking into the retrieval pipeline.
- Building multi-document retrieval workflows.
- Implementing token-aware context construction.
- Preserving complete chunk boundaries and citation metadata.
- Generating source-grounded responses from retrieved context.
- Validating citation references against retrieved evidence.
- Managing persistent vector data with PostgreSQL.
- Handling database schema evolution with Alembic.
- Implementing installation authentication and owner isolation.
- Applying URL ingestion security and rate limiting.
- Integrating LangSmith for observability and tracing.
- Building and packaging browser-specific extension releases.
- Deploying a cloud-hosted backend service.
- Adding focused regression and compatibility coverage.

---

## Engineering Milestones

The following ten milestones represent the completed engineering improvement cycle of PageGlance.

| Milestone | Engineering Area | Implementation Improvements |
|---|---|---|
| M1 | Universal Content Model & Ingestion Interface | Standardized content representation, ingestion contracts, metadata and processing results. |
| M2 | Universal Website Extraction | Webpage extraction integrated with the common content-processing workflow. |
| M3 | Universal PDF Engine | PDF extraction and processing integrated into the ingestion pipeline. |
| M4 | Universal Document Import | Unified document import, parsing and downstream indexing. |
| M5 | Image Understanding & OCR | OCR-based text extraction and integration of image-derived content into retrieval. |
| M6 | Video, Audio & Media Transcripts | Transcript ingestion, SRT and WebVTT parsing, and preservation of relevant source timing. |
| M7 | Universal Fallback & URL Import Engine | URL validation, extraction failure classification and fallback handling. |
| M8 | Universal Ingestion Interface | Consistent ingestion states, progress reporting, browser-facing behavior and duplicate-submission handling. |
| M9 | Security, Privacy & Compatibility Hardening | Authentication, owner isolation, URL security, document lifecycle protections and compatibility safeguards. |
| M10 | Universal Compatibility Test Suite & Release | Focused regression coverage, compatibility verification and browser release preparation. |

### Milestone Completion

**10/10 planned milestones completed.**

The engineering work improved the complete content-to-answer lifecycle rather than simply adding more supported file extensions.

The main improvements were:

- Moving from source-specific processing toward a standardized ingestion contract.
- Separating extraction logic from downstream indexing and retrieval.
- Connecting webpage, document, OCR and transcript processing to a common workflow.
- Improving extraction failure classification and URL fallback behavior.
- Standardizing ingestion states and browser-facing processing behavior.
- Strengthening document ownership, authentication and URL security.
- Improving retrieval context construction and citation preservation.
- Adding compatibility and regression coverage across the supported workflows.
- Preparing separate browser-specific production packages.

---

## Release & Distribution

PageGlance provides browser-specific production builds.

| Browser | Distribution | Status |
|---|---|---|
| Chrome | [GitHub Releases](https://github.com/anshh-py/PageGlance/releases) | v0.1.1 packaged for manual installation |
| Firefox | [Mozilla Add-ons Listing](https://addons.mozilla.org/en-GB/firefox/addon/pageglance/) | v0.1.1 published and available |

### Chrome

Chrome v0.1.1 has a packaged build with browser-specific manifest configuration and ZIP verification scripts.

The package can be distributed through GitHub Releases and installed manually using Chrome's Developer mode.

Chrome Web Store publication is not claimed.

### Firefox

Firefox v0.1.1 has been approved and published on Mozilla Add-ons.

Users can install it directly from the public listing.

For versioned builds and release notes, visit the [GitHub Releases](https://github.com/anshh-py/PageGlance/releases) page.

---

**Connect:**

- GitHub: [@anshh-py](https://github.com/anshh-py)
- Project: [PageGlance](https://github.com/anshh-py/PageGlance)

---

<p align="center">
  <strong>PageGlance — Understand more. Search less. Ask better questions.</strong>
</p>

---

# PageGlance — Technical & Engineering Reference

# PageGlance

**Chrome and Firefox extension + FastAPI backend for asking questions about webpages and uploaded documents with retrieval-grounded answers and citations.**

PageGlance is a personal portfolio project that combines browser extensions with a FastAPI backend to perform scoped RAG over the active webpage or uploaded PDFs/TXT files.

It uses local embeddings, PostgreSQL + pgvector, hybrid retrieval (vector + PostgreSQL FTS) with Reciprocal Rank Fusion, optional cross-encoder reranking, Groq for generation, and source-grounding with citations.

The ingestion system additionally supports standardized content processing, OCR, transcript ingestion, URL import, and fallback handling. The retrieval pipeline includes bounded context construction and source metadata preservation.

## Problem it solves

Users want to ask questions about content they're looking at without manually searching or copying context. PageGlance indexes the active webpage or uploaded documents into a scoped knowledge base and returns answers grounded in retrieved evidence with citations.

## Key features

- **Scoped chat**: Ask about the active webpage or explicitly selected uploaded document(s) (single or up to 5 documents).
- **Streaming responses**: `/ask/stream` returns Server-Sent Events (`sources`, `token`, `replace`, `done`, `error`).
- **RAG with grounding**: Answers are validated against retrieved sources; missing/invalid citations trigger abstention when evidence is insufficient.
- **Hybrid retrieval**: Vector similarity (pgvector) + PostgreSQL full-text search (simple config) fused with Reciprocal Rank Fusion (RRF). Optional cross-encoder reranking.
- **Local embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) by default; legacy OpenAI embeddings path supported via configuration.
- **Universal ingestion**: Standardized content representation and common downstream processing for supported source types.
- **Website extraction**: Webpage content extraction through the browser and ingestion workflow.
- **Document ingestion**: Upload TXT/PDF with OCR for scanned PDFs (PyMuPDF + pytesseract), chunking with overlap, bounded limits.
- **Image OCR**: OCR-derived text can enter the common content-processing workflow.
- **Transcript ingestion**: SRT, WebVTT and TXT transcript processing.
- **URL import and fallback**: Server-side URL ingestion with validation, failure classification and extraction fallback handling.
- **Context budgeting**: Token-aware prompt construction that accounts for instructions, conversation history, question, retrieved context and output capacity.
- **Citation preservation**: Source metadata remains associated with retained chunks; discarded context must not produce citations.
- **Observability**: LangSmith tracing integrated (opt-in via env vars).
- **Evaluation**: Retrieval evaluation suite and Ragas evaluation scaffolding for the four metrics.
- **Installation auth & owner isolation**: Tokens registered via `/auth/register` (only hashes are stored); every data route requires a bearer token and documents are scoped per installation.
- **Rate limiting**: Fixed-window counters stored in the shared database (no Redis dependency), with per-installation quotas and an IP backstop; fails closed (`503`) if the counter store is unreachable.
- **Security protections**: SSRF-guarded URL import, strict citation validation with abstention, and XSS-safe Markdown rendering in the extension.
- **Browser distribution**: Separate Chrome and Firefox build targets with browser-specific packaging and verification scripts.

## Architecture

```mermaid
flowchart TD
    A["Browser Extension<br/>React + TypeScript + Vite"] -->|HTTP POST /ask or /ask/stream| B["FastAPI Backend<br/>app.main"]

    B --> C["Scope Resolution<br/>page / document / documents"]
    C --> D["Content Ingestion & Retrieval"]

    D --> E["Website Extraction"]
    D --> F["PDF / TXT Processing"]
    D --> G["Image OCR"]
    D --> H["Transcript Parsing"]
    D --> I["URL Import & Fallback"]

    E --> J["Universal Content Model"]
    F --> J
    G --> J
    H --> J
    I --> J

    J --> K["Normalization & Metadata"]
    K --> L["Chunking"]
    L --> M["Embeddings"]
    L --> N["Lexical Indexing"]

    M --> O[(PostgreSQL + pgvector)]
    N --> O

    O --> P["Hybrid Retrieval<br/>Vector + FTS"]
    P --> Q["RRF Fusion"]
    Q --> R{Reranker enabled?}
    R -->|yes| S["Cross-Encoder Rerank"]
    R -->|no| T["Relevant Chunk Selection"]
    S --> T

    T --> U["Token-Aware Context Builder"]
    U --> V["Groq LLM Generation"]
    V --> W["Citation Validation"]
    W --> X["Answer + Source References"]
    X --> A
```

## Technology stack (actual)

| Layer | Technologies |
|---|---|
| Extension | React, TypeScript, Vite, browser-specific manifests |
| Backend API | FastAPI, Pydantic, Uvicorn |
| LLM | Groq API (via `groq` SDK), default model `openai/gpt-oss-20b` |
| Auth | Installation bearer tokens, token hashes, per-owner scoping |
| Rate limiting | Shared-DB fixed-window counters, fail-closed |
| Deployment | Hugging Face Spaces (Gradio SDK bundle), FastAPI mounted via `gr.mount_gradio_app()` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| Retrieval | pgvector cosine similarity, PostgreSQL full-text search (simple regconfig), RRF, optional `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Storage | PostgreSQL, SQLAlchemy 2.x, pgvector, Alembic |
| Document processing | PyPDF/pypdf, PyMuPDF, pytesseract, Pillow |
| Transcript processing | SRT, WebVTT and TXT parsing |
| Tracing/Eval | LangSmith SDK, Ragas, datasets, pytest |

## How the extension and backend communicate

- The browser extension sends JSON to the FastAPI backend over HTTP (same-origin/CORS configured in `app.main`; Chrome/Firefox extension origins are allowed).
- Endpoints: `POST /ask` (JSON response with `answer` and optional `sources`), `POST /ask/stream` (SSE). Additional endpoints support document upload, stats, health and deletion.
- Requests include `question`, scope (`page`, `document_id`, or `document_ids`), and optional `history`. The backend scopes retrieval to the provided scope and returns citations referencing retrieved chunks/sources.
- Every ingestion/retrieval/stats route is authenticated with the installation's bearer token (`Authorization: Bearer <token>`), registered once via `POST /auth/register`.

## RAG workflow (as implemented)

1. **Ingestion/indexing**: Page content is extracted by the extension or uploaded files are ingested via `app.document_ingestion`. Source-specific extraction and normalization feed the common content-processing workflow. Text is chunked (`app.chunking`), embedded (`app.embedding_pipeline`), and stored in Postgres with pgvector columns (local 384-dim profile for sentence-transformers).
2. **Query transformation**: Adaptive query splitting/retry logic in `app.rag` for multi-part questions or pronoun resolution.
3. **Retrieval**: `app.retrieval.VectorRetriever` performs vector search and/or lexical (FTS) search with scope filters applied first. Hybrid retrieval runs both paths and fuses via RRF (`fuse_rankings`). Optional reranking with cross-encoder when enabled.
4. **Context assembly**: `app.rag.RagOrchestrator` merges results and builds `retrieved_sources` with citation IDs (`S1`, `S2`, ...). The context builder applies a token-aware budget across prompt instructions, conversation history, the user question, retrieved context and reserved output capacity. It prioritizes relevant complete chunks and preserves their citation metadata.
5. **Generation**: `app.llm` builds grounded system/user messages and calls Groq Chat Completions. Streaming via `stream_answer()` with async generator.
6. **Citations & validation**: `app.main._validate_grounded_answer` ensures factual claims cite valid retrieved source IDs (or `P` for page) when RAG evidence is used; returns abstention if evidence is insufficient. Citation references must correspond to retained context, not discarded chunks.

## API endpoints (actual)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/documents/{document_id}/stats` | Indexed chunk stats for a document (no full text) |
| `POST` | `/documents/upload` | Upload TXT/PDF for ingestion/indexing |
| `POST` | `/ask` | Synchronous answer with citations |
| `POST` | `/ask/stream` | SSE streaming answer (`sources`, `token`, `replace`, `done`, `error`) |
| `POST` | `/auth/register` | Register an installation token (only its hash is stored), idempotent |
| `POST` | `/ingest/url` | Fetch and index a server-side URL (SSRF-guarded) |
| `DELETE` | `/documents/{document_id}` | Delete an owned document and its chunks/embeddings |

See `backend/app/main.py` for request/response schemas and validation.

## Authentication and owner isolation

- Each extension installation self-registers a token once via `POST /auth/register`; the backend stores **only a hash** of the token (`pga1:...`), never the raw token.
- All ingestion, retrieval, stats and deletion routes require the bearer token (`Authorization: Bearer <token>`); missing credentials → `401`, invalid → `403`.
- Documents are scoped by an internal `owner_id` derived from the installation. A request for another installation's document returns `404` (existence is never disclosed).
- Legacy ownerless documents (pre-auth rows with `owner_id = NULL`) remain unclaimed and are **not** reachable through authenticated routes; registration never backfills or claims them.

## Rate limiting

- Fixed-window counters are stored in the shared application database (no Redis dependency), so limits are consistent across workers and restarts on a free Space.
- Per-installation quotas are enforced on `/ask`, `/ask/stream`, `/documents/upload`, `/documents/{id}/stats` and `/ingest/url`, with a shared IP backstop on registration.
- The limiter is **fail-closed**: if the counter store is unreachable the request is refused with `503` rather than allowed through unlimited.

## Security protections

- **URL import (SSRF):** `/ingest/url` validates the URL scheme (HTTP/HTTPS only), resolves hostnames and rejects private-network/loopback targets before fetching; the fetched content is only used as evidence.
- **Citation validation:** Answers are validated against retrieved source IDs; missing/unknown citations trigger a safe abstention instead of unsourced claims.
- **XSS-safe rendering:** The extension renders model Markdown as React elements (no `innerHTML` / `dangerouslySetInnerHTML`), so raw text is escaped by the framework.
- **Provider safety:** Bounded concurrency, timeouts, and safe error mapping for the hosted LLM; streaming cancellation closes the provider stream.
- **Ingestion safety:** External URL ingestion and document operations are subject to validation and ownership controls.

## Deployment (Hugging Face Spaces)

The backend deploys to a **Gradio SDK** Space (free CPU tier, no Docker), mounting the unchanged FastAPI app via `gr.mount_gradio_app()`.

Two bundles exist:

- `pageglance_backend/` — the ready-to-upload Space bundle (entry point `app.py`, `requirements.txt`, `packages.txt`, `app/`).
- `deployment/huggingface/space/` — the same bundle maintained alongside the adapter sources.

Deploy by uploading the bundle contents to the Space repo and setting **Secrets** for the required variables (names only, never commit values):

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes | Production PostgreSQL connection (provider-agnostic) |
| `DATABASE_SSLMODE` | conditional | `require` for SSL-only providers |
| `DATABASE_POOL_SIZE` / `DATABASE_MAX_OVERFLOW` / `DATABASE_POOL_RECYCLE` | no | Pool tuning |
| `GROQ_API_KEY` | yes | LLM provider key |
| `GROQ_MODEL` | no | Default `openai/gpt-oss-20b` |
| `PAGEGLANCE_EMBEDDING_PROVIDER` | no | `sentence_transformers` (default, local CPU) |
| `PAGEGLANCE_LOCAL_EMBEDDING_MODEL` | no | Default `sentence-transformers/all-MiniLM-L6-v2` |
| `PAGEGLANCE_TESSERACT_CMD` | no | Linux default `/usr/bin/tesseract` |
| `PORT` | no | Spaces sets this (default `7860`) |

Full step-by-step staging and secret guidance is in `pageglance_backend/README.md` and `deployment/huggingface/README.md`.

## LangSmith tracing

Tracing is integrated via the standalone `langsmith` SDK. It is **opt-in** and **disabled** if not configured.

**Environment variables:**

- `LANGSMITH_TRACING` (true/false)
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT` (default `pageglance`)
- `LANGSMITH_ENDPOINT` (optional)

**Configuration:** `backend/app/langsmith_tracing.py` configures tracing and provides helpers; `backend/app/main.py` adds tracing hooks around request/LLM/citation steps. When disabled or credentials are missing, tracing is a no-op with no secrets logged.

## Ragas evaluation

**Metrics (configured):** Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall (using `ragas` with Groq LLM and PageGlance embeddings).

**Dataset:** `backend/evals/rag_eval_v2.json` (version `rag-eval-v2`, 22 cases, 16 corpus entries). Do not replace this file.

**Scripts:** `backend/evals/run_ragas_eval.py` (uses actual pipeline via `app.evaluation_interface.capture_rag_sample()`), retrieval eval in `backend/evals/run_retrieval_eval.py`.

**Running evaluation (example):**

```bash
cd backend
# set GROQ_API_KEY in env
.\.venv\Scripts\python.exe -c "from evals import run_ragas_eval; r,j = run_ragas_eval.run(max_cases=3, exclude_unsupported=True, report_prefix='ragas_eval_3cases'); print(r['metrics'])"
```

Reports written to `backend/evals/reports/` with distinct filenames for limited runs.

**Important:** Mocked tests demonstrate plumbing; live evaluation results depend on API availability/rate limits. This README does **not** claim specific benchmark scores - any real results should be reported from actual runs only.

## Installation and local setup (Windows)

### Prerequisites

- Python 3.10+
- Node.js/npm
- PostgreSQL with pgvector
- Groq API key
- Tesseract OCR (for scanned PDFs)

### Backend setup

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
cp ../.env.example ../.env  # configure as needed
# Run migrations
alembic upgrade head
# Start server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Or use Start_PageGlance.bat from repo root
```

### Extension setup

```bash
cd extension
npm install
npm run dev  # or build
# Load unpacked extension in Chrome or Firefox using the corresponding build output
```

### Chrome packaging and verification

```bash
npm run build:chrome
npm run pack:chrome
npm run verify:chrome
```

### Environment variables

Copy `.env.example` to `.env` and configure. See `.env.example` for all variables (keys redacted in examples). Never commit real API keys.

## Repository structure (actual, excluding venv/caches/secrets)

```text
pageglance/
+-- backend/
¦   +-- app/            # FastAPI app, RAG, retrieval, LLM, embeddings, ingestion
¦   +-- alembic/        # migrations
¦   +-- evals/          # evaluation datasets and runners
¦   +-- tests/          # pytest tests
¦   +-- pyproject.toml
+-- extension/          # React/TS/Vite browser extension
¦   +-- manifests/      # Browser-specific manifests
¦   +-- scripts/        # Browser packaging and verification
+-- deployment/         # Hugging Face Space deployment adapter + bundle
+-- pageglance_backend/ # Ready-to-upload Gradio SDK Space bundle
+-- Start_PageGlance.bat
+-- .env.example
+-- README.md
```

## Engineering Milestones — M1 to M10

The ten completed milestones expanded PageGlance from its original webpage/document RAG workflow into a more unified content-ingestion, retrieval, security and release pipeline.

| Milestone | Scope | Engineering outcome |
|---|---|---|
| M1 | Universal Content Model & Ingestion Interface | Common content representation and ingestion contract for source-specific extractors. |
| M2 | Universal Website Extraction | Webpage extraction connected to the standardized processing workflow. |
| M3 | Universal PDF Engine | PDF extraction and processing integrated into document ingestion. |
| M4 | Universal Document Import | Unified document import and downstream indexing workflow. |
| M5 | Image Understanding & OCR | OCR extraction and integration of image-derived text into retrieval. |
| M6 | Video, Audio & Media Transcripts | SRT, WebVTT and TXT transcript ingestion with source timing information. |
| M7 | Universal Fallback & URL Import Engine | URL validation, failure classification and extraction fallback handling. |
| M8 | Universal Ingestion Interface | Consistent processing states, progress reporting and browser-facing behavior. |
| M9 | Security, Privacy & Compatibility Hardening | Authentication, owner isolation, URL security, document lifecycle and compatibility safeguards. |
| M10 | Universal Compatibility Test Suite & Release | Regression coverage and browser release preparation. |

**Completion status: 10/10 milestones reported completed.**

## Known limitations

- Portfolio/portfolio-grade project: production readiness is only as good as the verified live database, LLM and deployment; this README does **not** claim benchmark scores, traffic or uptime measurements.
- Groq API rate limits can affect live evaluation/generation.
- OCR requires local Tesseract installation (system package on Spaces via `packages.txt`).
- Local embeddings model is downloaded on first use (cold start cost).
- Some features depend on PostgreSQL + pgvector being available and migrated.
- Production backup/migration and live end-to-end verification require operator access to the production database and the Hugging Face Space credentials.
- Chrome Web Store publication is not included; Chrome distribution uses a manually installable package.

## Future improvements

- Expand evaluation coverage.
- Improve error handling and observability.
- Additional retrieval strategies.
- Better document type support.
- Expand browser distribution and release automation.
- Further optimize extraction and retrieval for complex content sources.
