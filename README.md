# PageGlance

PageGlance is a Chrome extension that lets you ask questions about any webpage and get answers based on its content. It also supports chatting with uploaded PDF and text files.

## Features

- Ask questions about webpages and uploaded documents
- Stream responses as they're generated
- Follow-up questions with conversation history
- RAG with local embeddings and hybrid search
- Source citations for answers
- PDF text extraction and OCR for scanned pages
- PostgreSQL and pgvector for document storage and retrieval

## Tech Stack

**Frontend:** React, TypeScript, Vite, Chrome Extension API  
**Backend:** Python, FastAPI  
**AI:** Groq, Sentence Transformers, RAG  
**Database:** PostgreSQL, pgvector  
**Testing:** Pytest, ESLint

## Getting Started

### 1. Backend

From the project root, run:

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` in the project root and add your Groq API key and PostgreSQL connection details.

Start the backend:

```powershell
uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`. API documentation is available at `/docs`.

### 2. Database

Install PostgreSQL with pgvector, configure `DATABASE_URL` in `.env`, then run from the `backend` directory:

```powershell
alembic upgrade head
```

### 3. Chrome Extension

Open another terminal:

```powershell
cd extension
npm install
npm run build
```

Then open `chrome://extensions` in Chrome, enable **Developer mode**, click **Load unpacked**, and select `extension/dist`.

## Configuration

Keep API keys and database credentials in the backend's `.env` file. Never commit secrets to GitHub.

## Project Status

PageGlance is an ongoing project focused on webpage question answering, document retrieval, and practical RAG engineering.
