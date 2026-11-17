# PageGlance — Gradio SDK Space (Hugging Face Spaces)

This folder is a self-contained **Gradio SDK** Space bundle for hosting the
existing PageGlance FastAPI backend on a free Hugging Face Space, connected to
the Neon `pageglance` database from M1.

M0 verified the deployment feasibility and the Gradio/FastAPI mounting adapter.
This bundle turns that adapter into an uploadable Gradio SDK Space (no Docker
SDK, no paid infrastructure, no credit card).

## What goes in the Space

| File             | Purpose |
| ---------------- | ------- |
| `app.py`         | Entrypoint: mounts the real FastAPI app via `gr.mount_gradio_app()` and serves it on `$PORT`. |
| `requirements.txt` | Runtime dependencies (backend deps + Gradio). |
| `packages.txt`   | apt packages: `tesseract-ocr`, `libgomp1` (OCR + torch runtime). |
| `app/`           | The application package (copy of `backend/app/`). See below. |

> The `app/` package is the contents of `backend/app/`. It is the *same* code
> the backend runs locally — no RAG/embedding/schema logic is changed or copied
> into a new implementation. Because HF Spaces cannot symlink, we stage the
> package into the Space bundle for upload.

## Building the uploadable bundle

From the repository root, copy the application package into this folder:

```bash
# recreate space/app from the live backend (idempotent)
rm -rf deployment/huggingface/space/app
cp -R backend/app deployment/huggingface/space/app
rm -rf deployment/huggingface/space/app/__pycache__
```

Then upload the **contents** of `deployment/huggingface/space/` to the Space
(`app.py`, `requirements.txt`, `packages.txt`, `app/`).

## Environment variables / secrets checklist

The app reads configuration only from process environment variables (Spaces
Secrets). Set these in the Space's **Settings → Variables and secrets** (or via
`huggingface-cli`). Do not commit `.env` or real credentials.

| Variable | Required | Notes |
| -------- | -------- | ----- |
| `DATABASE_URL` | ✅ | Neon **pooled** URL (`*.pooler.neon.tech`). Never hardcode. |
| `DATABASE_SSLMODE` | ✅ | `require` (Neon requires SSL). |
| `DATABASE_POOL_SIZE` | recommended | Keep `5` (free-tier connection limit). |
| `DATABASE_MAX_OVERFLOW` | optional | Keep `10`. |
| `DATABASE_POOL_RECYCLE` | optional | Keep `300`. |
| `GROQ_API_KEY` | ✅ | LLM provider (answer generation). |
| `GROQ_MODEL` | optional | Default `openai/gpt-oss-20b`. |
| `GROQ_TIMEOUT_SECONDS` / `GROQ_MAX_COMPLETION_TOKENS` | optional | Defaults fine. |
| `PAGEGLANCE_EMBEDDING_PROVIDER` | optional | `sentence_transformers` (default, local CPU). |
| `PAGEGLANCE_LOCAL_EMBEDDING_MODEL` | optional | Default `sentence-transformers/all-MiniLM-L6-v2` (384 dim). |
| `PAGEGLANCE_TESSERACT_CMD` | optional | Defaults to `/usr/bin/tesseract` (set by `app.py`). |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | optional | LangSmith tracing; omit to disable. |
| `PAGEGLANCE_RETRIEVAL_*` / `PAGEGLANCE_CONTEXT_MAX_CHARS` | optional | Backend tuning only. |

> `PAGEGLANCE_EMBEDDING_PROVIDER=openai` would require `OPENAI_API_KEY`; the
> default local provider does not. Keep the default so embeddings stay free and
> match the existing 384-dim `pgvector` columns.

## Create the Space (free, personal)

1. Go to https://huggingface.co/new-space
   - **Name**: e.g. `pageglance-backend`
   - **License**: any (e.g. Apache 2.0)
   - **SDK**: select **Gradio** (not Docker).
   - **Hardware**: **CPU Free** (default). **Do not select a paid tier.**
   - **Python version**: **3.11** (matches `backend/pyproject.toml` and the app Dockerfile).
2. Create the Space, then add the secrets listed above (Settings → Variables and secrets).
3. Upload the bundle:
   - **Web UI**: drag-and-drop the four items (`app.py`, `requirements.txt`, `packages.txt`, `app/`) into the Space file browser, or
   - **Git**: clone the Space repo, add these files, commit, push.
4. The Space builds automatically and serves `https://huggingface.co/spaces/<user>/pageglance-backend`.

## Notable free-tier constraints

- **CPU Free**: 2 vCPU / 16 GB RAM. The local embedding model
  (`all-MiniLM-L6-v2`) is small and runs on CPU; OCR is CPU-only. Cold start
  (first request after idle) is slower because torch + the model are loaded
  lazily on first upload/embedding.
- **No apt in Gradio SDK builds**: system packages are supplied via
  `packages.txt`. If HF refuses a package, OCR on scanned PDFs degrades to the
  "Tesseract unavailable" error path; text PDFs/TXT still work.

## Verify after deploy

- `GET /health` → `{"status":"ok"}`.
- `GET /openapi.json` lists `/ask`, `/ask/stream`, `/documents/upload`, `/documents/{id}/stats`.
- `POST /documents/upload` (TXT) → document + chunk counts.
- `POST /ask` → grounded answer with citations.
- `POST /ask/stream` → SSE (`sources` → `token`* → `done`).
