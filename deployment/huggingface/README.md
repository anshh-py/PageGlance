# PageGlance — Hugging Face Space Deployment Adapter

This folder contains an isolated deployment adapter that hosts the existing
PageGlance FastAPI backend on Hugging Face Spaces using Gradio's **official**
`gr.mount_gradio_app()` integration.

## Why this adapter

- It imports the **real** application (`app.main`) — no RAG logic is duplicated,
  no response contracts change, no embedding model / providers are swapped.
- The FastAPI app (routes, middleware, CORS, SSE) is preserved as-is; Gradio is
  mounted **on top** of it.
- It is intentionally separate from `backend/` so application dependencies are
  not polluted by deployment-only packages.

## Gradio SDK Space (recommended, free)

To deploy on a **free** Hugging Face Space (no Docker SDK, no paid hardware), use
[`space/`](./space/README.md). `space/` is a self-contained uploadable bundle:

- `space/app.py` — entrypoint that mounts the real app and serves it on `$PORT`
  (also runnable directly as `python app.py`).
- `space/requirements.txt` — runtime dependencies (backend deps + Gradio).
- `space/packages.txt` — apt packages (`tesseract-ocr`, `libgomp1`).
- `space/app/` — the application package staged from `backend/app/`.

Follow [`space/README.md`](./space/README.md) for the secrets checklist and the
exact Space creation steps. The Docker-based flow below remains as an alternative
for a Docker SDK Space.

## Files

| File               | Purpose                                                                       |
| ------------------ | ----------------------------------------------------------------------------- |
| `app.py`           | Mounts `app.main:app` inside a Gradio Blocks app via `gr.mount_gradio_app()`. |
| `requirements.txt` | Deployment-only requirements (Gradio) added to the app requirements.          |
| `README.md`        | This document.                                                                |

## Environment variables

The application reads its existing configuration from the repository-root `.env`
(e.g. `DATABASE_URL`, `PAGEGLANCE_EMBEDDING_PROVIDER`, `GROQ_API_KEY`,
`PAGEGLANCE_TESSERACT_CMD`). The adapter only ensures the Linux Tesseract path is
used by default (`PAGEGLANCE_TESSERACT_CMD` defaults to `/usr/bin/tesseract`).

| Variable                   | Required             | Purpose                                                            |
| -------------------------- | -------------------- | ------------------------------------------------------------------ |
| `PORT`                     | no (Spaces sets it)  | HTTP port; defaults to `7860`.                                     |
| `PAGEGLANCE_TESSERACT_CMD` | no                   | Tesseract binary path (defaults to `/usr/bin/tesseract` on Linux). |
| `DATABASE_URL`             | yes (if DB used)     | PostgreSQL connection string (local or Neon).                      |
| `DATABASE_SSLMODE`         | no (Neon: `require`) | Driver SSL mode; required for Neon.                                |
| `DATABASE_POOL_SIZE`       | no                   | Connection pool size (default 5).                                  |
| `DATABASE_MAX_OVERFLOW`    | no                   | Max overflow connections (default 10).                             |
| `DATABASE_POOL_RECYCLE`    | no                   | Connection recycle seconds (default 300).                          |

> The `.env.example` template lives in the repository root (`.env.example`). Because
> the tooling treats `.env*` paths as protected secrets, the full template is also
> reproduced below for reference.

```bash
# env template (placeholders only — copy to `.env` at the repo root)
DATABASE_URL=postgresql://<user>:<password>@<host>.neon.tech:5432/<dbname>
DATABASE_SSLMODE=require
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_RECYCLE=300
GROQ_API_KEY=
PAGEGLANCE_TESSERACT_CMD=/usr/bin/tesseract
```

## Neon PostgreSQL (M1)

PageGlance connects to either a local PostgreSQL instance or a Neon cloud database
through the same `DATABASE_URL` mechanism — only the environment changes.

- **Connection**: use the Neon **pooled** connection string (host `*.pooler.neon.tech`)
  for application traffic; set `DATABASE_SSLMODE=require`. The pool size environment
  variables tune SQLAlchemy's `QueuePool` for the free-tier connection limit
  (default 5 active connections is appropriate).
- **Alembic migrations**: `backend/migrations/env.py` already reads the same
  `DATABASE_URL` (via `app.db.get_database_url()`); run `alembic upgrade head`
  with the Neon URL to apply migrations. Migrations use the **direct** connection
  string (non-pooled) to avoid pgbouncer transaction-mode limitations during DDL.
- **pgvector**: the first migration runs `CREATE EXTENSION IF NOT EXISTS vector`
  (supported on Neon). Vector columns are `Vector(384)`; the GIN FTS index and
  cosine-distance queries are fully supported.
- **Local stays intact**: when `DATABASE_URL` points at a local PostgreSQL and
  `DATABASE_SSLMODE` is unset, behavior is unchanged from before (pool defaults 5/10/300).

## Local run (with Docker / Python 3.11)

```bash
# From the repo root:
cd backend
pip install -r ../deployment/huggingface/requirements.txt
uvicorn deployment.huggingface.app:mounted_app --host 0.0.0.0 --port 7860
```

> The application is developed and tested on Python 3.11 (see root `Dockerfile`).
> Gradio only needs to be installed in the runtime environment; it is not a
> backend dependency.

## Hosting platform (Hugging Face Spaces)

1. Create a **Docker**-based Space.
2. Use the app `Dockerfile` (Python 3.11) as the Space container image.
3. In the Space, override the command to run the adapter:

   ```bash
   uvicorn deployment.huggingface.app:mounted_app --host 0.0.0.0 --port ${PORT:-7860}
   ```

4. Provide `DATABASE_URL` (Neon free tier) and any app secrets via Space Secrets.

## Verification

- `GET /health` → `{"status": "ok"}` (served by the unchanged FastAPI app).
- `POST /ask` / `POST /ask/stream` → same behavior as local.
- CORS allows Chrome and Firefox extension origins (`chrome-extension://*`,
  `moz-extension://*`).

## Notes

- This adapter is a deployment convenience only; it does not change application
  behavior. If the hosting platform exposes `$PORT`, the adapter uses it.
