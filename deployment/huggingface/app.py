"""PageGlance Hugging Face Space deployment adapter.

This module mounts the *actual* PageGlance FastAPI application inside a Gradio
Blocks app using Gradio's official `gr.mount_gradio_app()` integration.

It is isolated from the application code: it imports `app.main` (the real
entrypoint), preserves its routes, middleware, CORS and response contracts, and
does NOT duplicate RAG logic or modify the embedding model / providers.

Usage (server entrypoint):
    uvicorn deployment.huggingface.app:mounted_app --host 0.0.0.0 --port ${PORT:-7860}

Gradio SDK Spaces run this file directly (`python app.py` in the Space root);
the `if __name__ == "__main__"` block below serves the mounted app with uvicorn
on $PORT. Both entrypoints start exactly one server and reuse the same app.

Hosting platform (Hugging Face Spaces) exposes the port via $PORT; the adapter
respects it and falls back to 7860 (Space default) for local runs.
"""

import os

# --- Environment configuration (must be set before importing app code) ----
# The application's default Tesseract executable path is a Windows path. On the
# Linux hosting environment we point it at the standard Linux binary. Safe to
# override via the same env var the app already reads.
os.environ.setdefault("PAGEGLANCE_TESSERACT_CMD", "/usr/bin/tesseract")

import gradio as gr  # noqa: E402

from app.main import app as existing_fastapi_app  # noqa: E402 (real app)

# A minimal Blocks UI for the management / landing surface of the Space.
# It does not replace or duplicate any PageGlance functionality; it simply gives
# the Gradio mount a concrete UI to serve at the Space root. All real PageGlance
# routes (including /health, /ask, /ask/stream, /documents/upload, ...) remain
# available on the SAME app, untouched.
with gr.Blocks(title="PageGlance") as demo:
    gr.Markdown("# PageGlance")
    gr.Markdown(
        "Backend API is mounted at the Space root. "
        "`/health`, `/ask`, `/ask/stream`, and document endpoints are served "
        "by the unchanged PageGlance FastAPI application."
    )

# Official Gradio FastAPI mounting integration.
mounted_app = gr.mount_gradio_app(existing_fastapi_app, demo, path="/")


def run() -> None:
    """Serve the mounted FastAPI app (used by Gradio SDK Spaces)."""
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(mounted_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
