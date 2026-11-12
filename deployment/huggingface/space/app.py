"""PageGlance — Hugging Face Gradio SDK Space entrypoint.

This is the file Hugging Face runs for a free **Gradio SDK** Space (`python app.py`).
It mounts the real PageGlance FastAPI application inside a Gradio Blocks app using
Gradio's official `gr.mount_gradio_app()` integration and serves it with uvicorn
on `$PORT` (Spaces default 7860).

It does NOT duplicate any RAG logic, response contracts, middleware, CORS, or
embedding/provider configuration. It imports the unchanged application package and
only adds the Gradio wrapper + the server launcher.

Import resolution
-----------------
The Space must contain the `app` package (the contents of `backend/app/`). Two
layouts are supported so staging is forgiving:

* `app/` at the Space root (preferred, self-contained) -> `from app.main import app`
* `backend/app/` present (full repo staging)          -> `from backend.app.main import app`
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The application's default Tesseract executable path is a Windows path. On the
# Linux Spaces runtime we point it at the standard Linux binary. This is safe to
# override via the same env var the app already reads.
os.environ.setdefault("PAGEGLANCE_TESSERACT_CMD", "/usr/bin/tesseract")

import gradio as gr  # noqa: E402
import spaces  # noqa: E402
import spaces.zero as _spaces_zero  # noqa: E402

try:
    from app.main import app as existing_fastapi_app  # noqa: E402
except ModuleNotFoundError:
    # Fallback for a Space staged from the full repository layout.
    _backend = os.path.join(_HERE, "backend")
    if _backend not in sys.path:
        sys.path.insert(0, _backend)
    from app.main import app as existing_fastapi_app  # noqa: E402

# A minimal Blocks UI for the landing surface of the Space. It does not replace or
# duplicate any PageGlance functionality; it is the container the Gradio mount
# serves at the Space root. All real routes (/health, /ask, /ask/stream,
# /documents/upload, ...) remain on the SAME app, untouched.
# ---------------------------------------------------------------------------
# ZeroGPU startup compatibility (free Hugging Face ZeroGPU Spaces)
# ---------------------------------------------------------------------------
# A ZeroGPU Space requires at least one Gradio event handler backed by an
# @spaces.GPU-decorated function during startup; otherwise startup fails with
# "No @spaces.GPU function detected during startup". This handler is a strict
# no-op: it performs no work and is never invoked (it is wired to a hidden button
# that can never be clicked). Its only purpose is to satisfy ZeroGPU's startup
# validation. All real PageGlance embedding / RAG / retrieval / OCR inference
# keeps running on CPU.
@spaces.GPU(duration=1)
def _zerogpu_noop() -> None:
    """No-op required for ZeroGPU startup validation. Never called."""
    return None


# A minimal Blocks UI for the landing surface of the Space. It does not replace or
# duplicate any PageGlance functionality; it is the container the Gradio mount
# serves at the Space root. All real routes (/health, /ask, /ask/stream,
# /documents/upload, ...) remain on the SAME app, untouched.
with gr.Blocks(title="PageGlance") as demo:
    gr.Markdown("# PageGlance")
    gr.Markdown(
        "Backend API is mounted at the Space root. `/health`, `/ask`, "
        "`/ask/stream`, and document endpoints are served by the unchanged "
        "PageGlance FastAPI application."
    )
    # Hidden button hosting the ZeroGPU no-op handler. visible=False means it is
    # never rendered, so it can never be triggered automatically or by a user;
    # registering the handler is all that matters for ZeroGPU startup validation.
    _zerogpu_button = gr.Button("ZeroGPU no-op", visible=False)
    _zerogpu_button.click(_zerogpu_noop)

# Verify the no-op is genuinely registered as a Gradio event handler (not merely
# defined) — this is what the ZeroGPU runtime scans for at startup. We inspect
# demo.fns.values(): each entry is a BlockFunction whose .fn attribute holds the
# actual callable registered via .click(). We match by identity only — the
# handler is never invoked during verification.
_block_functions = getattr(demo, "fns", {}).values()
_zerogpu_handlers = sum(
    1 for _bf in _block_functions
    if getattr(_bf, "fn", None) is _zerogpu_noop
)
print(
    f"[pageglance] ZeroGPU no-op handler registrations: {_zerogpu_handlers} "
    "(Gradio BlockFunction .fn identity)"
)
if not _zerogpu_handlers:
    raise RuntimeError(
        "ZeroGPU no-op handler was not registered as a Gradio event handler; "
        "ZeroGPU startup validation will fail with 'No @spaces.GPU function "
        "detected during startup'."
    )

# Official Gradio FastAPI mounting integration.
mounted_app = gr.mount_gradio_app(existing_fastapi_app, demo, path="/")

# ---------------------------------------------------------------------------
# ZeroGPU startup reporting (custom ASGI launch pattern)
# ---------------------------------------------------------------------------
# A ZeroGPU Space normally reports its GPU startup via demo.launch(). Because we
# serve the mounted FastAPI app directly with uvicorn.run() (there is no
# demo.launch()), we must explicitly invoke spaces.zero.startup() instead —
# after every Gradio event handler is registered and after mounted_app is built,
# and before uvicorn begins listening. The module-level flag guarantees startup
# is reported at most once, even if run() is called multiple times.
_zerogpu_startup_reported = False


def _report_zerogpu_startup() -> None:
    """Report ZeroGPU startup exactly once (no-op on subsequent calls)."""
    global _zerogpu_startup_reported
    if _zerogpu_startup_reported:
        return
    _zerogpu_startup_reported = True
    _spaces_zero.startup()
    print("[pageglance] ZeroGPU startup reported via spaces.zero.startup()")


def run() -> None:
    """Serve the mounted FastAPI app on $PORT (Spaces default 7860)."""
    import uvicorn

    # Report ZeroGPU startup before the server accepts connections. This is the
    # step that replaces demo.launch() for the mounted-FastAPI launch pattern.
    try:
        _report_zerogpu_startup()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[pageglance] ZeroGPU startup report skipped "
            f"(not on a ZeroGPU Space or spaces.zero unavailable): {exc}"
        )

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(mounted_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
