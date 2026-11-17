"""Stage the Gradio SDK Space bundle (copy backend/app -> space/app).

The Space folder is self-contained for upload to Hugging Face. HF Spaces cannot
use symlinks or editable repo paths, so the application package is staged here
from the live `backend/app` source. Re-run whenever the backend changes; it is
idempotent and only touches `deployment/huggingface/space/app`.

Run from anywhere:

    python deployment/huggingface/assemble_space.py
"""

import shutil
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]  # repository root
SRC = ROOT / "backend" / "app"
DST = ROOT / "deployment" / "huggingface" / "space" / "app"

_IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"Expected application package at {SRC}")
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=_IGNORED)
    print(f"Staged app package -> {DST}")
    print(f"files: {sum(1 for _ in DST.rglob('*.py'))}")


if __name__ == "__main__":
    main()
