"""Console entry points declared in ``pyproject.toml``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import uvicorn


def run_api() -> None:
    uvicorn.run(
        "myai_rag.api:app",
        host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("BACKEND_PORT", "8001")),
        reload=False,
    )


def run_ui() -> None:
    ui_path = Path(__file__).with_name("ui.py")
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(ui_path)])
    )
