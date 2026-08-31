"""Filesystem locations shared by indexing, API, UI, and evaluation code."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(
    os.getenv("MYAI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("MYAI_DATA_DIR", PROJECT_ROOT / "data")).resolve()
DOCUMENTS_DIR = Path(
    os.getenv("MYAI_DOCUMENTS_DIR", DATA_DIR / "documents")
).resolve()
STRUCTURED_FINANCE_PATH = Path(
    os.getenv("MYAI_STRUCTURED_FINANCE_PATH", DATA_DIR / "structured_financial_data.json")
).resolve()

ARTIFACTS_DIR = Path(
    os.getenv("MYAI_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
).resolve()
INDEX_DIR = Path(os.getenv("MYAI_INDEX_DIR", ARTIFACTS_DIR / "faiss_index")).resolve()

RUNTIME_DIR = Path(os.getenv("MYAI_RUNTIME_DIR", PROJECT_ROOT / "runtime")).resolve()
FAQ_CACHE_PATH = RUNTIME_DIR / "faq_cache" / "faq_finance.json"
FEEDBACK_DB_PATH = RUNTIME_DIR / "feedback.json"
FEW_SHOT_PATH = RUNTIME_DIR / "few_shot_examples.json"

CACHE_DIR = Path(os.getenv("MYAI_CACHE_DIR", PROJECT_ROOT / ".cache")).resolve()


def ensure_runtime_directories() -> None:
    """Create local-only directories required while the service is running."""
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    FAQ_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
