"""Case-by-case quality loop for RAG failures.

The store deliberately separates raw user feedback from curated evaluation
data.  A thumbs-down creates a pending case; only a reviewed case can be
exported as an evaluation or few-shot candidate.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from threading import RLock
import time
from typing import Any
from uuid import uuid4


FAILURE_STAGES = (
    "query_rewrite",
    "knowledge_boundary",
    "pdf_parsing",
    "chunking",
    "dense_recall",
    "bm25_recall",
    "fusion",
    "reranker",
    "generation",
    "citation",
    "refusal",
    "unknown",
)
DISPOSITIONS = ("evaluation_candidate", "few_shot_candidate", "ignore")


class BadCaseStore:
    """Persist recent traces and a human-review queue in local runtime files."""

    def __init__(self, trace_path: Path, case_path: Path, max_traces: int = 500):
        self.trace_path = Path(trace_path)
        self.case_path = Path(case_path)
        self.max_traces = max(1, max_traces)
        self._lock = RLock()

    @staticmethod
    def _read_list(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} 必须保存 JSON 数组")
        return payload

    @staticmethod
    def _write_list(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def record_trace(self, payload: dict[str, Any]) -> str:
        """Store one answer trace and return its stable identifier."""
        trace_id = uuid4().hex
        row = {
            "trace_id": trace_id,
            "created_at": time.time(),
            **deepcopy(payload),
        }
        with self._lock:
            rows = self._read_list(self.trace_path)
            rows.append(row)
            self._write_list(self.trace_path, rows[-self.max_traces :])
        return trace_id

    def get_trace(self, trace_id: str | None) -> dict[str, Any] | None:
        if not trace_id:
            return None
        with self._lock:
            for row in reversed(self._read_list(self.trace_path)):
                if row.get("trace_id") == trace_id:
                    return deepcopy(row)
        return None

    @staticmethod
    def _source_refs(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep citation identity without duplicating full report text."""
        return [
            {
                "company": source.get("company"),
                "source_file": source.get("source_file"),
                "page_number": source.get("page_number"),
            }
            for source in sources
        ]

    def create_from_feedback(
        self,
        *,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        feedback: str,
        trace_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Create a pending Bad Case only for explicit negative feedback."""
        if feedback != "useless":
            return None
        trace = self.get_trace(trace_id)
        case = {
            "bad_case_id": uuid4().hex,
            "created_at": time.time(),
            "status": "pending_review",
            "failure_stage": None,
            "disposition": None,
            "review_notes": "",
            "expected_answer": "",
            "expected_pages": [],
            "question": question,
            "answer": answer,
            "sources": self._source_refs(sources),
            "feedback": feedback,
            "trace_id": trace_id,
            "trace": trace,
        }
        with self._lock:
            rows = self._read_list(self.case_path)
            rows.append(case)
            self._write_list(self.case_path, rows)
        return deepcopy(case)

    def list_cases(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._read_list(self.case_path)
        if status:
            rows = [row for row in rows if row.get("status") == status]
        return deepcopy(rows)

    def review_case(
        self,
        bad_case_id: str,
        *,
        failure_stage: str,
        disposition: str,
        review_notes: str = "",
        expected_answer: str = "",
        expected_pages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Apply human attribution before a case can enter curated data."""
        if failure_stage not in FAILURE_STAGES:
            raise ValueError(f"failure_stage 必须是: {', '.join(FAILURE_STAGES)}")
        if disposition not in DISPOSITIONS:
            raise ValueError(f"disposition 必须是: {', '.join(DISPOSITIONS)}")
        with self._lock:
            rows = self._read_list(self.case_path)
            for row in rows:
                if row.get("bad_case_id") != bad_case_id:
                    continue
                row.update(
                    {
                        "status": "reviewed",
                        "reviewed_at": time.time(),
                        "failure_stage": failure_stage,
                        "disposition": disposition,
                        "review_notes": review_notes.strip(),
                        "expected_answer": expected_answer.strip(),
                        "expected_pages": expected_pages or [],
                    }
                )
                self._write_list(self.case_path, rows)
                return deepcopy(row)
        raise KeyError(f"Bad Case 不存在: {bad_case_id}")

    def export_reviewed(self, output_path: Path) -> int:
        """Export reviewed candidates; ignored cases never enter data sets."""
        rows = [
            row
            for row in self.list_cases(status="reviewed")
            if row.get("disposition") != "ignore"
        ]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )
        output_path.write_text(content, encoding="utf-8")
        return len(rows)
