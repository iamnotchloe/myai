"""Build few-shot examples only from human-reviewed Bad Cases."""

from __future__ import annotations

import json

from .config import (
    CURATED_BAD_CASE_PATH,
    FEEDBACK_DB_PATH,
    FEW_SHOT_PATH,
    ensure_runtime_directories,
)


def _read_jsonl() -> list[dict]:
    if not CURATED_BAD_CASE_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in CURATED_BAD_CASE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def update_few_shot() -> None:
    """Promote reviewed, corrected cases; raw likes/dislikes never enter prompts."""
    ensure_runtime_directories()
    feedbacks = []
    if FEEDBACK_DB_PATH.exists():
        feedbacks = json.loads(FEEDBACK_DB_PATH.read_text(encoding="utf-8"))

    context_by_trace = {
        item.get("trace_id"): str(item.get("context", ""))
        for item in feedbacks
        if item.get("trace_id")
    }
    candidates = []
    for case in _read_jsonl():
        if case.get("status") != "reviewed":
            continue
        if case.get("disposition") != "few_shot_candidate":
            continue
        expected_answer = str(case.get("expected_answer", "")).strip()
        context = context_by_trace.get(case.get("trace_id"), "").strip()
        if not expected_answer or len(context) <= 100:
            continue
        candidates.append(
            {
                "question": case.get("question", ""),
                "context": context,
                "answer": expected_answer,
                "reviewed_at": float(case.get("reviewed_at", 0)),
            }
        )

    latest_by_question = {}
    for example in candidates:
        key = "".join(str(example["question"]).casefold().split())
        previous = latest_by_question.get(key)
        if previous is None or example["reviewed_at"] > previous["reviewed_at"]:
            latest_by_question[key] = example

    selected = sorted(
        latest_by_question.values(),
        key=lambda item: -item["reviewed_at"],
    )[:5]
    output = [
        {
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"],
        }
        for item in selected
    ]
    FEW_SHOT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"更新完成，共 {len(output)} 条人工审核示例")


if __name__ == "__main__":
    update_few_shot()
