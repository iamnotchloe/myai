"""Offline evaluation for multi-turn RAG query rewriting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from myai_rag.query_rewrite import rewrite_retrieval_query  # noqa: E402


ALIASES = {
    "滨江消费品": "滨江消费品有限公司",
    "滨江消费品有限公司": "滨江消费品有限公司",
    "澜赋科技": "澜赋科技有限公司",
    "澜赋科技有限公司": "澜赋科技有限公司",
    "蓝天旅游": "蓝天旅游有限公司",
    "蓝天旅游有限公司": "蓝天旅游有限公司",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation/datasets/multiturn_query_rewrite.jsonl",
    )
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    failures = []
    for case in cases:
        result = rewrite_retrieval_query(case["question"], case["history"], ALIASES)
        query_ok = result.rewritten_query == case.get(
            "expected_query", result.rewritten_query
        )
        contains_ok = all(
            expected in result.rewritten_query
            for expected in case.get("expected_contains", [])
        )
        clarification_ok = (
            result.needs_clarification == case["expected_clarification"]
        )
        if query_ok and contains_ok and clarification_ok:
            passed += 1
        else:
            failures.append(
                {
                    "id": case["id"],
                    "actual_query": result.rewritten_query,
                    "actual_clarification": result.needs_clarification,
                }
            )

    accuracy = passed / len(cases) if cases else 0.0
    print(json.dumps({"cases": len(cases), "passed": passed, "accuracy": accuracy,
                      "failures": failures}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
