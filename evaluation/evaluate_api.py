#!/usr/bin/env python3
"""调用正在运行的 FastAPI，评测最终答案、引用、拒答和延迟。"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import requests


DEFAULT_DATASET = Path(__file__).with_name("datasets") / "golden_test_set.jsonl"
REFUSAL_PATTERNS = (
    "无法回答",
    "无法获取答案",
    "未找到",
    "没有相关",
    "知识库中没有",
    "不包含",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"[\s,，]", "", text).lower()


def answer_checks_pass(answer: str, groups: list[list[str]]) -> bool:
    normalized = normalize(answer)
    return all(
        any(normalize(alternative) in normalized for alternative in alternatives)
        for alternatives in groups
    )


def is_refusal(answer: str) -> bool:
    return any(pattern in answer for pattern in REFUSAL_PATTERNS)


def source_pages(payload: dict) -> set[tuple[str, int]]:
    return {
        (str(source.get("source_file", "")), int(source.get("page_number", 0)))
        for source in payload.get("source_documents", [])
        if source.get("source_file") and source.get("page_number")
    }


def gold_pages(case: dict) -> set[tuple[str, int]]:
    return {
        (str(page["source_file"]), int(page["page_number"]))
        for page in case.get("gold_pages", [])
    }


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="MyAI端到端API评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只调用重排与相关性判断，不调用生成模型和消耗生成额度",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    if args.split:
        cases = [case for case in cases if case.get("split") == args.split]
    if args.category:
        cases = [case for case in cases if case.get("category") == args.category]
    if args.limit:
        cases = cases[: args.limit]

    endpoint = args.base_url.rstrip("/") + "/rag_query"
    results = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        try:
            response = requests.post(
                endpoint,
                json={
                    "question": case["question"],
                    "debug": args.retrieval_only,
                    "retrieval_only": args.retrieval_only,
                },
                timeout=args.timeout,
            )
            latency = time.perf_counter() - started
            response.raise_for_status()
            payload = response.json()
            answer = str(payload.get("answer", ""))
            route = str((payload.get("retrieval_debug") or {}).get("route", "unknown"))
            predicted_pages = source_pages(payload)
            expected_pages = gold_pages(case)
            if args.retrieval_only and case.get("answerable"):
                answer_pass = None
                citation_hit = bool(expected_pages.intersection(predicted_pages))
                citation_recall = (
                    len(expected_pages.intersection(predicted_pages)) / len(expected_pages)
                    if expected_pages
                    else 0.0
                )
                refusal_pass = None
            elif args.retrieval_only:
                answer_pass = None
                citation_hit = not predicted_pages
                citation_recall = 1.0 if not predicted_pages else 0.0
                refusal_pass = not predicted_pages
            elif case.get("answerable"):
                answer_pass = answer_checks_pass(answer, case.get("answer_checks", []))
                citation_hit = bool(expected_pages.intersection(predicted_pages))
                citation_recall = (
                    len(expected_pages.intersection(predicted_pages)) / len(expected_pages)
                    if expected_pages
                    else 0.0
                )
                refusal_pass = not is_refusal(answer)
            else:
                answer_pass = is_refusal(answer)
                citation_hit = not predicted_pages
                citation_recall = 1.0 if not predicted_pages else 0.0
                refusal_pass = answer_pass
            row = {
                "query_id": case["query_id"],
                "question": case["question"],
                "answerable": bool(case.get("answerable")),
                "answer": answer,
                "answer_pass": answer_pass,
                "citation_hit": citation_hit,
                "citation_recall": citation_recall,
                "refusal_pass": refusal_pass,
                "latency_seconds": latency,
                "predicted_pages": sorted(predicted_pages),
                "gold_pages": sorted(expected_pages),
                "error": None,
                "route": route,
            }
        except Exception as exc:
            latency = time.perf_counter() - started
            row = {
                "query_id": case["query_id"],
                "question": case["question"],
                "answerable": bool(case.get("answerable")),
                "answer": "",
                "answer_pass": False,
                "citation_hit": False,
                "citation_recall": 0.0,
                "refusal_pass": False,
                "latency_seconds": latency,
                "predicted_pages": [],
                "gold_pages": sorted(gold_pages(case)),
                "error": str(exc),
                "route": "error",
            }
        results.append(row)
        if args.retrieval_only:
            status = "PASS" if row["citation_hit"] else "FAIL"
        else:
            status = "PASS" if row["answer_pass"] and row["citation_hit"] else "FAIL"
        print(f"[{index}/{len(cases)}] {case['query_id']} {status} {row['latency_seconds']:.2f}s")

    answerable = [row for row in results if row["answerable"]]
    unanswerable = [row for row in results if not row["answerable"]]
    successful_latencies = [row["latency_seconds"] for row in results if not row["error"]]
    summary = {
        "total": len(results),
        "mode": "retrieval_only" if args.retrieval_only else "end_to_end",
        "answer_accuracy": None if args.retrieval_only else safe_mean(
            [float(row["answer_pass"]) for row in answerable]
        ),
        "citation_hit_rate": safe_mean([float(row["citation_hit"]) for row in answerable]),
        "citation_recall": safe_mean([row["citation_recall"] for row in answerable]),
        "refusal_accuracy": None if args.retrieval_only else safe_mean(
            [float(row["refusal_pass"]) for row in unanswerable]
        ),
        "zero_source_rate_on_unanswerable": safe_mean(
            [float(not row["predicted_pages"]) for row in unanswerable]
        ),
        "end_to_end_correct_rate": None if args.retrieval_only else safe_mean(
            [float(row["answer_pass"] and row["citation_hit"]) for row in results]
        ),
        "mean_latency_seconds": safe_mean(successful_latencies),
        "errors": sum(1 for row in results if row["error"]),
        "route_distribution": dict(Counter(row["route"] for row in results)),
    }
    print("\n=== 端到端评测结果 ===")
    for name, value in summary.items():
        print(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")

    report = {"endpoint": endpoint, "summary": summary, "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
