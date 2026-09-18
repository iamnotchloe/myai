#!/usr/bin/env python3
"""Evaluate the actual API pipeline, with auditable human semantic reviews.

Debug is always enabled to measure Dense/BM25/RRF/reranker. Keyword checks are
lexical_field_pass_rate, not semantic accuracy. review_answers.py reviews a saved
run without calling the API again.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests

try:
    from .metrics import (aggregate_stage_metrics, human_review_metrics, mean_or_none,
                          performance_metrics, refusal_metrics, unique_pages)
except ImportError:
    from metrics import (aggregate_stage_metrics, human_review_metrics, mean_or_none,
                         performance_metrics, refusal_metrics, unique_pages)


DEFAULT_DATASET = Path(__file__).with_name("datasets") / "dev_set_v2.jsonl"
REFUSAL_PATTERNS = (
    "无法回答", "无法获取答案", "未找到", "没有相关", "知识库中没有", "不包含",
    "未披露", "无法确定", "信息不足", "没有足够", "不能提供", "不支持查询",
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize(text):
    return re.sub(r"[\s,，]", "", text).lower()


def answer_checks_pass(answer, groups):
    if not groups:
        return None
    normalized = normalize(answer)
    return all(any(normalize(alternative) in normalized for alternative in alternatives)
               for alternatives in groups)


def is_refusal(answer):
    return any(pattern in answer for pattern in REFUSAL_PATTERNS)


def source_pages(payload):
    return {(str(source["source_file"]), int(source["page_number"]))
            for source in payload.get("source_documents", [])
            if source.get("source_file") and source.get("page_number")}


def gold_pages(case):
    return {(str(page["source_file"]), int(page["page_number"]))
            for page in case.get("gold_pages", []) if page.get("relevance_grade", 3) > 0}


def evaluate_response(case, payload, latency, retrieval_only=False):
    answer = str(payload.get("answer", ""))
    debug = payload.get("retrieval_debug") or {}
    telemetry = payload.get("telemetry") or {}
    route = str(debug.get("route", "unknown"))
    expected, predicted = gold_pages(case), source_pages(payload)
    hits = expected & predicted
    # A failed task is not evidence of a correctly handled refusal.
    outcome = telemetry.get("outcome")
    llm_status = telemetry.get("llm_status")
    failed = (payload.get("success") is False or outcome == "service_error"
              or llm_status in {"failed", "error", "timeout", "not_configured", "invalid_response"})
    if retrieval_only or failed or outcome == "clarification":
        refusal = None
    elif outcome in {"refused", "insufficient_evidence"}:
        refusal = True
    elif outcome == "answered":
        refusal = False
    else:
        refusal = is_refusal(answer)
    field_pass = (None if retrieval_only or not case["answerable"] else
                  False if failed else answer_checks_pass(answer, case.get("answer_checks", [])))
    stage_pages = {}
    if route == "rag":
        for stage in ("dense", "bm25", "fused", "reranked"):
            if isinstance(debug.get(stage), list):
                stage_pages[stage] = unique_pages([
                    (str(item["source_file"]), int(item["page_number"]))
                    for item in debug[stage] if item.get("source_file") and item.get("page_number")
                ])
    gold_relevance = [{"source_file": page["source_file"], "page_number": page["page_number"],
                       "relevance_grade": page.get("relevance_grade", 3)}
                      for page in case.get("gold_pages", [])]
    stage_diagnostics = {}
    if "fused" in stage_pages and "reranked" in stage_pages:
        prior, later = set(stage_pages["fused"]), set(stage_pages["reranked"])
        stage_diagnostics["fused_to_reranked"] = {
            "lost_gold_pages": sorted((expected & prior) - later),
            "gained_gold_pages": sorted((expected & later) - prior),
        }
    e2e_pass = None if retrieval_only or (case["answerable"] and field_pass is None) else (
        not failed and refusal is False and bool(field_pass) and bool(hits) if case["answerable"] else
        not failed and refusal is True and not predicted
    )
    return {
        "query_id": case["query_id"], "question": case["question"],
        "history": case.get("history", []), "answerable": bool(case["answerable"]),
        "answer": answer, "lexical_field_pass": field_pass,
        "answer_pass": field_pass if case["answerable"] else refusal,
        "predicted_refusal": refusal, "refusal_pass": None if refusal is None else
        (not refusal if case["answerable"] else refusal),
        "citation_hit": bool(hits) if case["answerable"] else None,
        "citation_recall": len(hits) / len(expected) if expected else None,
        "citation_precision": len(hits) / len(predicted) if predicted else 0.0,
        "lexical_end_to_end_pass": e2e_pass,
        "latency_seconds": latency, "predicted_pages": sorted(predicted),
        "gold_pages": sorted(expected), "gold_relevance": gold_relevance,
        "gold_answer": case.get("gold_answer"), "answer_checks": case.get("answer_checks", []),
        "source_documents": payload.get("source_documents", []),
        "resolved_question": payload.get("resolved_question"), "trace_id": payload.get("trace_id"),
        "retrieval_debug": debug, "stage_pages": stage_pages, "stage_diagnostics": stage_diagnostics,
        "telemetry": telemetry, "outcome": outcome,
        "error": "API reported task failure" if failed else None,
        "error_type": "timeout" if llm_status == "timeout" else "task_failure" if failed else None,
        "route": route,
    }


def summarize(results, retrieval_only=False, ks=(1, 3, 5, 10), reviews=(),
              prompt_cost_per_million=None, completion_cost_per_million=None):
    answerable = [row for row in results if row["answerable"]]
    unanswerable = [row for row in results if not row["answerable"]]
    refusal = refusal_metrics(results) if not retrieval_only else None
    performance = performance_metrics(results, prompt_cost_per_million, completion_cost_per_million)
    lexical = mean_or_none([row["lexical_field_pass"] for row in answerable])
    e2e = mean_or_none([row["lexical_end_to_end_pass"] for row in results])
    return {
        "total": len(results), "mode": "retrieval_only" if retrieval_only else "end_to_end",
        "lexical_field_pass_rate": lexical,
        "lexical_field_check_coverage": (sum(row["lexical_field_pass"] is not None for row in answerable)
                                         / len(answerable) if answerable else None),
        "answer_accuracy": lexical,
        "answer_accuracy_definition": "Deprecated alias of lexical_field_pass_rate; not semantic accuracy.",
        "citation_hit_rate": mean_or_none([float(bool(row["citation_hit"])) for row in answerable]),
        "citation_recall": mean_or_none([row["citation_recall"] for row in answerable]),
        "citation_precision": mean_or_none([row["citation_precision"] for row in answerable]),
        "refusal": refusal,
        "refusal_accuracy": refusal["refusal_success_rate"] if refusal else None,
        "refusal_detection_method": "API outcome when available, keyword fallback for legacy responses; human reviews judge semantic correctness.",
        "zero_source_rate_on_unanswerable": mean_or_none([
            float(not row.get("error") and not row["predicted_pages"]) for row in unanswerable]),
        "lexical_end_to_end_pass_rate": e2e, "end_to_end_correct_rate": e2e,
        "stage_metrics": aggregate_stage_metrics(results, ks),
        "human_review": human_review_metrics(results, reviews) if not retrieval_only else None,
        "mean_latency_seconds": performance["successful_latency_seconds"]["mean"],
        "p50_latency_seconds": performance["latency_seconds"]["p50"],
        "p95_latency_seconds": performance["latency_seconds"]["p95"],
        "errors": sum(bool(row.get("error")) for row in results), "performance": performance,
        "route_distribution": dict(Counter(row["route"] for row in results)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--timeout", type=float, default=390.0,
                        help="Client timeout; budget for backend LLM and reranker retries (default 390s).")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip generation; no answer/faithfulness/refusal scores are claimed.")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--reviews", type=Path, help="Human reviews JSONL bound to exact answer hashes.")
    parser.add_argument("--prompt-cost-per-million", type=float)
    parser.add_argument("--completion-cost-per-million", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(k <= 0 for k in args.k):
        parser.error("--k values must be positive")
    for price in (args.prompt_cost_per_million, args.completion_cost_per_million):
        if price is not None and price < 0:
            parser.error("Token prices cannot be negative")
    cases = load_jsonl(args.dataset)
    if args.split:
        cases = [case for case in cases if case.get("split") == args.split]
    if args.category:
        cases = [case for case in cases if case.get("category") == args.category]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        parser.error("No matching cases; choose the intended --dataset and --split")
    if len({case["query_id"] for case in cases}) != len(cases):
        parser.error("Dataset query_id values must be unique")
    endpoint = args.base_url.rstrip("/") + "/rag_query"
    results = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        try:
            response = requests.post(endpoint, json={
                "question": case["question"], "history": case.get("history", []),
                "debug": True, "retrieval_only": args.retrieval_only,
            }, timeout=args.timeout)
            response.raise_for_status()
            row = evaluate_response(case, response.json(), time.perf_counter() - started,
                                    args.retrieval_only)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            row = evaluate_response(case, {"success": False}, time.perf_counter() - started,
                                    args.retrieval_only)
            row.update(error=str(exc), error_type="timeout" if isinstance(exc, requests.Timeout)
                       else "request_error", route="error")
        results.append(row)
        passed = row["citation_hit"] if args.retrieval_only else row["lexical_end_to_end_pass"]
        status = "ERROR" if row["error"] else "PASS" if passed else "FAIL"
        print(f"[{index}/{len(cases)}] {case['query_id']} {status} {row['latency_seconds']:.2f}s")
    reviews = load_jsonl(args.reviews) if args.reviews else []
    summary = summarize(results, args.retrieval_only, args.k, reviews,
                        args.prompt_cost_per_million, args.completion_cost_per_million)
    report = {"endpoint": endpoint, "dataset": str(args.dataset), "summary": summary, "results": results}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
