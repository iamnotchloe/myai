"""Offline slot evaluation, with explicitly opt-in paired retrieval evaluation.

Labels are development regression annotations, not independently reviewed data.
The default command performs no network calls and loads no model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from myai_rag.query_rewrite import FINANCE_TOPICS, rewrite_retrieval_query  # noqa: E402


ALIASES = {
    alias: company
    for company in ("滨江消费品有限公司", "澜赋科技有限公司", "蓝天旅游有限公司")
    for alias in (company, company.removesuffix("有限公司"))
}
STAGES = ("dense", "bm25", "fused", "reranked", "final_sources")


def extract_slots(query: str) -> dict[str, list[str]]:
    topics = [topic for topic in FINANCE_TOPICS if topic in query]
    topics = [topic for topic in topics if not any(topic != other and topic in other for other in topics)]
    return {
        "companies": sorted({company for alias, company in ALIASES.items() if alias in query}),
        "years": sorted(set(re.findall(r"(?:19|20)\d{2}", query))),
        "topics": sorted(set(topics)),
    }


def evaluate_case(case: dict) -> dict:
    result = rewrite_retrieval_query(case["question"], case.get("history", []), ALIASES)
    actual = extract_slots(result.rewritten_query)
    original_slots = extract_slots(case["question"])
    expected = case.get("gold_slots", {})
    checks = {
        f"{slot}_correct": sorted(expected[slot]) == actual[slot]
        for slot in ("companies", "years", "topics") if slot in expected
    }
    checks["clarification_correct"] = result.needs_clarification == case["expected_clarification"]
    inherited_slots = {}
    for slot in ("companies", "years", "topics"):
        if slot not in expected:
            continue
        inherited = list(getattr(result, f"inherited_{slot}"))
        if slot == "topics":
            inherited = [topic for topic in inherited if not any(topic != other and topic in other for other in inherited)]
        inherited_slots[slot] = sorted(inherited)
        wanted = [] if case["expected_clarification"] else sorted(set(expected[slot]) - set(original_slots[slot]))
        checks[f"inherited_{slot}_correct"] = sorted(inherited) == wanted
    checks["forbidden_entities_absent"] = not any(
        value in result.rewritten_query for value in case.get("forbidden_entities", [])
    )
    if "expected_query" in case:
        checks["exact_query_correct"] = result.rewritten_query == case["expected_query"]
    if "expected_contains" in case:
        checks["required_text_present"] = all(value in result.rewritten_query for value in case["expected_contains"])
    return {
        "id": case["id"], "actual_query": result.rewritten_query,
        "actual_clarification": result.needs_clarification, "actual_slots": actual,
        "inherited_slots": inherited_slots,
        "checks": checks, "passed": all(checks.values()),
    }


def offline_report(cases: list[dict]) -> dict:
    rows = [evaluate_case(case) for case in cases]
    metrics = {}
    for name in sorted({name for row in rows for name in row["checks"]}):
        values = [row["checks"][name] for row in rows if name in row["checks"]]
        metrics[name] = {"accuracy": mean(values), "labeled_cases": len(values)}
    tp = sum(case["expected_clarification"] and row["actual_clarification"] for case, row in zip(cases, rows))
    fp = sum(not case["expected_clarification"] and row["actual_clarification"] for case, row in zip(cases, rows))
    fn = sum(case["expected_clarification"] and not row["actual_clarification"] for case, row in zip(cases, rows))
    return {
        "cases": len(rows), "passed": sum(row["passed"] for row in rows),
        "accuracy": mean(row["passed"] for row in rows) if rows else None,
        "slot_metrics": metrics, "failures": [row for row in rows if not row["passed"]],
        "results": rows,
        "clarification": {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None},
        "annotation_status": "Project-authored development regression labels; no independent human review claimed.",
    }


def stage_pages(payload: dict, stage: str) -> list[tuple[str, int]] | None:
    debug = payload.get("retrieval_debug") or {}
    if stage == "final_sources":
        items = payload.get("source_documents", [])
    else:
        if debug.get("route") != "rag":
            return None
        items = debug.get(stage)
        if items is None:
            return None
        items = sorted(items, key=lambda item: item.get("rank", 0))
    pages = []
    for item in items:
        if item.get("source_file") and item.get("page_number"):
            key = (str(item["source_file"]), int(item["page_number"]))
            if key not in pages:
                pages.append(key)
    return pages


def retrieval_metrics(pages: list[tuple[str, int]], gold: set[tuple[str, int]], k: int) -> dict:
    first = next((rank for rank, page in enumerate(pages[:k], 1) if page in gold), None)
    return {"hit": float(first is not None), "mrr": 1.0 / first if first else 0.0}


def paired_retrieval_report(cases: list[dict], base_url: str, k: int, timeout: float, post=None) -> dict:
    if post is None:
        import requests
        post = requests.post
    labeled = [case for case in cases if case.get("gold_pages") and not case["expected_clarification"]]
    rows = []
    for case in labeled:
        rewrite = evaluate_case(case)
        row = {"id": case["id"], "stages": {}, "error": None, "routes": {}}
        payloads = {}
        try:
            for name, query in (("original", case["question"]), ("rewritten", rewrite["actual_query"])):
                response = post(base_url.rstrip("/") + "/rag_query", json={
                    "question": query, "history": [], "debug": True, "retrieval_only": True,
                }, timeout=timeout)
                response.raise_for_status()
                payloads[name] = response.json()
                if payloads[name].get("success") is False:
                    raise ValueError(f"{name} API response reported success=false")
                row["routes"][name] = (payloads[name].get("retrieval_debug") or {}).get("route", "unknown")
            gold = {(page["source_file"], int(page["page_number"])) for page in case["gold_pages"]}
            for stage in STAGES:
                before = stage_pages(payloads["original"], stage)
                after = stage_pages(payloads["rewritten"], stage)
                if before is not None and after is not None:
                    row["stages"][stage] = {
                        "original": retrieval_metrics(before, gold, k),
                        "rewritten": retrieval_metrics(after, gold, k),
                    }
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    stage_reports = {}
    for stage in STAGES:
        pairs = [row["stages"][stage] for row in rows if stage in row["stages"]]
        metrics = {}
        for metric in ("hit", "mrr"):
            before = mean(pair["original"][metric] for pair in pairs) if pairs else None
            after = mean(pair["rewritten"][metric] for pair in pairs) if pairs else None
            metrics[f"{metric}@{k}"] = {"original": before, "rewritten": after,
                "lift": after - before if pairs else None}
        stage_reports[stage] = {"paired_cases": len(pairs),
            "coverage": len(pairs) / len(labeled) if labeled else None, "metrics": metrics}
    return {"eligible_cases": len(labeled), "errors": sum(row["error"] is not None for row in rows),
        "stages": stage_reports, "results": rows,
        "limitations": ["Both requests have empty history; final_sources includes API route behavior.",
            "Stage lift uses pairs where both routes ran RAG; missing stages are unmeasured, not zero.",
            "retrieval_only disables answer generation, but configured reranker API calls may incur cost."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation/datasets/multiturn_query_rewrite.jsonl")
    parser.add_argument("--base-url", help="Opt in to paired retrieval-only API calls; external reranking may incur cost.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = offline_report(cases)
    if args.base_url:
        report["paired_retrieval"] = paired_retrieval_report(cases, args.base_url, args.k, args.timeout)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report["failures"] or report.get("paired_retrieval", {}).get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
