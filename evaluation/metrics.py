"""Dependency-free metrics; missing measurements are None, never successes.

Page precision@K divides by K, including unfilled result positions. Human
semantic judgments are reported separately from keyword checks.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from statistics import mean


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def percentile(values, quantile):
    """Linear interpolation between adjacent ordered observations."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def unique_pages(pages):
    return list(dict.fromkeys(tuple(page) for page in pages))


def page_ranking_metrics(ranked_pages, relevance, k):
    if k <= 0:
        raise ValueError("k must be positive")
    grades = {tuple(page): grade for page, grade in relevance.items() if grade > 0}
    ranked = unique_pages(ranked_pages)[:k]
    hits = 0
    precisions = []
    first_rank = None
    for rank, page in enumerate(ranked, 1):
        if page in grades:
            hits += 1
            precisions.append(hits / rank)
            first_rank = first_rank or rank
    dcg = sum((2 ** grades.get(page, 0) - 1) / math.log2(rank + 1)
              for rank, page in enumerate(ranked, 1))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 1)
                for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:k], 1))
    return {
        "hit": float(bool(hits)),
        "recall": hits / len(grades) if grades else None,
        "precision": hits / k,
        "mrr": 1 / first_rank if first_rank else 0.0,
        "map": sum(precisions) / len(grades) if grades else None,
        "ndcg": dcg / ideal if ideal else None,
    }


def aggregate_stage_metrics(rows, ks):
    stages = {}
    eligible = [row for row in rows if row["answerable"] and row.get("gold_relevance")]
    for stage in ("dense", "bm25", "fused", "reranked"):
        measured = [row for row in eligible if stage in row.get("stage_pages", {})]
        by_k = {}
        for k in ks:
            values = [page_ranking_metrics(
                row["stage_pages"][stage],
                {(item["source_file"], item["page_number"]): item["relevance_grade"]
                 for item in row["gold_relevance"]}, k
            ) for row in measured]
            by_k[str(k)] = {name: mean_or_none([value[name] for value in values])
                            for name in ("hit", "recall", "precision", "mrr", "map", "ndcg")}
        stages[stage] = {
            "evaluated_queries": len(measured), "eligible_queries": len(eligible),
            "coverage": len(measured) / len(eligible) if eligible else None, "at_k": by_k,
        }
    return stages


def refusal_metrics(rows):
    """Positive=refusal, matrix measured on actual decisions, errors unclassified.

    refusal_success_rate also penalizes failed unanswerable requests, whereas
    precision/recall/F1 measure the classifier on actual successful decisions.
    """
    measured = [row for row in rows
                if not row.get("error") and isinstance(row.get("predicted_refusal"), bool)]
    counts = Counter()
    for row in measured:
        expected, predicted = not row["answerable"], row["predicted_refusal"]
        counts["tp" if expected and predicted else
               "fn" if expected else "fp" if predicted else "tn"] += 1
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    unanswerable = sum(not row["answerable"] for row in rows)
    answerable = sum(row["answerable"] for row in measured)
    return {
        "positive_label": "refusal",
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "accuracy_on_measured": (tp + tn) / len(measured) if measured else None,
        "false_refusal_rate": fp / answerable if answerable else None,
        "refusal_success_rate": tp / unanswerable if unanswerable else None,
        "measured_queries": len(measured), "unclassified_queries": len(rows) - len(measured),
        "coverage": len(measured) / len(rows) if rows else None,
    }


def latency_statistics(values):
    return {"count": len(values), "mean": mean_or_none(values),
            "p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}


def performance_metrics(rows, prompt_cost_per_million=None, completion_cost_per_million=None):
    successful = [row for row in rows if not row.get("error")]
    routes = {}
    for route in sorted({row["route"] for row in rows}):
        subset = [row for row in rows if row["route"] == route]
        routes[route] = {"requests": len(subset),
                         "errors": sum(bool(row.get("error")) for row in subset),
                         "latency_seconds": latency_statistics([row["latency_seconds"] for row in subset])}
    names = sorted({name for row in rows
                    for name in (row.get("telemetry") or {}).get("stage_latency_ms", {})})
    stage_latency = {name: latency_statistics([
        row["telemetry"]["stage_latency_ms"][name] for row in rows
        if name in (row.get("telemetry") or {}).get("stage_latency_ms", {})
    ]) for name in names}
    usage_rows = [row for row in rows if isinstance((row.get("telemetry") or {}).get("llm_usage"), dict)
                  and all(isinstance(row["telemetry"]["llm_usage"].get(name), (int, float))
                          for name in ("prompt_tokens", "completion_tokens", "total_tokens"))]
    totals = {name: sum(row["telemetry"]["llm_usage"][name] for row in usage_rows)
              for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if not usage_rows:
        totals = {name: None for name in totals}
    has_prices = prompt_cost_per_million is not None and completion_cost_per_million is not None
    observed_cost = ((totals["prompt_tokens"] * prompt_cost_per_million
                      + totals["completion_tokens"] * completion_cost_per_million) / 1_000_000
                     if has_prices and usage_rows else None)
    return {
        "latency_seconds": latency_statistics([row["latency_seconds"] for row in rows]),
        "successful_latency_seconds": latency_statistics([row["latency_seconds"] for row in successful]),
        "error_rate": sum(bool(row.get("error")) for row in rows) / len(rows) if rows else None,
        "timeout_rate": sum(row.get("error_type") == "timeout" for row in rows) / len(rows) if rows else None,
        "by_route": routes, "stage_latency_ms": stage_latency,
        "llm_usage": {**totals, "measured_queries": len(usage_rows), "total_queries": len(rows)},
        "observed_llm_cost_estimate": observed_cost,
        "cost_scope": "Estimate for reported LLM tokens only; missing usage or unknown retry billing, reranker and embedding fees are excluded. Not total system cost.",
        "cost_prices_per_million": {"prompt": prompt_cost_per_million, "completion": completion_cost_per_million},
    }


def answer_digest(answer):
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _whitespace_normalized(value):
    return "".join(value.split())


def validate_review(review, result):
    """Validate auditability, not semantic entailment: the latter is human work."""
    if review.get("query_id") != result.get("query_id"):
        raise ValueError("Review query_id does not match result")
    if result.get("error"):
        raise ValueError("Cannot review a failed request as a generated answer")
    if review.get("answer_sha256") != answer_digest(result["answer"]):
        raise ValueError("Review answer_sha256 does not match this exact answer")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("A named human reviewer is required")
    if review.get("review_complete") is not True:
        raise ValueError("review_complete must certify all factual claims were assessed")
    if not isinstance(review.get("semantic_answer_correct"), bool):
        raise ValueError("semantic_answer_correct must be a human boolean judgment")
    for name in ("answer_relevance", "completeness"):
        if type(review.get(name)) is not int or review[name] not in (0, 1, 2):
            raise ValueError(f"{name} must be 0 (none), 1 (partial), or 2 (full)")
    claims = review.get("claims")
    if not isinstance(claims, list):
        raise ValueError("claims must be a list, empty only when the answer has no factual assertions")
    contexts = {(item.get("source_file"), item.get("page_number")): item.get("content", "")
                for item in result.get("source_documents", [])}
    normalized_answer = _whitespace_normalized(result["answer"])
    seen_claims = set()
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim"), str):
            raise ValueError("Each claim requires an exact answer span")
        text = _whitespace_normalized(claim["claim"])
        if not text or text not in normalized_answer or text in seen_claims:
            raise ValueError("Claim must be a unique literal span of the evaluated answer")
        seen_claims.add(text)
        if not isinstance(claim.get("supported"), bool):
            raise ValueError("Claim supported must be a human boolean judgment")
        evidence = claim.get("evidence", [])
        if not isinstance(evidence, list) or (claim["supported"] and not evidence):
            raise ValueError("A supported claim requires evidence")
        for citation in evidence:
            if not isinstance(citation, dict):
                raise ValueError("Evidence must contain source_file, page_number and quote")
            if (not isinstance(citation.get("source_file"), str)
                    or type(citation.get("page_number")) is not int
                    or citation["page_number"] < 1
                    or not isinstance(citation.get("quote"), str)):
                raise ValueError("Evidence requires a source filename, positive integer page and text quote")
            key = (citation.get("source_file"), citation.get("page_number"))
            quote = _whitespace_normalized(citation["quote"])
            if not quote or key not in contexts or quote not in _whitespace_normalized(contexts[key]):
                raise ValueError("Evidence quote must occur on a source page returned for this answer")
    return review


def human_review_metrics(results, reviews):
    by_id = {row["query_id"]: row for row in results}
    if len(by_id) != len(results):
        raise ValueError("Duplicate query_id in evaluation results")
    accepted, seen = [], set()
    for review in reviews:
        query_id = review.get("query_id")
        if query_id not in by_id or query_id in seen:
            raise ValueError(f"Unknown or duplicate review query_id: {query_id}")
        seen.add(query_id)
        accepted.append(validate_review(review, by_id[query_id]))
    claims = [claim for review in accepted for claim in review["claims"]]
    per_answer = [sum(claim["supported"] for claim in review["claims"]) / len(review["claims"])
                  for review in accepted if review["claims"]]
    return {
        "method": "human_review_with_source_quote_validation",
        "reviewed_queries": len(accepted), "total_queries": len(results),
        "coverage": len(accepted) / len(results) if results else None,
        "semantic_answer_accuracy": mean_or_none([float(r["semantic_answer_correct"]) for r in accepted]),
        "answer_relevance": mean_or_none([r["answer_relevance"] / 2 for r in accepted]),
        "completeness": mean_or_none([r["completeness"] / 2 for r in accepted]),
        "faithfulness": mean_or_none(per_answer),
        "claim_support": sum(c["supported"] for c in claims) / len(claims) if claims else None,
        "reviewed_claims": len(claims), "supported_claims": sum(c["supported"] for c in claims),
        "scope": "Human judges correctness and entailment; code checks answer identity and evidence provenance only.",
    }
