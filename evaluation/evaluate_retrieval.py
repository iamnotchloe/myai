#!/usr/bin/env python3
"""对 MyAI 的 Dense、BM25 和 RRF 召回做离线评测。

评测单位是 PDF 页面。一个页面若命中多个 Chunk，只计为同一条页面结果。
该脚本不调用大模型和 SiliconFlow API。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable

import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("golden_test_set.jsonl")
DEFAULT_METADATA = PROJECT_DIR / "faiss_index" / "documents_metadata.json"
DEFAULT_INDEX = PROJECT_DIR / "faiss_index"


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSON格式错误: {exc}") from exc
    return cases


def page_key_from_metadata(metadata: dict) -> tuple[str, int]:
    return str(metadata.get("source_file", "")), int(metadata.get("page", 0)) + 1


def gold_pages(case: dict) -> set[tuple[str, int]]:
    return {
        (str(page["source_file"]), int(page["page_number"]))
        for page in case.get("gold_pages", [])
    }


def relevance_by_page(case: dict) -> dict[tuple[str, int], int]:
    """返回页面相关性等级；V1 未标等级时按完全相关处理。"""
    return {
        (str(page["source_file"]), int(page["page_number"])): int(
            page.get("relevance_grade", 3)
        )
        for page in case.get("gold_pages", [])
    }


def dedupe_pages(pages: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
    result = []
    seen = set()
    for page in pages:
        if page not in seen:
            result.append(page)
            seen.add(page)
    return result


def rrf_page_ranking(
    dense: list[tuple[str, int]],
    bm25: list[tuple[str, int]],
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> list[tuple[str, int]]:
    scores: defaultdict[tuple[str, int], float] = defaultdict(float)
    for ranking, weight in ((dense, dense_weight), (bm25, bm25_weight)):
        for rank, page in enumerate(ranking, 1):
            scores[page] += weight / (rrf_k + rank)
    return [
        page for page, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def tokenize_current(text: str) -> list[str]:
    """复现当前 LangChain BM25 默认的空白分词行为。"""
    return text.lower().split()


def tokenize_char_bigram(text: str) -> list[str]:
    """无需额外词典的中文字符 unigram + bigram 基线。"""
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9]+(?:[._%-][a-z0-9]+)*", lowered)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese_tokens: list[str] = []
    for run in chinese_runs:
        chinese_tokens.extend(run)
        chinese_tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return latin_tokens + chinese_tokens


def metric_for_query(
    ranked_pages: list[tuple[str, int]], relevance: dict[tuple[str, int], int], k: int
) -> dict[str, float]:
    relevant_pages = set(relevance)
    top_k = ranked_pages[:k]
    hits = relevant_pages.intersection(top_k)
    first_rank = next(
        (rank for rank, page in enumerate(top_k, 1) if page in relevant_pages), None
    )
    precisions_at_relevant_ranks = []
    hit_count = 0
    for rank, ranked_page in enumerate(top_k, 1):
        if ranked_page in relevant_pages:
            hit_count += 1
            precisions_at_relevant_ranks.append(hit_count / rank)
    average_precision = (
        sum(precisions_at_relevant_ranks) / len(relevant_pages)
        if relevant_pages
        else 0.0
    )

    dcg = sum(
        (2 ** relevance.get(ranked_page, 0) - 1) / math.log2(rank + 1)
        for rank, ranked_page in enumerate(top_k, 1)
    )
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, 1)
    )

    return {
        "hit": 1.0 if hits else 0.0,
        "recall": len(hits) / len(relevant_pages) if relevant_pages else 0.0,
        "precision": len(hits) / k,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "map": average_precision,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def aggregate(
    cases: list[dict], rankings: dict[str, list[tuple[str, int]]], ks: list[int]
) -> dict[str, dict[str, float]]:
    answerable_cases = [case for case in cases if case.get("answerable")]
    summary: dict[str, dict[str, float]] = {}
    for k in ks:
        per_query = [
            metric_for_query(
                rankings[case["query_id"]], relevance_by_page(case), k
            )
            for case in answerable_cases
        ]
        summary[str(k)] = {
            name: mean(row[name] for row in per_query)
            for name in ("hit", "recall", "precision", "mrr", "map", "ndcg")
        }
    return summary


def format_page(page: tuple[str, int]) -> str:
    return f"{page[0]}#p{page[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="MyAI离线召回评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--tokenizer", choices=("current", "char-bigram"), default="current")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--sweep-rrf", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))
    cases = load_jsonl(args.dataset)
    if args.split:
        cases = [case for case in cases if case.get("split") == args.split]
    answerable_cases = [case for case in cases if case.get("answerable")]
    with args.metadata.open("r", encoding="utf-8") as stream:
        chunks = json.load(stream)

    tokenizer: Callable[[str], list[str]] = (
        tokenize_current if args.tokenizer == "current" else tokenize_char_bigram
    )
    bm25 = BM25Okapi([tokenizer(chunk["content"]) for chunk in chunks])
    bm25_rankings: dict[str, list[tuple[str, int]]] = {}
    for case in answerable_cases:
        scores = bm25.get_scores(tokenizer(case["question"]))
        indices = sorted(range(len(chunks)), key=lambda index: float(scores[index]), reverse=True)
        bm25_rankings[case["query_id"]] = dedupe_pages(
            page_key_from_metadata(chunks[index]["metadata"]) for index in indices
        )

    all_rankings: dict[str, dict[str, list[tuple[str, int]]]] = {"bm25": bm25_rankings}
    if not args.skip_dense:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model_name = os.getenv("EMBEDDING_MODEL_NAME_OR_PATH", "BAAI/bge-small-zh-v1.5")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
        vector_db = FAISS.load_local(
            args.index,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        dense_rankings: dict[str, list[tuple[str, int]]] = {}
        dense_k = min(len(chunks), max(max(args.k), 50))
        for case in answerable_cases:
            docs_and_scores = vector_db.similarity_search_with_score(case["question"], k=dense_k)
            dense_rankings[case["query_id"]] = dedupe_pages(
                page_key_from_metadata(document.metadata)
                for document, _score in docs_and_scores
            )
        all_rankings["dense"] = dense_rankings

        selected_rrf_k = args.rrf_k
        selected_dense_weight = args.dense_weight
        selected_bm25_weight = args.bm25_weight
        if args.sweep_rrf:
            sweep_rows = []
            target_k = 3 if 3 in args.k else min(args.k)
            report_k = max(args.k)
            for candidate_k in (1, 5, 10, 30, 60, 100):
                for dense_weight in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
                    rankings = {
                        case["query_id"]: rrf_page_ranking(
                            dense_rankings[case["query_id"]],
                            bm25_rankings[case["query_id"]],
                            candidate_k,
                            dense_weight,
                            1.0,
                        )
                        for case in answerable_cases
                    }
                    metrics = aggregate(answerable_cases, rankings, args.k)
                    objective = (
                        metrics[str(target_k)]["ndcg"],
                        metrics[str(target_k)]["recall"],
                        metrics[str(report_k)]["map"],
                    )
                    sweep_rows.append((objective, candidate_k, dense_weight, metrics))
            sweep_rows.sort(key=lambda row: row[0], reverse=True)
            print("\nRRF 开发集参数扫描 Top 8（按 NDCG@3、Recall@3、MAP@最大K 排序）:")
            for objective, candidate_k, dense_weight, _metrics in sweep_rows[:8]:
                print(
                    f"  rrf_k={candidate_k:>3} dense_weight={dense_weight:.2f} "
                    f"bm25_weight=1.00 objective={tuple(round(v, 4) for v in objective)}"
                )
            _objective, selected_rrf_k, selected_dense_weight, selected_metrics = sweep_rows[0]
            selected_bm25_weight = 1.0
            print(
                f"  选中: rrf_k={selected_rrf_k}, dense_weight={selected_dense_weight}, "
                f"bm25_weight={selected_bm25_weight}"
            )

        rrf_rankings = {
            case["query_id"]: rrf_page_ranking(
                dense_rankings[case["query_id"]],
                bm25_rankings[case["query_id"]],
                selected_rrf_k,
                selected_dense_weight,
                selected_bm25_weight,
            )
            for case in answerable_cases
        }
        all_rankings["rrf"] = rrf_rankings
        args.rrf_k = selected_rrf_k
        args.dense_weight = selected_dense_weight
        args.bm25_weight = selected_bm25_weight

    report = {
        "dataset": str(args.dataset),
        "answerable_queries": len(answerable_cases),
        "tokenizer": args.tokenizer,
        "rrf_k": args.rrf_k,
        "dense_weight": args.dense_weight,
        "bm25_weight": args.bm25_weight,
        "metrics": {
            method: aggregate(answerable_cases, rankings, args.k)
            for method, rankings in all_rankings.items()
        },
    }

    print(f"题目数: {len(cases)}，可回答题: {len(answerable_cases)}")
    print(f"BM25分词: {args.tokenizer}")
    print("\n| 方法 | K | Hit@K | Recall@K | Precision@K | MAP@K | MRR@K | NDCG@K |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method, by_k in report["metrics"].items():
        for k, metrics in by_k.items():
            print(
                f"| {method} | {k} | {metrics['hit']:.3f} | {metrics['recall']:.3f} | "
                f"{metrics['precision']:.3f} | {metrics['map']:.3f} | "
                f"{metrics['mrr']:.3f} | {metrics['ndcg']:.3f} |"
            )

    if args.show_failures:
        max_k = max(args.k)
        print(f"\nTop{max_k} 未命中的题目:")
        for method, rankings in all_rankings.items():
            failures = []
            for case in answerable_cases:
                ranked = rankings[case["query_id"]][:max_k]
                if not gold_pages(case).intersection(ranked):
                    failures.append(
                        {
                            "query_id": case["query_id"],
                            "question": case["question"],
                            "gold": [format_page(page) for page in sorted(gold_pages(case))],
                            "top": [format_page(page) for page in ranked[:3]],
                        }
                    )
            print(f"- {method}: {len(failures)}题")
            for failure in failures:
                print(f"  {failure['query_id']} {failure['question']}")
                print(f"    gold={failure['gold']}")
                print(f"    top3={failure['top']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
