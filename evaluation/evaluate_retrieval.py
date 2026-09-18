#!/usr/bin/env python3
"""Evaluate production chunk recall/fusion; deduplicate pages only for scoring.

Defaults use the development set and the API's shared retrieval configuration.
RRF sweeps accept development rows only. Test runs require a frozen config.
No generation or SiliconFlow API calls are made by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Iterable

from myai_rag.config import CACHE_DIR, INDEX_DIR
from myai_rag.retrieval import (
    RankedDocument, RetrievalEngine, rrf_fuse, tokenize_chinese_bm25, tokenize_whitespace,
)
from myai_rag.retrieval_config import RetrievalConfig


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("datasets") / "dev_set_v2.jsonl"
DEFAULT_METADATA = INDEX_DIR / "documents_metadata.json"
DEFAULT_INDEX = INDEX_DIR


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return cases


def page_key_from_metadata(metadata: dict) -> tuple[str, int]:
    return str(metadata.get("source_file", "")), int(metadata.get("page", 0)) + 1


def gold_pages(case: dict) -> set[tuple[str, int]]:
    return {(str(page["source_file"]), int(page["page_number"])) for page in case.get("gold_pages", [])}


def relevance_by_page(case: dict) -> dict[tuple[str, int], int]:
    return {
        (str(page["source_file"]), int(page["page_number"])): int(page.get("relevance_grade", 3))
        for page in case.get("gold_pages", [])
    }


def dedupe_pages(pages: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
    return list(dict.fromkeys(pages))


def pages_from_ranking(ranking: list[RankedDocument]) -> list[tuple[str, int]]:
    return dedupe_pages(page_key_from_metadata(item.document.metadata) for item in ranking)


# Backwards-compatible names used in historical analysis notebooks.
tokenize_current = tokenize_whitespace
tokenize_char_bigram = tokenize_chinese_bm25


def metric_for_query(
    ranked_pages: list[tuple[str, int]], relevance: dict[tuple[str, int], int], k: int
) -> dict[str, float]:
    if k < 1:
        raise ValueError("Metric K must be positive")
    relevant_pages = {page for page, grade in relevance.items() if grade > 0}
    top_k = dedupe_pages(ranked_pages)[:k]
    hits = relevant_pages.intersection(top_k)
    first_rank = next((rank for rank, page in enumerate(top_k, 1) if page in relevant_pages), None)
    precisions = []
    hit_count = 0
    for rank, page in enumerate(top_k, 1):
        if page in relevant_pages:
            hit_count += 1
            precisions.append(hit_count / rank)
    dcg = sum(
        (2 ** max(0, relevance.get(page, 0)) - 1) / math.log2(rank + 1)
        for rank, page in enumerate(top_k, 1)
    )
    ideal_grades = sorted((max(0, grade) for grade in relevance.values()), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    return {
        "hit": float(bool(hits)),
        "recall": len(hits) / len(relevant_pages) if relevant_pages else 0.0,
        "precision": len(hits) / k,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "map": sum(precisions) / len(relevant_pages) if relevant_pages else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def aggregate(cases: list[dict], rankings: dict[str, list[tuple[str, int]]], ks: list[int]) -> dict:
    answerable = [case for case in cases if case.get("answerable")]
    summary = {}
    for k in ks:
        rows = [metric_for_query(rankings[case["query_id"]], relevance_by_page(case), k) for case in answerable]
        summary[str(k)] = {
            name: mean(row[name] for row in rows) if rows else None
            for name in ("hit", "recall", "precision", "mrr", "map", "ndcg")
        }
    return summary


def validate_split_policy(cases: list[dict], *, sweep: bool, frozen: bool) -> None:
    if not cases:
        raise ValueError("No dataset rows selected")
    if not any(case.get("answerable") for case in cases):
        raise ValueError("Retrieval evaluation requires at least one answerable row")
    splits = {case.get("split") for case in cases}
    if sweep and splits != {"dev"}:
        raise ValueError("RRF sweeps require development rows only; test/unspecified rows are forbidden")
    if "test" in splits and not frozen:
        raise ValueError("Test evaluation requires --config with a frozen development configuration")
    if "test" in splits and len(splits) != 1:
        raise ValueError("Run the test split separately from development rows")


def frozen_payload_digest(payload: dict) -> str:
    """Detect accidental edits to either parameters or frozen artifact references."""
    content = {key: value for key, value in payload.items() if key != "integrity_sha256"}
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_frozen_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("selection_split") != "dev":
        raise ValueError("Expected a configuration created with --freeze-config on development data")
    digest = payload.get("integrity_sha256")
    if not isinstance(digest, str) or not hmac.compare_digest(digest, frozen_payload_digest(payload)):
        raise ValueError("Frozen configuration integrity mismatch; run a new development freeze")
    if set(payload.get("index_files", {})) != {"index.faiss", "index.pkl"}:
        raise ValueError("Frozen configuration must identify both FAISS index files")
    if not payload.get("metadata_sha256") or not payload.get("embedding_model"):
        raise ValueError("Frozen configuration must identify metadata and embedding model")
    return payload


def load_frozen_config(path: Path) -> RetrievalConfig:
    payload = load_frozen_manifest(path)
    return RetrievalConfig(**payload["configuration"])


def validate_frozen_artifacts(
    path: Path, *, metadata: Path, index: Path, embedding_model: str
) -> None:
    """Reject corpus, vector index, or model drift before loading any model."""
    payload = load_frozen_manifest(path)
    if payload["embedding_model"] != embedding_model:
        raise ValueError("Embedding model differs from the frozen development run")
    if file_sha256(metadata) != payload["metadata_sha256"]:
        raise ValueError("Metadata differs from the frozen development run")
    for name, expected_hash in payload["index_files"].items():
        artifact = index / name
        if not artifact.is_file() or file_sha256(artifact) != expected_hash:
            raise ValueError(f"{name} differs from the frozen development run")


def frozen_manifest_from_report(report: dict) -> dict:
    payload = {
        "schema_version": 2, "selection_split": "dev",
        "configuration": report["configuration"],
        "dataset_sha256": report["dataset_sha256"],
        "selected_query_ids": report["selected_query_ids"],
        "metadata_sha256": report["metadata_sha256"],
        "index_files": report["index_files"],
        "embedding_model": report["embedding_model"],
        "created_at": report["created_at"],
    }
    payload["integrity_sha256"] = frozen_payload_digest(payload)
    return payload


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_page(page: tuple[str, int]) -> str:
    return f"{page[0]}#p{page[1]}"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
    runtime = RetrievalConfig.from_env()
    parser = argparse.ArgumentParser(description="MyAI production-aligned offline retrieval evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, help="Defaults to documents_metadata.json inside --index")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--tokenizer", choices=("current", "char-bigram"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--dense-weight", type=float)
    parser.add_argument("--bm25-weight", type=float)
    parser.add_argument("--dense-top-k", type=int)
    parser.add_argument("--bm25-top-k", type=int)
    parser.add_argument("--fused-top-k", type=int)
    parser.add_argument("--sweep-rrf", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--config", type=Path, help="Frozen development configuration JSON")
    parser.add_argument("--freeze-config", type=Path, help="Save selected development configuration")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.metadata = args.metadata or args.index / "documents_metadata.json"
    if min(args.k) < 1:
        parser.error("--k values must be positive")
    if args.sweep_rrf and (args.skip_dense or args.config):
        parser.error("--sweep-rrf cannot be combined with --skip-dense or --config")

    overrides = {
        name: value for name, value in {
            "rrf_k": args.rrf_k, "dense_rrf_weight": args.dense_weight,
            "bm25_rrf_weight": args.bm25_weight, "dense_top_k": args.dense_top_k,
            "bm25_top_k": args.bm25_top_k, "fused_top_k": args.fused_top_k,
            "bm25_tokenizer": args.tokenizer,
        }.items() if value is not None
    }
    if args.config and overrides:
        parser.error("A frozen --config cannot be overridden by parameter flags")
    if args.config and args.skip_dense:
        parser.error("A frozen --config requires its complete Dense + BM25 retrieval pipeline")
    config = load_frozen_config(args.config) if args.config else replace(runtime, **overrides)
    model_name = os.getenv("EMBEDDING_MODEL_NAME_OR_PATH", "BAAI/bge-small-zh-v1.5")
    if args.config:
        try:
            validate_frozen_artifacts(args.config, metadata=args.metadata, index=args.index,
                                      embedding_model=model_name)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    cases = load_jsonl(args.dataset)
    if args.split:
        cases = [case for case in cases if case.get("split") == args.split]
    try:
        validate_split_policy(cases, sweep=args.sweep_rrf, frozen=bool(args.config))
    except ValueError as exc:
        parser.error(str(exc))
    if args.freeze_config and ({case.get("split") for case in cases} != {"dev"} or args.skip_dense):
        parser.error("--freeze-config requires a full development retrieval run")

    from langchain_core.documents import Document
    from rank_bm25 import BM25Okapi

    chunks = json.loads(args.metadata.read_text(encoding="utf-8"))
    documents = [Document(page_content=item["content"], metadata=item["metadata"]) for item in chunks]
    if not documents:
        parser.error("The metadata file contains no chunks")
    tokenizer = tokenize_chinese_bm25 if config.bm25_tokenizer == "char-bigram" else tokenize_whitespace
    bm25 = BM25Okapi([tokenizer(document.page_content) for document in documents])
    vectorstore = None
    if not args.skip_dense:
        import torch
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
        vectorstore = FAISS.load_local(args.index, embeddings, allow_dangerous_deserialization=True)
    engine = RetrievalEngine(documents, vectorstore, bm25, config)
    answerable = [case for case in cases if case.get("answerable")]
    chunk_rankings = {case["query_id"]: engine.search(case["question"]) for case in answerable}
    sweep_rows = []
    if args.sweep_rrf:
        objective_k = 3 if 3 in args.k else min(args.k)
        for candidate_k in (1, 5, 10, 30, 60, 100):
            for dense_weight in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
                candidate = replace(config, rrf_k=candidate_k, dense_rrf_weight=dense_weight, bm25_rrf_weight=1.0)
                rankings = {
                    query_id: pages_from_ranking(rrf_fuse(
                        [stages["dense"], stages["bm25"]], candidate.rrf_k, candidate.fused_top_k,
                        (candidate.dense_rrf_weight, candidate.bm25_rrf_weight),
                    )) for query_id, stages in chunk_rankings.items()
                }
                metrics = aggregate(answerable, rankings, args.k)
                objective = [metrics[str(objective_k)]["ndcg"], metrics[str(objective_k)]["recall"], metrics[str(max(args.k))]["map"]]
                sweep_rows.append({"objective": objective, "configuration": candidate.to_dict(), "metrics": metrics})
        sweep_rows.sort(key=lambda row: row["objective"], reverse=True)
        config = RetrievalConfig(**sweep_rows[0]["configuration"])
        for stages in chunk_rankings.values():
            stages["rrf"] = rrf_fuse([stages["dense"], stages["bm25"]], config.rrf_k, config.fused_top_k, (config.dense_rrf_weight, config.bm25_rrf_weight))

    methods = ("bm25",) if args.skip_dense else ("dense", "bm25", "rrf")
    rankings = {
        method: {query_id: pages_from_ranking(stages[method]) for query_id, stages in chunk_rankings.items()}
        for method in methods
    }
    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset), "dataset_sha256": file_sha256(args.dataset),
        "selected_query_ids": [case["query_id"] for case in cases],
        "splits": sorted({str(case.get("split", "unspecified")) for case in cases}),
        "answerable_queries": len(answerable),
        "configuration": config.to_dict(),
        "production_configuration": runtime.to_dict(),
        "matches_production_retrieval": config == runtime and not args.skip_dense,
        "evaluation_scope": "recall_and_chunk_rrf; no routing, query rewrite, reranker or generation",
        "metric_unit": "unique PDF pages after production chunk candidate limits",
        "precision_denominator": "requested K, including missing positions",
        "metadata_sha256": file_sha256(args.metadata),
        "embedding_model": model_name if not args.skip_dense else None,
        "index_files": {name: file_sha256(args.index / name) for name in ("index.faiss", "index.pkl") if not args.skip_dense and (args.index / name).is_file()},
        "metrics": {method: aggregate(answerable, by_query, args.k) for method, by_query in rankings.items()},
        "per_query": [
            {"query_id": case["query_id"], "question": case["question"],
             "gold_pages": [format_page(page) for page in sorted(gold_pages(case))],
             "stages": {method: {
                 "pages": [format_page(page) for page in rankings[method][case["query_id"]]],
                 "chunk_count": len(chunk_rankings[case["query_id"]][method]),
                 "metrics": {str(k): metric_for_query(rankings[method][case["query_id"]], relevance_by_page(case), k) for k in args.k},
             } for method in methods}}
            for case in answerable
        ],
        "sweep": sweep_rows,
    }
    print(json.dumps({"configuration": report["configuration"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    if args.show_failures:
        for row in report["per_query"]:
            for method, stage in row["stages"].items():
                if stage["metrics"][str(max(args.k))]["recall"] < 1.0:
                    print(f"{method}: {row['query_id']} {row['question']} gold={row['gold_pages']} top={stage['pages'][:max(args.k)]}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.freeze_config:
        payload = frozen_manifest_from_report(report)
        args.freeze_config.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
