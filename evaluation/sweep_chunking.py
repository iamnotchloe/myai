#!/usr/bin/env python3
"""Controlled chunk-size/overlap sweep for the MyAI retrieval pipeline.

Default runs evaluate development candidates only and save a frozen manifest.
An explicit --evaluate-frozen run evaluates only its selected and baseline
configurations on the unchanged test split. No remote API is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from myai_rag.chunking import AdaptiveChunkConfig, adaptive_split_documents
from myai_rag.retrieval import RetrievalEngine, tokenize_chinese_bm25, tokenize_whitespace
from myai_rag.retrieval_config import RetrievalConfig


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "data" / "documents"
DATASET = ROOT / "evaluation" / "datasets" / "golden_dataset_v2.jsonl"
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME_OR_PATH", "BAAI/bge-small-zh-v1.5")

RETRIEVAL_CONFIG = RetrievalConfig.from_env()
RRF_K = RETRIEVAL_CONFIG.rrf_k
DENSE_WEIGHT = RETRIEVAL_CONFIG.dense_rrf_weight
BM25_WEIGHT = RETRIEVAL_CONFIG.bm25_rrf_weight
KS = (1, 3, 5, 10)
DENSE_TOP_K = RETRIEVAL_CONFIG.dense_top_k
BM25_TOP_K = RETRIEVAL_CONFIG.bm25_top_k
SELECTION_RULE = "dev NDCG@3, Recall@3, lexical slot complete@3, slot coverage@3, MRR@3, then fewer chunks"


@dataclass(frozen=True)
class SweepConfig:
    label: str
    strategy: str
    max_size: int
    overlap: int


def configurations() -> list[SweepConfig]:
    configs = [SweepConfig("fixed_chars_500_80", "fixed_chars", 500, 80)]
    for max_tokens in (256, 320, 384, 448, 480, 510):
        for overlap_tokens in (0, 32, 64, 96, 128):
            if overlap_tokens < max_tokens:
                configs.append(
                    SweepConfig(
                        f"adaptive_tokens_{max_tokens}_{overlap_tokens}",
                        "adaptive_tokens",
                        max_tokens,
                        overlap_tokens,
                    )
                )
    return configs


def company_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_", "", stem)
    return re.sub(r"_report$", "", stem, flags=re.IGNORECASE)


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().casefold()


def load_pages(documents_dir: Path = DOCUMENTS_DIR) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    pages: list[Document] = []
    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        for page in PyPDFLoader(str(pdf_path)).load():
            page.metadata["company"] = company_name_from_filename(pdf_path.name)
            page.metadata["source_file"] = pdf_path.name
            pages.append(page)
    return pages


def dedupe_chunks(chunks: list[Document]) -> list[Document]:
    result: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        fingerprint = (
            str(chunk.metadata.get("company", "")),
            normalized_text(chunk.page_content),
        )
        if not fingerprint[1] or fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(chunk)
    return result


def build_chunks(
    pages: list[Document], config: SweepConfig, count_tokens
) -> tuple[list[Document], float]:
    started = time.perf_counter()
    if config.strategy == "fixed_chars":
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.max_size,
            chunk_overlap=config.overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = splitter.split_documents(pages)
    else:
        chunks = adaptive_split_documents(
            pages,
            AdaptiveChunkConfig(
                max_tokens=config.max_size,
                overlap_tokens=config.overlap,
            ),
            count_tokens=count_tokens,
        )
    return dedupe_chunks(chunks), time.perf_counter() - started


def load_cases(dataset: Path = DATASET) -> list[dict]:
    with dataset.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_splits(cases: list[dict]) -> None:
    seen_ids: set[str] = set()
    questions = {"dev": set(), "test": set()}
    for case in cases:
        split = case.get("split")
        query_id = case.get("query_id")
        if split not in questions or not query_id or query_id in seen_ids:
            raise ValueError("Every case must have a unique query_id and explicit dev/test split")
        seen_ids.add(query_id)
        questions[split].add(normalized_text(case["question"]))
    if not all(questions.values()) or questions["dev"] & questions["test"]:
        raise ValueError("Development and test splits must be nonempty and have distinct questions")
    for split in questions:
        if not any(case.get("answerable") and case["split"] == split for case in cases):
            raise ValueError(f"{split} needs answerable cases")


def object_digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_manifest(dataset: Path, documents_dir: Path, cases: list[dict]) -> dict:
    validate_splits(cases)
    documents = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(documents_dir.glob("*.pdf"))}
    if not documents:
        raise ValueError("No PDFs found for the experiment")
    return {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "split_sha256": {split: object_digest(sorted(
            [case for case in cases if case["split"] == split], key=lambda case: case["query_id"]
        )) for split in ("dev", "test")},
        "pdf_sha256": documents,
    }


def retrieval_settings() -> dict:
    return {**RETRIEVAL_CONFIG.to_dict(), "dense_weight": DENSE_WEIGHT, "bm25_weight": BM25_WEIGHT,
            "company_filter": True, "ks": list(KS)}


def freeze_selection(selected: dict, baseline: dict, inputs: dict, model: str) -> dict:
    manifest = {"schema_version": 1, "selected_config": selected, "baseline_config": baseline,
                "inputs": inputs, "model": model, "retrieval": retrieval_settings(),
                "selection_rule": SELECTION_RULE}
    manifest["manifest_sha256"] = object_digest(manifest)
    return manifest


def validate_frozen_manifest(manifest: dict, inputs: dict, model: str) -> list[SweepConfig]:
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != object_digest(content):
        raise ValueError("Frozen manifest integrity check failed")
    if manifest.get("schema_version") != 1 or manifest.get("inputs") != inputs:
        raise ValueError("Dataset/PDF inputs differ from the frozen development selection")
    if manifest.get("model") != model or manifest.get("retrieval") != retrieval_settings():
        raise ValueError("Model/retrieval settings differ from the frozen development selection")
    allowed = {config.label: asdict(config) for config in configurations()}
    configs = []
    for key in ("baseline_config", "selected_config"):
        value = manifest[key]
        if allowed.get(value["label"]) != value:
            raise ValueError("Frozen configuration is outside the declared development grid")
        config = SweepConfig(**value)
        if config not in configs:
            configs.append(config)
    if manifest["baseline_config"] != asdict(configurations()[0]):
        raise ValueError("Baseline must stay fixed at characters 500/80")
    return configs


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def tokenize_char_bigram(text: str) -> list[str]:
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9]+(?:[._%-][a-z0-9]+)*", lowered)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese_tokens: list[str] = []
    for run in chinese_runs:
        chinese_tokens.extend(run)
        chinese_tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return latin_tokens + chinese_tokens


def page_key(document: Document) -> tuple[str, int]:
    return (
        str(document.metadata.get("source_file", "")),
        int(document.metadata.get("page", 0)) + 1,
    )


def dedupe_page_ranking(indices: list[int], chunks: list[Document]) -> list[tuple[str, int]]:
    pages: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for index in indices:
        key = page_key(chunks[index])
        if key not in seen:
            pages.append(key)
            seen.add(key)
    return pages


def rrf_indices(dense: list[int], bm25: list[int]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking, weight in ((dense, DENSE_WEIGHT), (bm25, BM25_WEIGHT)):
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0.0) + weight / (RRF_K + rank)
    return [index for index, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def gold_relevance(case: dict) -> dict[tuple[str, int], int]:
    return {
        (str(page["source_file"]), int(page["page_number"])): int(
            page.get("relevance_grade", 3)
        )
        for page in case.get("gold_pages", [])
    }


def ranking_metrics(ranked_pages: list[tuple[str, int]], case: dict, k: int) -> dict[str, float]:
    relevance = gold_relevance(case)
    relevant_pages = set(relevance)
    top_k = ranked_pages[:k]
    hits = relevant_pages.intersection(top_k)
    first_rank = next(
        (rank for rank, page in enumerate(top_k, 1) if page in relevant_pages), None
    )
    hit_count = 0
    precisions = []
    for rank, page in enumerate(top_k, 1):
        if page in relevant_pages:
            hit_count += 1
            precisions.append(hit_count / rank)
    average_precision = sum(precisions) / len(relevant_pages) if relevant_pages else 0.0
    dcg = sum(
        (2 ** relevance.get(page, 0) - 1) / math.log2(rank + 1)
        for rank, page in enumerate(top_k, 1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return {
        "hit": float(bool(hits)),
        "recall": len(hits) / len(relevant_pages) if relevant_pages else 0.0,
        "precision": len(hits) / k,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "map": average_precision,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def lexical_slot_coverage(case: dict, indices: list[int], chunks: list[Document], k: int) -> tuple[float, float]:
    checks = case.get("answer_checks", [])
    if not checks:
        return 0.0, 0.0
    context = normalized_text("\n".join(chunks[index].page_content for index in indices[:k]))
    covered = 0
    for alternatives in checks:
        if any(normalized_text(str(alternative)) in context for alternative in alternatives):
            covered += 1
    ratio = covered / len(checks)
    return ratio, float(covered == len(checks))


def aggregate(rows: list[dict], k: int) -> dict[str, float]:
    keys = ("hit", "recall", "precision", "mrr", "map", "ndcg", "slot_coverage", "slot_complete")
    return {key: statistics.mean(row[key] for row in rows) if rows else 0.0 for key in keys}


def percentile(values: list[int], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile_value))


def evaluate_config(
    config: SweepConfig,
    chunks: list[Document],
    chunk_vectors: np.ndarray,
    query_vectors: dict[str, np.ndarray],
    cases: list[dict],
    split: str,
) -> dict:
    selected_cases = [
        case for case in cases if case.get("split") == split and case.get("answerable")
    ]
    tokenizer = tokenize_chinese_bm25 if RETRIEVAL_CONFIG.bm25_tokenizer == "char-bigram" else tokenize_whitespace
    bm25 = BM25Okapi([tokenizer(chunk.page_content) for chunk in chunks])
    vectors_by_question = {case["question"]: query_vectors[case["query_id"]] for case in selected_cases}

    class PrecomputedVectorStore:
        def similarity_search_with_score(self, question, k):
            distances = np.sum((chunk_vectors - vectors_by_question[question]) ** 2, axis=1)
            indices = np.argsort(distances, kind="stable")[:k]
            return [(chunks[index], float(distances[index])) for index in indices]

    engine = RetrievalEngine(chunks, PrecomputedVectorStore(), bm25, RETRIEVAL_CONFIG)
    chunk_indices = {id(chunk): index for index, chunk in enumerate(chunks)}
    rows_by_k: dict[int, list[dict]] = {k: [] for k in KS}
    per_query_at_3: list[dict] = []
    for case in selected_cases:
        question = case["question"]
        rankings = engine.search(question)
        fused_indices = [chunk_indices[id(item.document)] for item in rankings["rrf"]]
        ranked_pages = dedupe_page_ranking(fused_indices, chunks)
        for k in KS:
            row = ranking_metrics(ranked_pages, case, k)
            slot_coverage, slot_complete = lexical_slot_coverage(
                case, fused_indices, chunks, k
            )
            row["slot_coverage"] = slot_coverage
            row["slot_complete"] = slot_complete
            rows_by_k[k].append(row)
            if k == 3:
                per_query_at_3.append({"query_id": case["query_id"], **row})
    return {
        "split": split,
        "answerable_queries": len(selected_cases),
        "metrics": {str(k): aggregate(rows, k) for k, rows in rows_by_k.items()},
        "per_query_at_3": per_query_at_3,
    }


def selection_key(result: dict) -> tuple[float, ...]:
    metrics = result["dev"]["metrics"]["3"]
    return (
        metrics["ndcg"],
        metrics["recall"],
        metrics["slot_complete"],
        metrics["slot_coverage"],
        metrics["mrr"],
        -result["chunk_stats"]["count"],
    )


def paired_bootstrap(candidate: dict, baseline: dict, samples: int = 10_000, split: str = "dev") -> dict:
    candidate_rows = {
        row["query_id"]: row for row in candidate[split]["per_query_at_3"]
    }
    baseline_rows = {
        row["query_id"]: row for row in baseline[split]["per_query_at_3"]
    }
    if candidate_rows.keys() != baseline_rows.keys() or not candidate_rows:
        raise ValueError("Paired comparison needs the same nonempty query IDs")
    query_ids = sorted(candidate_rows)
    rng = np.random.default_rng(20260901)
    report: dict[str, dict] = {}
    for metric in ("ndcg", "recall", "mrr", "slot_coverage", "slot_complete"):
        differences = np.asarray(
            [candidate_rows[query_id][metric] - baseline_rows[query_id][metric] for query_id in query_ids],
            dtype=np.float64,
        )
        sampled_indices = rng.integers(0, len(differences), size=(samples, len(differences)))
        sampled_means = differences[sampled_indices].mean(axis=1)
        report[metric] = {
            "mean_difference": float(differences.mean()),
            "ci95": [
                float(np.percentile(sampled_means, 2.5)),
                float(np.percentile(sampled_means, 97.5)),
            ],
            "wins": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
            "losses": int(np.sum(differences < 0)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MyAI chunk参数受控扫描")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_DIR)
    parser.add_argument("--output", type=Path, help="New output path; existing reports are never overwritten")
    parser.add_argument("--freeze-output", type=Path, help="Save selected configuration and data hashes after dev-only sweep")
    parser.add_argument("--evaluate-frozen", type=Path, help="Test ONLY baseline and selected configuration in this frozen manifest")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    inputs = input_manifest(args.dataset, args.documents, cases)
    split = "test" if args.evaluate_frozen else "dev"
    output = args.output or ROOT / "evaluation/results" / (
        "chunk_frozen_test.json" if args.evaluate_frozen else "chunk_parameter_sweep_dev.json"
    )
    freeze_output = args.freeze_output or output.with_suffix(".frozen.json")
    if args.evaluate_frozen and args.freeze_output:
        parser.error("--freeze-output cannot be combined with --evaluate-frozen")
    if output.exists() or (not args.evaluate_frozen and freeze_output.exists()):
        parser.error("Output already exists; supply a new path to preserve previous experiments")
    if not args.evaluate_frozen and output.resolve() == freeze_output.resolve():
        parser.error("--output and --freeze-output must be different files")
    frozen = None
    if args.evaluate_frozen:
        frozen = json.loads(args.evaluate_frozen.read_text(encoding="utf-8"))
        configs = validate_frozen_manifest(frozen, inputs, MODEL_NAME)
    else:
        configs = configurations()

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))

    pages = load_pages(args.documents)
    # Never embed or evaluate held-out questions during parameter selection.
    answerable = [case for case in cases if case.get("answerable") and case["split"] == split]
    questions = [case["question"] for case in answerable]
    query_matrix = model.encode(
        questions,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)
    query_vectors = {
        case["query_id"]: query_matrix[index] for index, case in enumerate(answerable)
    }

    results = []
    for index, config in enumerate(configs, 1):
        print(f"[{index:02d}/{len(configs)}] {config.label} ({split})", flush=True)
        chunks, split_seconds = build_chunks(pages, config, count_tokens)
        encode_started = time.perf_counter()
        chunk_vectors = model.encode(
            [chunk.page_content for chunk in chunks],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        encode_seconds = time.perf_counter() - encode_started
        token_lengths = [count_tokens(chunk.page_content) for chunk in chunks]
        actual_overlaps = [
            int(chunk.metadata.get("chunk_overlap_tokens", 0)) for chunk in chunks
        ] if config.strategy == "adaptive_tokens" else []
        result = {
            "config": asdict(config),
            "chunk_stats": {
                "count": len(chunks),
                "mean_tokens": statistics.mean(token_lengths),
                "p95_tokens": percentile(token_lengths, 95),
                "max_tokens_observed": max(token_lengths),
                "estimated_vector_bytes": len(chunks) * int(chunk_vectors.shape[1]) * 4,
                "split_seconds": split_seconds,
                "embedding_seconds": encode_seconds,
                "actual_overlap": {
                    "mean_tokens": statistics.mean(actual_overlaps),
                    "p95_tokens": percentile(actual_overlaps, 95),
                    "max_tokens": max(actual_overlaps),
                    "zero_fraction": sum(value == 0 for value in actual_overlaps)
                    / len(actual_overlaps),
                }
                if actual_overlaps
                else None,
            },
            split: evaluate_config(config, chunks, chunk_vectors, query_vectors, answerable, split),
        }
        results.append(result)

    by_label = {result["config"]["label"]: result for result in results}
    fixed_result = by_label["fixed_chars_500_80"]
    if frozen:
        selected = frozen["selected_config"]["label"]
        comparisons = {"selected_vs_fixed_500_80": paired_bootstrap(by_label[selected], fixed_result, split="test")}
    else:
        ranked = sorted(results, key=selection_key, reverse=True)
        selected = ranked[0]["config"]["label"]
        selected_result = by_label[selected]
        runner_up_result = ranked[1]
        comparisons = {
            "selected_vs_fixed_500_80": paired_bootstrap(selected_result, fixed_result),
            "selected_vs_runner_up": {"runner_up": runner_up_result["config"]["label"],
                "metrics": paired_bootstrap(selected_result, runner_up_result)},
        }
        frozen = freeze_selection(selected_result["config"], fixed_result["config"], inputs, MODEL_NAME)
    report = {
        "experiment": {
            "phase": "frozen_test" if args.evaluate_frozen else "dev_selection",
            "dataset": str(args.dataset),
            "documents": len(inputs["pdf_sha256"]),
            "pages": len(pages),
            "model": MODEL_NAME,
            "device": device,
            **retrieval_settings(),
            "selection_rule": SELECTION_RULE,
            "selected_on_dev": selected,
            "manifest_sha256": frozen["manifest_sha256"],
            "limitations": [
                "Offline retrieval-only evaluation; no reranker or LLM generation.",
                "Page labels and lexical answer checks come from a small synthetic project dataset.",
                "The winner is best only among the tested grid, not universally optimal.",
            ],
        },
        f"paired_{split}_comparisons": comparisons,
        "results": results,
    }
    if not args.evaluate_frozen:
        report["ranked_by_dev"] = [result["config"]["label"] for result in ranked]
        write_new_json(freeze_output, frozen)
        print(f"Frozen selection: {freeze_output}")
    write_new_json(output, report)
    print(f"Selected on dev: {selected}")
    print(f"Saved {split} results: {output}")


if __name__ == "__main__":
    main()
