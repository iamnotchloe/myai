"""Regression checks for production/offline retrieval alignment without models."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from myai_rag.retrieval import (
    RankedDocument, RetrievalEngine, bm25_search, build_company_aliases,
    companies_mentioned_in, dedupe_documents, dense_search, document_key,
    rrf_fuse, tokenize_chinese_bm25,
)
from myai_rag.retrieval_config import RetrievalConfig


def doc(text, *, company="澜赋科技有限公司", page=0, start=0, source="a.pdf"):
    return SimpleNamespace(page_content=text, metadata={"company": company, "page": page, "start_index": start, "source_file": source})


class VectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def similarity_search_with_score(self, question, k):
        self.calls.append((question, k))
        return [(document, index * 0.1) for index, document in enumerate(self.documents[:k])]


class BM25:
    def __init__(self, scores):
        self.scores = scores
        self.tokens = None

    def get_scores(self, tokens):
        self.tokens = tokens
        return self.scores


def evaluation_module():
    path = Path(__file__).resolve().parents[1] / "evaluation" / "evaluate_retrieval.py"
    spec = importlib.util.spec_from_file_location("retrieval_evaluation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scoped_dense_search_does_not_lose_evidence_outside_global_top_k():
    other = doc("unrelated", company="蓝天旅游有限公司")
    target = doc("evidence")
    store = VectorStore([other, target])
    ranking = dense_search("澜赋科技", vectorstore=store, documents=[other, target], k=1, target_companies={"澜赋科技有限公司"})
    assert store.calls[0][1] == 2
    assert ranking[0].document is target
    assert ranking[0].rank == 1


def test_bm25_company_filter_and_nonpositive_scores_match_api():
    documents = [doc("other", company="蓝天旅游有限公司"), doc("hit"), doc("zero"), doc("negative")]
    ranking = bm25_search("净利润", bm25_model=BM25([10, 2, 0, -1]), documents=documents, k=3, target_companies={"澜赋科技有限公司"})
    assert [item.document.page_content for item in ranking] == ["hit"]
    assert "净利" in tokenize_chinese_bm25("净利润")


def test_chunk_identity_preserves_different_positions_and_sources():
    first = doc("same", start=0)
    copied = doc("same", start=0)
    later = doc("same", start=10)
    another = doc("same", source="b.pdf")
    assert dedupe_documents([first, copied, later, another]) == [first, later, another]


def test_rrf_uses_chunks_before_page_deduplication_and_counts_one_vote_per_route():
    first = doc("first", start=0)
    second = doc("second", start=30)
    other = doc("other", page=1)
    dense = [RankedDocument(first, 0.1, 1, "dense"), RankedDocument(second, 0.2, 2, "dense"), RankedDocument(other, 0.3, 3, "dense")]
    sparse = [RankedDocument(second, 5.0, 1, "bm25")]
    ranking = rrf_fuse([dense, sparse], limit=2)
    assert [item.document for item in ranking] == [second, first]
    assert evaluation_module().pages_from_ranking(ranking) == [("a.pdf", 1)]
    duplicated = rrf_fuse([[dense[0], dense[0]], []])
    assert duplicated[0].score == pytest.approx(2 / 6)


def test_engine_and_shared_operations_produce_identical_production_rankings():
    documents = [doc("company one"), doc("company two", company="蓝天旅游有限公司"), doc("more", page=1)]
    store = VectorStore(documents)
    sparse = BM25([1, 3, 2])
    config = RetrievalConfig(dense_top_k=1, bm25_top_k=2, fused_top_k=2)
    engine = RetrievalEngine(documents, store, sparse, config)
    result = engine.search("澜赋科技净利润")
    scope = {"澜赋科技有限公司"}
    dense = dense_search("澜赋科技净利润", vectorstore=store, documents=documents, k=1, target_companies=scope)
    bm25 = bm25_search("澜赋科技净利润", bm25_model=sparse, documents=documents, k=2, target_companies=scope)
    expected = rrf_fuse([dense, bm25], limit=2)
    assert [(document_key(item.document), item.score) for item in result["rrf"]] == [(document_key(item.document), item.score) for item in expected]


def test_defaults_and_aliases_match_document_baseline():
    config = RetrievalConfig.from_env({})
    assert (config.dense_top_k, config.bm25_top_k, config.fused_top_k, config.rrf_k) == (10, 20, 30, 5)
    assert (config.dense_rrf_weight, config.bm25_rrf_weight, config.bm25_tokenizer) == (2, 1, "char-bigram")
    aliases = build_company_aliases(["ACME研发有限公司", "澜赋科技有限公司"])
    assert set(companies_mentioned_in("acme与澜赋科技", aliases)) == {"澜赋科技有限公司", "ACME研发有限公司"}


@pytest.mark.parametrize("kwargs", [{"rrf_k": 0}, {"dense_top_k": -1}, {"dense_rrf_weight": float("nan")}, {"dense_rrf_weight": 0, "bm25_rrf_weight": 0}])
def test_invalid_configs_fail_before_model_startup(kwargs):
    with pytest.raises(ValueError):
        RetrievalConfig(**kwargs)


def test_test_rows_are_rejected_for_sweeps_and_require_frozen_configuration():
    evaluation = evaluation_module()
    rows = [{"answerable": True, "split": "test"}]
    with pytest.raises(ValueError, match="development rows only"):
        evaluation.validate_split_policy(rows, sweep=True, frozen=True)
    with pytest.raises(ValueError, match="frozen"):
        evaluation.validate_split_policy(rows, sweep=False, frozen=False)
    evaluation.validate_split_policy(rows, sweep=False, frozen=True)


def test_page_metrics_use_requested_k_and_ignore_zero_relevance_grades():
    evaluation = evaluation_module()
    actual = evaluation.metric_for_query([("a.pdf", 1), ("a.pdf", 1)], {("a.pdf", 1): 3, ("b.pdf", 2): 0}, 3)
    assert actual["recall"] == 1
    assert actual["precision"] == pytest.approx(1 / 3)
    assert actual["ndcg"] == 1


def freeze_fixture(tmp_path):
    evaluation = evaluation_module()
    metadata = tmp_path / "documents_metadata.json"
    metadata.write_text("[]", encoding="utf-8")
    for name in ("index.faiss", "index.pkl"):
        (tmp_path / name).write_bytes(name.encode())
    report = {
        "configuration": RetrievalConfig().to_dict(), "dataset_sha256": "dev-data-hash",
        "selected_query_ids": ["dev-1"], "metadata_sha256": evaluation.file_sha256(metadata),
        "index_files": {name: evaluation.file_sha256(tmp_path / name) for name in ("index.faiss", "index.pkl")},
        "embedding_model": "BAAI/bge-small-zh-v1.5", "created_at": "2026-09-18T00:00:00Z",
    }
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(evaluation.frozen_manifest_from_report(report)), encoding="utf-8")
    return evaluation, frozen, metadata


def test_frozen_manifest_accepts_exact_model_metadata_and_index(tmp_path):
    evaluation, frozen, metadata = freeze_fixture(tmp_path)
    assert evaluation.load_frozen_config(frozen) == RetrievalConfig()
    evaluation.validate_frozen_artifacts(frozen, metadata=metadata, index=tmp_path,
                                         embedding_model="BAAI/bge-small-zh-v1.5")


def test_frozen_configuration_edit_fails_integrity_check(tmp_path):
    evaluation, frozen, _metadata = freeze_fixture(tmp_path)
    payload = json.loads(frozen.read_text())
    payload["configuration"]["dense_rrf_weight"] = 3
    frozen.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        evaluation.load_frozen_config(frozen)


@pytest.mark.parametrize("changed_file", ["documents_metadata.json", "index.faiss", "index.pkl"])
def test_frozen_artifact_drift_is_rejected(tmp_path, changed_file):
    evaluation, frozen, metadata = freeze_fixture(tmp_path)
    (tmp_path / changed_file).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        evaluation.validate_frozen_artifacts(frozen, metadata=metadata, index=tmp_path,
                                             embedding_model="BAAI/bge-small-zh-v1.5")


def test_frozen_model_drift_is_rejected(tmp_path):
    evaluation, frozen, metadata = freeze_fixture(tmp_path)
    with pytest.raises(ValueError, match="Embedding model differs"):
        evaluation.validate_frozen_artifacts(frozen, metadata=metadata, index=tmp_path,
                                             embedding_model="another-model")
