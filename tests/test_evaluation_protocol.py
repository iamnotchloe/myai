"""Regression contracts for query-rewrite scoring and held-out data isolation."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import evaluate_query_rewrite as rewrite_eval
from evaluation import sweep_chunking as sweep


def test_slot_scoring_detects_entity_pollution_even_if_required_words_exist():
    case = {
        "id": "pollution", "question": "滨江消费品与澜赋科技2021年净利润是多少？", "history": [],
        "gold_slots": {"companies": ["澜赋科技有限公司"], "years": ["2021"], "topics": ["净利润"]},
        "expected_clarification": False, "forbidden_entities": ["滨江消费品"],
    }
    row = rewrite_eval.evaluate_case(case)
    assert not row["checks"]["companies_correct"]
    assert row["checks"]["years_correct"]
    assert not row["checks"]["forbidden_entities_absent"]
    assert not row["passed"]


def test_adversarial_development_rewrite_dataset_is_labeled_and_offline():
    path = rewrite_eval.PROJECT_ROOT / "evaluation/datasets/multiturn_query_rewrite.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(cases) >= 20
    assert all(case["split"] == "dev" for case in cases)
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(set(case["gold_slots"]) == {"companies", "years", "topics"} for case in cases)
    report = rewrite_eval.offline_report(cases)
    assert report["failures"] == []
    assert all(metric["labeled_cases"] == len(cases) for key, metric in report["slot_metrics"].items()
               if key in {"companies_correct", "years_correct", "topics_correct", "clarification_correct"})


def test_paired_api_scores_lift_and_reports_missing_stage_coverage():
    gold = {"source_file": "report.pdf", "page_number": 1}
    cases = [{"id": "followup", "question": "那净利润呢？",
              "history": [{"role": "user", "content": "澜赋科技2021年营业收入是多少？"}],
              "expected_clarification": False, "gold_pages": [gold]}]
    calls = []
    payloads = [
        {"retrieval_debug": {"route": "query_clarification", "dense": []}, "source_documents": []},
        {"retrieval_debug": {"route": "structured_finance", "dense": []}, "source_documents": [gold]},
    ]

    def post(url, json, timeout):
        calls.append(json)
        payload = payloads.pop(0)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    report = rewrite_eval.paired_retrieval_report(cases, "http://localhost:8001", 3, 10, post=post)
    assert all(call["history"] == [] and call["retrieval_only"] for call in calls)
    assert report["stages"]["final_sources"]["metrics"]["hit@3"]["lift"] == 1
    assert report["stages"]["dense"]["coverage"] == 0
    assert report["stages"]["dense"]["metrics"]["mrr@3"]["lift"] is None


def test_page_deduplication_and_reciprocal_rank():
    payload = {"retrieval_debug": {"route": "rag", "reranked": [
        {"source_file": "a.pdf", "page_number": 1, "rank": 2},
        {"source_file": "a.pdf", "page_number": 1, "rank": 1},
        {"source_file": "b.pdf", "page_number": 2, "rank": 3},
    ]}}
    pages = rewrite_eval.stage_pages(payload, "reranked")
    assert pages == [("a.pdf", 1), ("b.pdf", 2)]
    assert rewrite_eval.retrieval_metrics(pages, {("b.pdf", 2)}, 3) == {"hit": 1, "mrr": 0.5}


def test_paired_api_failure_is_not_a_successful_empty_retrieval():
    cases = [{"id": "failed", "question": "澜赋科技2021年净利润是多少？", "history": [],
              "expected_clarification": False,
              "gold_pages": [{"source_file": "report.pdf", "page_number": 1}]}]

    def post(*args, **kwargs):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"success": False})

    report = rewrite_eval.paired_retrieval_report(cases, "http://localhost", 3, 10, post=post)
    assert report["errors"] == 1
    assert report["stages"]["final_sources"]["paired_cases"] == 0
    assert report["stages"]["final_sources"]["metrics"]["hit@3"]["lift"] is None


def cases_for_protocol():
    return [
        {"query_id": "dev1", "question": "dev question", "split": "dev", "answerable": True},
        {"query_id": "test1", "question": "held-out question", "split": "test", "answerable": True},
    ]


def test_split_guard_rejects_shared_questions_and_ids():
    cases = cases_for_protocol()
    cases[1]["question"] = "dev question"
    with pytest.raises(ValueError, match="distinct"):
        sweep.validate_splits(cases)
    cases = cases_for_protocol()
    cases[1]["query_id"] = "dev1"
    with pytest.raises(ValueError, match="unique"):
        sweep.validate_splits(cases)


def test_frozen_manifest_rejects_data_drift_and_edits(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    cases = cases_for_protocol()
    dataset.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")
    (tmp_path / "a.pdf").write_bytes(b"original")
    inputs = sweep.input_manifest(dataset, tmp_path, cases)
    baseline, selected = [sweep.asdict(config) for config in sweep.configurations()[:2]]
    frozen = sweep.freeze_selection(selected, baseline, inputs, "model")
    assert len(sweep.validate_frozen_manifest(frozen, inputs, "model")) == 2
    (tmp_path / "a.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs differ"):
        sweep.validate_frozen_manifest(frozen, sweep.input_manifest(dataset, tmp_path, cases), "model")
    frozen["selected_config"]["overlap"] = 42
    with pytest.raises(ValueError, match="integrity"):
        sweep.validate_frozen_manifest(frozen, inputs, "model")


def test_dev_sweep_never_embeds_or_evaluates_test_and_frozen_run_never_selects(tmp_path, monkeypatch):
    dataset = tmp_path / "cases.jsonl"
    cases = cases_for_protocol()
    dataset.write_text("\n".join(json.dumps(case) for case in cases), encoding="utf-8")
    (tmp_path / "a.pdf").write_bytes(b"fixture-only")
    configs = [sweep.SweepConfig("fixed_chars_500_80", "fixed_chars", 500, 80),
               sweep.SweepConfig("adaptive_tokens_448_96", "adaptive_tokens", 448, 96)]
    monkeypatch.setattr(sweep, "configurations", lambda: configs)
    monkeypatch.setattr(sweep, "load_pages", lambda directory: [Document(page_content="fixture")])
    monkeypatch.setattr(sweep, "build_chunks", lambda *args: ([Document(page_content="fixture")], 0.01))
    encoded = []

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            encoded.extend(texts)
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda name: SimpleNamespace(encode=lambda text, **kwargs: list(text)))))
    evaluated = []

    def evaluate(config, chunks, vectors, query_vectors, given_cases, split):
        assert all(case["split"] == split for case in given_cases)
        evaluated.append((config.label, split))
        value = 0.9 if config.strategy == "adaptive_tokens" else 0.4
        metric = {name: value for name in ("ndcg", "recall", "mrr", "slot_coverage", "slot_complete")}
        return {"metrics": {"3": metric}, "per_query_at_3": [{"query_id": given_cases[0]["query_id"], **metric}]}

    monkeypatch.setattr(sweep, "evaluate_config", evaluate)
    dev_output = tmp_path / "dev.json"
    common = ["--dataset", str(dataset), "--documents", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["sweep", *common, "--output", str(dev_output)])
    sweep.main()
    report = json.loads(dev_output.read_text())
    assert all("test" not in result for result in report["results"])
    assert "held-out question" not in encoded
    assert all(split == "dev" for _, split in evaluated)
    encoded.clear()
    evaluated.clear()
    monkeypatch.setattr(sweep, "selection_key", lambda result: pytest.fail("Test must not reselect parameters"))
    monkeypatch.setattr(sys, "argv", ["sweep", *common, "--output", str(tmp_path / "test.json"),
                                     "--evaluate-frozen", str(dev_output.with_suffix(".frozen.json"))])
    sweep.main()
    assert "dev question" not in encoded
    assert evaluated == [(configs[0].label, "test"), (configs[1].label, "test")]


def test_historical_outputs_are_not_overwritten(tmp_path):
    output = tmp_path / "historical.json"
    output.write_text("historical", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sweep.write_new_json(output, {"new": True})
    assert output.read_text() == "historical"
