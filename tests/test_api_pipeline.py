"""Isolated API checks: no model downloads, credentials, or paid calls."""

import importlib
import json
import sys
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient
from langchain_core.documents import Document


@pytest.fixture
def api(monkeypatch, tmp_path):
    import langchain_huggingface
    from langchain_community.vectorstores import FAISS
    from myai_rag import config

    index = tmp_path / "index"
    index.mkdir()
    metadata = {"company": "滨江消费品有限公司", "source_file": "report.pdf", "page": 0,
                "start_index": 0}
    doc = Document(page_content="滨江消费品有限公司调整了董事会治理结构。", metadata=metadata)
    (index / "documents_metadata.json").write_text(
        json.dumps([{"content": doc.page_content, "metadata": metadata}], ensure_ascii=False),
        encoding="utf-8",
    )
    for key, value in {
        "INDEX_DIR": index, "FEEDBACK_DB_PATH": tmp_path / "feedback.json",
        "FEW_SHOT_PATH": tmp_path / "few_shot.json", "QUERY_TRACE_PATH": tmp_path / "trace.json",
        "BAD_CASE_PATH": tmp_path / "cases.json", "CURATED_BAD_CASE_PATH": tmp_path / "curated.jsonl",
    }.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "")
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", lambda **kwargs: SimpleNamespace())
    db = SimpleNamespace(as_retriever=lambda **kwargs: None,
                         similarity_search_with_score=lambda query, k: [(doc, 0.1)])
    monkeypatch.setattr(FAISS, "load_local", lambda *args, **kwargs: db)
    monkeypatch.delitem(sys.modules, "myai_rag.api", raising=False)
    module = importlib.import_module("myai_rag.api")
    monkeypatch.setitem(module.pdf_page_texts, ("report.pdf", 0), "2021年报告。" + doc.page_content)
    monkeypatch.setattr(module, "is_retrieval_relevant", lambda *args: True)

    def no_network(*args, **kwargs):
        raise AssertionError("Tests must not access paid endpoints")

    monkeypatch.setattr(module.requests, "post", no_network)
    yield module
    sys.modules.pop("myai_rag.api", None)


def query(api, **kwargs):
    return TestClient(api.app).post("/rag_query", json={
        "question": "滨江消费品进行了哪些治理调整？", "debug": True, **kwargs,
    }).json()


def test_retrieval_only_has_stage_telemetry_and_stable_chunk_trace(api):
    result = query(api, retrieval_only=True)
    assert result["success"] is True
    assert result["telemetry"]["outcome"] == "retrieval_only"
    assert result["telemetry"]["llm_attempts"] == 0
    assert result["telemetry"]["llm_usage"] is None
    assert {"dense", "bm25", "fusion", "rerank", "parent_expansion", "total"} <= set(
        result["telemetry"]["stage_latency_ms"])
    assert result["retrieval_debug"]["dense"][0]["chunk_id"]
    trace = api.bad_case_store.get_trace(result["trace_id"])
    assert trace["telemetry"] == result["telemetry"]


def test_missing_api_key_is_service_failure_not_correct_refusal(api):
    result = query(api)
    assert result["success"] is False
    assert result["telemetry"]["llm_status"] == "not_configured"
    assert result["telemetry"]["outcome"] == "service_error"
    assert not result["source_documents"]


def test_boundary_does_not_call_models(api):
    result = query(api, question="今天滨江消费品股价是多少？")
    assert result["success"] is True
    assert result["retrieval_debug"]["route"] == "knowledge_boundary"
    assert result["telemetry"]["llm_attempts"] == 0
    assert result["telemetry"]["reranker_attempts"] == 0
    assert "dense" not in result["telemetry"]["stage_latency_ms"]


def test_retrieval_exception_is_traced_without_leaking_details(api, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("internal sensitive detail")
    monkeypatch.setattr(api, "dense_search", fail)
    result = query(api)
    assert result["success"] is False
    assert result["trace_id"]
    assert result["telemetry"]["error_type"] == "RuntimeError"
    assert "sensitive" not in json.dumps(result)


def test_generation_records_real_usage_after_retry(api, monkeypatch):
    monkeypatch.setattr(api, "SILICONFLOW_API_KEY", "test-only")
    monkeypatch.setattr(api.time, "sleep", lambda seconds: None)
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.Timeout()
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "choices": [{"message": {"content": "已调整治理结构。"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        })
    monkeypatch.setattr(api.requests, "post", post)
    telemetry = api.new_telemetry()
    assert api.generate_answer("问题", [], telemetry) == "已调整治理结构。"
    assert telemetry["llm_status"] == "success"
    assert telemetry["llm_attempts"] == 2
    assert telemetry["llm_usage"]["total_tokens"] == 110
    assert calls[-1]["json"]["temperature"] == 0.1
    assert calls[-1]["json"]["max_tokens"] == 1024


def test_non_retryable_credentials_error_is_not_retried(api, monkeypatch):
    monkeypatch.setattr(api, "SILICONFLOW_API_KEY", "test-only")
    def post(*args, **kwargs):
        response = requests.Response()
        response.status_code = 401
        raise requests.HTTPError(response=response)
    monkeypatch.setattr(api.requests, "post", post)
    telemetry = api.new_telemetry()
    api.generate_answer("问题", [], telemetry)
    assert telemetry["llm_attempts"] == 1
    assert telemetry["llm_status"] == "error"


def test_negative_feedback_requires_review_and_keeps_trace(api):
    client = TestClient(api.app)
    result = query(api, retrieval_only=True)
    response = client.post("/save_feedback", json={
        "question": result["question"], "answer": result["answer"],
        "sources": result["source_documents"], "feedback": "useless",
        "trace_id": result["trace_id"],
    })
    assert response.json()["queued_for_review"] is True
    cases = api.bad_case_store.list_cases(status="pending_review")
    assert cases[0]["trace"]["trace_id"] == result["trace_id"]
    assert cases[0]["trace"]["telemetry"]["outcome"] == "retrieval_only"
    assert api.bad_case_store.export_reviewed(api.CURATED_BAD_CASE_PATH) == 0
    assert not api.FEW_SHOT_PATH.exists()


def test_multiturn_trace_can_replay_original_history(api):
    history = [{"role": "user", "content": "滨江消费品2021年营业收入是多少？"}]
    result = query(api, question="那净利润呢？", history=history)
    assert result["resolved_question"] == "滨江消费品有限公司2021年净利润是多少？"
    assert result["retrieval_debug"]["route"] == "structured_finance"
    assert api.bad_case_store.get_trace(result["trace_id"])["history"] == history


def test_empty_retrieval_is_insufficient_evidence_not_service_failure(api, monkeypatch):
    monkeypatch.setattr(api, "dense_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(api, "bm25_search", lambda *args, **kwargs: [])
    result = query(api)
    assert result["success"] is True
    assert result["telemetry"]["outcome"] == "insufficient_evidence"
    assert result["telemetry"]["llm_attempts"] == 0
