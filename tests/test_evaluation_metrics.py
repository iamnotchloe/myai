import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from evaluate_api import evaluate_response, summarize
from metrics import (answer_digest, human_review_metrics, page_ranking_metrics,
                     performance_metrics, refusal_metrics)


def case(answerable=True):
    return {"query_id": "q1", "question": "What is revenue?", "answerable": answerable,
            "answer_checks": [["100"]], "gold_answer": "100",
            "gold_pages": [{"source_file": "a.pdf", "page_number": 1, "relevance_grade": 3}]
            if answerable else []}


def payload(answer="Revenue is 100.", **extra):
    return {"success": True, "answer": answer,
            "source_documents": [{"source_file": "a.pdf", "page_number": 1,
                                  "content": "Revenue is 100."}], **extra}


def review(answer="Revenue is 100."):
    return {"query_id": "q1", "answer_sha256": answer_digest(answer), "reviewer": "human",
            "review_complete": True, "semantic_answer_correct": True,
            "answer_relevance": 2, "completeness": 2,
            "claims": [{"claim": answer, "supported": True,
                        "evidence": [{"source_file": "a.pdf", "page_number": 1,
                                      "quote": "Revenue is 100."}]}]}


def test_page_metrics_deduplicate_chunks_and_ignore_zero_grade():
    metrics = page_ranking_metrics([("a", 1), ("a", 1), ("b", 2)],
                                   {("a", 1): 3, ("b", 2): 0, ("c", 3): 1}, 3)
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["map"] == 0.5
    assert metrics["mrr"] == 1.0
    assert 0 < metrics["ndcg"] < 1


def test_false_refusals_count_and_errors_are_not_correct_abstentions():
    rows = [{"answerable": a, "predicted_refusal": r, "error": e}
            for a, r, e in [(False, True, None), (True, True, None),
                            (False, False, None), (True, False, None), (False, None, "timeout")]]
    result = refusal_metrics(rows)
    assert result["confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["refusal_success_rate"] == pytest.approx(1 / 3)
    assert result["unclassified_queries"] == 1


def test_api_failure_never_counts_as_zero_source_success():
    row = evaluate_response(case(False), {"success": False, "answer": "无法回答"}, 2)
    result = summarize([row])
    assert result["zero_source_rate_on_unanswerable"] == 0
    assert result["refusal_accuracy"] == 0
    assert result["lexical_end_to_end_pass_rate"] == 0
    assert result["errors"] == 1


@pytest.mark.parametrize("llm_status", ["failed", "error", "timeout", "not_configured", "invalid_response"])
def test_provider_failure_is_not_a_successful_refusal_even_with_success_flag(llm_status):
    row = evaluate_response(case(False), {"success": True, "answer": "无法回答",
                                          "telemetry": {"llm_status": llm_status}}, 2)
    result = summarize([row])
    assert row["predicted_refusal"] is None
    assert result["errors"] == 1
    assert result["zero_source_rate_on_unanswerable"] == 0
    assert result["performance"]["timeout_rate"] == (1 if llm_status == "timeout" else 0)


def test_service_error_outcome_is_failed_without_status_field():
    row = evaluate_response(case(False), {"success": True, "answer": "无法回答",
                                          "telemetry": {"outcome": "service_error"}}, 1)
    assert row["error"]
    assert row["predicted_refusal"] is None


def test_retrieval_only_has_no_answer_or_refusal_scores():
    row = evaluate_response(case(), payload(), 0.2, retrieval_only=True)
    result = summarize([row], retrieval_only=True)
    assert result["answer_accuracy"] is None
    assert result["refusal"] is None
    assert result["human_review"] is None


def test_stage_metrics_and_lost_evidence_exclude_rule_routes():
    item = {"source_file": "a.pdf", "page_number": 1}
    rag = evaluate_response(case(), payload(retrieval_debug={
        "route": "rag", "dense": [item, item], "bm25": [item], "fused": [item], "reranked": []}), 1)
    rule = evaluate_response({**case(), "query_id": "q2"}, payload(retrieval_debug={
        "route": "structured_finance", "dense": [], "bm25": [], "fused": [], "reranked": []}), 1)
    stages = summarize([rag, rule])["stage_metrics"]
    assert stages["dense"]["evaluated_queries"] == 1
    assert stages["dense"]["at_k"]["1"]["recall"] == 1
    assert stages["reranked"]["at_k"]["1"]["recall"] == 0
    assert rag["stage_diagnostics"]["fused_to_reranked"]["lost_gold_pages"] == [("a.pdf", 1)]


def test_missing_human_review_is_not_invented_faithfulness():
    row = evaluate_response(case(), payload(), 1)
    scores = summarize([row])["human_review"]
    assert scores["faithfulness"] is None
    assert scores["semantic_answer_accuracy"] is None
    assert scores["coverage"] == 0


def test_human_review_has_evidence_and_answer_identity_validation():
    row = evaluate_response(case(), payload(), 1)
    scores = human_review_metrics([row], [review()])
    assert scores["faithfulness"] == scores["claim_support"] == 1
    assert scores["semantic_answer_accuracy"] == 1
    bad_quote = review()
    bad_quote["claims"][0]["evidence"][0]["quote"] = "Fabricated evidence"
    with pytest.raises(ValueError, match="Evidence quote"):
        human_review_metrics([row], [bad_quote])
    with pytest.raises(ValueError, match="sha256"):
        human_review_metrics([row], [review("Different answer")])


def test_unsupported_claims_reduce_faithfulness_without_fake_evidence():
    row = evaluate_response(case(), payload("Revenue is 100. Profit is 90."), 1)
    assessment = review(row["answer"])
    assessment["claims"] = [review()["claims"][0],
                             {"claim": "Profit is 90.", "supported": False, "evidence": []}]
    scores = human_review_metrics([row], [assessment])
    assert scores["faithfulness"] == scores["claim_support"] == 0.5


def test_missing_checks_do_not_pass_vacuously():
    row = evaluate_response({**case(), "answer_checks": []}, payload(), 1)
    assert row["lexical_field_pass"] is None
    assert row["lexical_end_to_end_pass"] is None


def test_latency_percentiles_and_usage_are_reported_without_assumed_prices():
    rows = [evaluate_response({**case(), "query_id": str(i)}, payload(telemetry={
        "stage_latency_ms": {"llm": i * 100},
        "llm_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}), i)
        for i in (1, 2, 3)]
    result = performance_metrics(rows)
    assert result["latency_seconds"]["p50"] == 2
    assert result["latency_seconds"]["p95"] == pytest.approx(2.9)
    assert result["llm_usage"]["total_tokens"] == 360
    assert result["observed_llm_cost_estimate"] is None
    assert performance_metrics(rows, 1, 2)["observed_llm_cost_estimate"] == pytest.approx(0.00042)


@pytest.mark.parametrize("outcome, expected", [("refused", True), ("insufficient_evidence", True),
                                              ("answered", False), ("clarification", None)])
def test_server_outcome_takes_priority_over_keyword_heuristic(outcome, expected):
    row = evaluate_response(case(), payload("No keyword", telemetry={"outcome": outcome}), 1)
    assert row["predicted_refusal"] is expected


def test_cli_passes_history_and_requests_full_debug(monkeypatch, tmp_path):
    import evaluate_api

    calls = []
    dataset = tmp_path / "dev.jsonl"
    history = [{"role": "user", "content": "The previous company"}]
    dataset.write_text(json.dumps({**case(), "history": history}) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload()

    def post(*args, **kwargs):
        assert kwargs["timeout"] == 390
        calls.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(evaluate_api.requests, "post", post)
    monkeypatch.setattr(sys, "argv", ["evaluate_api", "--dataset", str(dataset), "--output", str(output)])
    evaluate_api.main()
    assert calls[0]["history"] == history
    assert calls[0]["debug"] is True
    assert json.loads(output.read_text())["summary"]["human_review"]["faithfulness"] is None


def test_review_cli_replays_saved_answers_without_requests(monkeypatch, tmp_path):
    import review_answers

    row = evaluate_response(case(), payload(), 1)
    report = tmp_path / "run.json"
    report.write_text(json.dumps({"summary": {"mode": "end_to_end"}, "results": [row]}))
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(json.dumps(review()) + "\n")
    output = tmp_path / "scored.json"
    monkeypatch.setattr(sys, "argv", ["review_answers", "--report", str(report), "--reviews", str(reviews),
                                      "--output", str(output)])
    review_answers.main()
    assert json.loads(output.read_text())["summary"]["human_review"]["faithfulness"] == 1
