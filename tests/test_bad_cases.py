import json

import pytest

from myai_rag.bad_cases import BadCaseStore


def make_store(tmp_path, max_traces=2):
    return BadCaseStore(
        tmp_path / "traces.json",
        tmp_path / "cases.json",
        max_traces=max_traces,
    )


def test_only_negative_feedback_creates_pending_case(tmp_path):
    store = make_store(tmp_path)
    trace_id = store.record_trace({"question": "收入是多少？", "retrieval_debug": {}})
    sources = [
        {
            "content": "报告正文不应重复写进 Bad Case",
            "company": "示例公司",
            "source_file": "report.pdf",
            "page_number": 3,
        }
    ]

    assert store.create_from_feedback(
        question="收入是多少？",
        answer="100万元",
        sources=sources,
        feedback="useful",
        trace_id=trace_id,
    ) is None

    case = store.create_from_feedback(
        question="收入是多少？",
        answer="100万元",
        sources=sources,
        feedback="useless",
        trace_id=trace_id,
    )

    assert case["status"] == "pending_review"
    assert case["trace"]["trace_id"] == trace_id
    assert case["sources"] == [
        {"company": "示例公司", "source_file": "report.pdf", "page_number": 3}
    ]


def test_review_then_export_curated_case(tmp_path):
    store = make_store(tmp_path)
    case = store.create_from_feedback(
        question="利润是多少？",
        answer="错误答案",
        sources=[],
        feedback="useless",
    )
    reviewed = store.review_case(
        case["bad_case_id"],
        failure_stage="generation",
        disposition="evaluation_candidate",
        review_notes="证据正确，生成数字错误",
        expected_answer="正确答案",
        expected_pages=[{"source_file": "report.pdf", "page_number": 2}],
    )
    output = tmp_path / "curated.jsonl"

    assert reviewed["status"] == "reviewed"
    assert store.export_reviewed(output) == 1
    exported = json.loads(output.read_text(encoding="utf-8").strip())
    assert exported["failure_stage"] == "generation"
    assert exported["expected_answer"] == "正确答案"


def test_ignored_case_is_not_exported(tmp_path):
    store = make_store(tmp_path)
    case = store.create_from_feedback(
        question="测试",
        answer="测试",
        sources=[],
        feedback="useless",
    )
    store.review_case(
        case["bad_case_id"],
        failure_stage="unknown",
        disposition="ignore",
    )
    output = tmp_path / "curated.jsonl"

    assert store.export_reviewed(output) == 0
    assert output.read_text(encoding="utf-8") == ""


def test_invalid_attribution_is_rejected(tmp_path):
    store = make_store(tmp_path)
    case = store.create_from_feedback(
        question="测试",
        answer="测试",
        sources=[],
        feedback="useless",
    )
    with pytest.raises(ValueError):
        store.review_case(
            case["bad_case_id"],
            failure_stage="随便写的阶段",
            disposition="evaluation_candidate",
        )


def test_trace_retention_limit(tmp_path):
    store = make_store(tmp_path, max_traces=2)
    first = store.record_trace({"sequence": 1})
    store.record_trace({"sequence": 2})
    store.record_trace({"sequence": 3})

    assert store.get_trace(first) is None
    assert len(json.loads((tmp_path / "traces.json").read_text(encoding="utf-8"))) == 2
