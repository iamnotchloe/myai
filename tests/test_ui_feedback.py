"""Real Streamlit reruns must preserve answer-linked feedback and errors."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from streamlit.testing.v1 import AppTest


UI_PATH = Path(__file__).resolve().parents[1] / "src/myai_rag/ui.py"


def response(payload):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


def answer(trace_id="trace-one"):
    return {"success": True, "answer": "营业收入为8000万元。", "trace_id": trace_id,
            "source_documents": [{"company": "澜赋科技有限公司", "source_file": "report.pdf",
                                  "page_number": 1, "content": "营业收入8000万元。"}]}


def app():
    result = AppTest.from_file(str(UI_PATH), default_timeout=10).run()
    assert not result.exception
    return result


def ask(application, question="澜赋科技营业收入是多少？"):
    application.chat_input[0].set_value(question).run()
    assert not application.exception


def test_old_answer_feedback_survives_new_question_and_is_sent_only_once(monkeypatch):
    calls = []

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        if url.endswith("/save_feedback"):
            return response({"status": "success", "queued_for_review": True})
        question_count = sum(url.endswith("/rag_query") for url, _, _ in calls)
        return response(answer(f"trace-{question_count}"))

    monkeypatch.setattr(requests, "post", post)
    application = app()
    first_question = "澜赋科技营业收入是多少？"
    ask(application, first_question)
    ask(application, "那净利润呢？")
    application.run()
    application.button(key="bad_1_trace-1").click().run()
    assert not application.exception
    feedback_calls = [body for url, body, _ in calls if url.endswith("/save_feedback")]
    assert len(feedback_calls) == 1
    assert feedback_calls[0] == {
        "question": first_question, "answer": "营业收入为8000万元。", "feedback": "useless",
        "trace_id": "trace-1", "sources": answer()["source_documents"],
    }
    assert application.session_state["messages"][1]["feedback_saved"] == "已进入 Bad Case 待审核队列。"
    assert "feedback_saved" not in application.session_state["messages"][3]
    assert "bad_1_trace-1" not in [button.key for button in application.button]
    assert "bad_3_trace-2" in [button.key for button in application.button]
    application.run()
    assert len([call for call in calls if call[0].endswith("/save_feedback")]) == 1


@pytest.mark.parametrize("failure", ["http", "success_false", "invalid_json"])
def test_chat_backend_failure_is_visible_and_keeps_trace_when_available(monkeypatch, failure):
    def post(*args, **kwargs):
        if failure == "http":
            raise requests.HTTPError("service unavailable")
        if failure == "invalid_json":
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: (_ for _ in ()).throw(ValueError("invalid JSON")))
        return response({"success": False, "answer": "backend failed", "trace_id": "failed"})

    monkeypatch.setattr(requests, "post", post)
    application = app()
    ask(application)
    message = application.session_state["messages"][-1]
    if failure == "success_false":
        assert message["content"] == "backend failed"
        assert message["trace_id"] == "failed"
        assert message["service_error"] is True
        assert any("backend failed" in item.value for item in application.error)
        assert "bad_1_failed" in [button.key for button in application.button]
    else:
        assert "连接服务失败" in message["content"]
        assert not message.get("trace_id")
        assert len(application.button) == 0


@pytest.mark.parametrize("failure", ["http", "success_false"])
def test_feedback_failure_remains_retryable_without_false_confirmation(monkeypatch, failure):
    feedback_attempts = []

    def post(url, json, timeout):
        if url.endswith("/rag_query"):
            return response(answer())
        feedback_attempts.append(json)
        if len(feedback_attempts) == 1:
            if failure == "http":
                raise requests.Timeout("feedback timeout")
            return response({"success": False, "queued_for_review": False})
        return response({"status": "success", "queued_for_review": True})

    monkeypatch.setattr(requests, "post", post)
    application = app()
    ask(application)
    application.run()
    application.button(key="bad_1_trace-one").click().run()
    assert not application.exception
    assert "feedback_saved" not in application.session_state["messages"][1]
    assert any("尚未确认保存" in item.value for item in application.error)
    application.run()
    application.button(key="bad_1_trace-one").click().run()
    assert not application.exception
    assert application.session_state["messages"][1]["feedback_saved"] == "已进入 Bad Case 待审核队列。"
    assert len(feedback_attempts) == 2
    assert feedback_attempts[0] == feedback_attempts[1]


@pytest.mark.parametrize("rerank_timeout", [10, 60])
def test_chat_read_timeout_covers_backend_retry_and_rerank_budget(monkeypatch, rerank_timeout):
    monkeypatch.setenv("RERANK_TIMEOUT_SECONDS", str(rerank_timeout))
    timeouts = []

    def post(url, json, timeout):
        timeouts.append(timeout)
        return response(answer())

    monkeypatch.setattr(requests, "post", post)
    application = app()
    ask(application)
    assert timeouts[0][0] > 0
    assert timeouts[0][1] >= 3 * 120 + 2 * 1.5 + rerank_timeout
