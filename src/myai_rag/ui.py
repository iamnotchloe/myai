"""Streamlit chat with persistent, trace-linked feedback on every answer."""

import os

import requests
import streamlit as st

st.set_page_config(page_title="智能金融问答助手", page_icon="🤖", layout="wide")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
BACKEND_API_URL = f"{BACKEND_BASE_URL}/rag_query"
BACKEND_READ_TIMEOUT = max(
    float(os.getenv("BACKEND_READ_TIMEOUT", "390")),
    3 * 120 + 2 * 1.5 + float(os.getenv("RERANK_TIMEOUT_SECONDS", "10")) + 17,
)

st.title("🤖 智能金融问答助手")
st.markdown("根据已导入的企业报告回答问题，并提供可核对的原文引用。支持结合上文追问。")
with st.sidebar:
    st.header("系统信息")
    st.info("PDF → 自适应切分 → Dense + BM25 → RRF → Chunk 重排 → 父页面 → 回答与引用")
    st.caption("财务计算优先使用有来源的结构化数据；资料不足时拒答或请您补充问题。")
    st.code(BACKEND_API_URL)
    st.caption("负反馈先进入 Bad Case 待审核队列，不会直接改成标准答案或 few-shot。")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(message, index):
    with st.chat_message(message["role"]):
        if message.get("service_error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        if message.get("resolved_question"):
            st.caption(f"已结合上文理解为：{message['resolved_question']}")
        if message.get("sources"):
            with st.expander("查看引用来源"):
                for source in message["sources"]:
                    st.info(f"{source.get('company', '未知公司')}｜{source.get('source_file', '')}"
                            f"｜第 {source.get('page_number', '?')} 页")
                    st.text(source.get("content", ""))
        if not message.get("trace_id"):
            return
        if message.get("feedback_saved"):
            st.caption(message["feedback_saved"])
            return
        # Buttons must be rendered from history on every rerun, not only inside chat_input.
        good, bad = st.columns(2)
        selected = None
        if good.button("👍 回答有用", key=f"good_{index}_{message['trace_id']}"):
            selected = "useful"
        if bad.button("👎 回答无用", key=f"bad_{index}_{message['trace_id']}"):
            selected = "useless"
        if selected:
            try:
                result = requests.post(
                    f"{BACKEND_BASE_URL}/save_feedback",
                    json={
                        "question": message["question"], "answer": message["content"],
                        "sources": message.get("sources", []), "feedback": selected,
                        "trace_id": message["trace_id"],
                    }, timeout=15,
                )
                result.raise_for_status()
                payload = result.json()
                if payload.get("status") != "success":
                    raise ValueError("feedback not saved")
                queued = payload.get("queued_for_review", False)
                message["feedback_saved"] = (
                    "已进入 Bad Case 待审核队列。" if queued else "感谢反馈！"
                )
                st.rerun()
            except (requests.RequestException, ValueError):
                st.error("反馈保存失败，请重试；本次反馈尚未确认保存。")


for index, message in enumerate(st.session_state.messages):
    render_message(message, index)

if prompt := st.chat_input("请输入关于报告的问题…"):
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
    ][-8:]
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)
    with st.spinner("正在检索报告并生成回答…"):
        try:
            response = requests.post(
                BACKEND_API_URL, json={"question": prompt, "history": history},
                # Backend allows three 120-second attempts plus rerank/backoff.
                timeout=(10, BACKEND_READ_TIMEOUT),
            )
            response.raise_for_status()
            result = response.json()
            message = {
                "role": "assistant", "question": prompt,
                "content": result.get("answer") or "暂时没有得到答案，请重试。",
                "sources": result.get("source_documents", []),
                "trace_id": result.get("trace_id"),
                "resolved_question": result.get("resolved_question"),
                "service_error": result.get("success") is False,
            }
        except (requests.RequestException, ValueError):
            message = {"role": "assistant", "content": "连接服务失败，请稍后重试。"}
        st.session_state.messages.append(message)
    st.rerun()
