# streamlit_app.py

import json
import os

import requests
import streamlit as st

# --- 页面配置 ---
st.set_page_config(
    page_title="智能金融问答助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 应用常量 ---
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8001")
BACKEND_API_URL = f"{BACKEND_BASE_URL}/rag_query"

# --- 页面标题和描述 ---
st.title("🤖 智能金融问答助手")
st.markdown(
    "欢迎使用基于RAG（检索增强生成）技术的智能问答系统。您可以就金融相关问题进行提问。"
)
st.divider()

# --- 侧边栏 ---
with st.sidebar:
    st.header("💡 系统信息")
    st.info(
        "本项目结合了 **检索(Retrieval)**、**重排(Rerank)** 和 **生成(Generation)** "
        "技术，为您提供更精准的回答。"
    )

    st.subheader("技术栈:")
    st.markdown("""
    - **前端:** Streamlit
    - **后端:** FastAPI
    - **向量检索:** FAISS + BM25 + BGE-Embeddings 
    - **精排:** BGE-Reranker (via SiliconFlow)
    - **大模型:** Qwen-8B (via SiliconFlow)
    """)

    st.subheader("API 端点:")
    st.code(BACKEND_API_URL, language="bash")

# --- 初始化会话状态 (Session State) ---
# 会话状态用于在用户与应用交互时跨多次运行保留变量的值
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 显示历史聊天记录 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果是助手的回答，并且有引用来源，则显示它们
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("查看引用来源"):
                for i, source in enumerate(message["sources"]):
                    source_label = source.get("company", "未知公司")
                    if source.get("source_file"):
                        source_label += f"｜文件：{source['source_file']}"
                    if source.get("page_number"):
                        source_label += f"｜第 {source['page_number']} 页"
                    st.info(
                        f"**来源 {i + 1}｜公司：{source_label}**"
                    )
                    st.text(source["content"])

# --- 处理用户输入 ---
if prompt := st.chat_input("请输入您关于金融服务的问题..."):
    # 1. 在界面上显示用户的问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端API并显示助手的回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("正在思考中... 🤔")

        try:
            # 准备请求数据
            payload = {"question": prompt}
            headers = {"Content-Type": "application/json"}

            # 发送POST请求
            response = requests.post(
                BACKEND_API_URL, data=json.dumps(payload), headers=headers, timeout=120
            )

            # 检查响应状态码
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    answer = result.get("answer", "抱歉，未能生成回答。")
                    sources = result.get("source_documents", [])

                    # 更新聊天占位符为最终答案
                    message_placeholder.markdown(answer)

                    # 将助手的回答和引用来源存入会话状态
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                    # 显示引用来源
                    with st.expander("查看引用来源"):
                        if sources:
                            for i, source in enumerate(sources):
                                source_label = source.get("company", "未知公司")
                                if source.get("source_file"):
                                    source_label += f"｜文件：{source['source_file']}"
                                if source.get("page_number"):
                                    source_label += f"｜第 {source['page_number']} 页"
                                st.info(
                                    f"**来源 {i + 1}｜公司：{source_label}**"
                                )
                                st.text(source["content"])
                        else:
                            st.write("本次回答没有直接引用外部文档。")
                    # -------------------- 在此处添加反馈按钮 --------------------
                    # 用columns创建两个并排按钮
                    col1, col2 = st.columns(2)
                    with col1:
                        # 按钮key必须唯一，用消息数量区分
                        if st.button(
                            "👍 回答有用", key=f"good_{len(st.session_state.messages)}"
                        ):
                            feedback = {
                                "question": prompt,
                                "answer": answer,
                                "sources": sources,
                                "feedback": "useful",
                            }
                            requests.post(
                                f"{BACKEND_BASE_URL}/save_feedback",
                                json=feedback,
                                headers={"Content-Type": "application/json"},
                            )
                            st.success("感谢反馈！")
                    with col2:
                        if st.button(
                            "👎 回答无用", key=f"bad_{len(st.session_state.messages)}"
                        ):
                            feedback = {
                                "question": prompt,
                                "answer": answer,
                                "sources": sources,
                                "feedback": "useless",
                            }
                            requests.post(
                                f"{BACKEND_BASE_URL}/save_feedback",
                                json=feedback,
                                headers={"Content-Type": "application/json"},
                            )
                            st.info("我们会改进回答~")
                else:
                    error_message = result.get("error", "发生未知错误。")
                    message_placeholder.error(f"后端处理失败: {error_message}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"后端处理失败: {error_message}",
                        }
                    )
            else:
                error_text = response.text
                message_placeholder.error(
                    f"API请求失败，状态码: {response.status_code}\n错误信息: {error_text}"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"API请求失败: {response.status_code}",
                    }
                )

        except requests.exceptions.RequestException as e:
            message_placeholder.error(f"连接后端API时发生网络错误: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"网络错误: {e}"}
            )

# streamlit run .\streamlit_app.py
