# 架构说明

## 离线知识库构建

`myai_rag.indexing` 读取 `data/documents/` 下的 PDF，使用 `PyPDFLoader` 保留页码元数据，再交给 `myai_rag.chunking` 分块。分块器先识别标题、正文句子和表格式行，然后按真实 tokenizer 的 token 数累积元素；加入下一个元素会超过上限时结束当前块，并用完整尾部元素生成重叠。

每个 Chunk 都保存公司、源文件、页码、token 数、字符数和切分策略。构建流程同时写出 FAISS 向量索引、与索引对应的 Chunk 元数据以及中文 BM25 语料。这些文件写入 `artifacts/faiss_index/`，属于可再生运行产物。

## 在线问答链路

`myai_rag.api` 提供三条路由：

1. 知识边界：实时信息、报告年份外问题、提示注入、隐私或虚构请求直接拒答。
2. 结构化财务：高置信的公司、年份和财务指标使用 `Decimal` 确定性取数与计算。
3. RAG：其余问题进入混合检索、融合、重排、生成与校验链路。

普通 RAG 路径依次执行 Dense、中文 BM25、加权 RRF、公司过滤、页面折叠、Reranker、相关性阈值、完整页上下文扩展、LLM 生成、数字校验与页码引用。

## 本地与云端边界

本地执行 PDF 解析、自适应分块、BGE Embedding、FAISS、BM25、RRF、结构化财务计算和数字校验。云端执行 BGE Reranker 与生成模型。Reranker 超时或未配置 API Key 时，系统会使用本地 BM25 重排；生成模型没有可用 Key 时无法返回最终自然语言答案。

## 运行数据

`artifacts/`、`runtime/`、`.cache/` 和 `.env` 都是本地运行数据，不提交到 Git。
