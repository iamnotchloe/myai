# 架构说明

## 离线知识库构建

`myai_rag.indexing` 读取 `data/documents/` 下的 PDF，使用 `PyPDFLoader` 保留页码元数据，再交给 `myai_rag.chunking` 分块。分块器先识别标题、正文句子和表格式行，然后按真实 tokenizer 的 token 数累积元素；加入下一个元素会超过上限时结束当前块，并用完整尾部元素生成重叠。

每个 Chunk 都保存公司、源文件、页码、token 数、字符数和切分策略。构建流程同时写出 FAISS 向量索引、与索引对应的 Chunk 元数据以及中文 BM25 语料。这些文件写入 `artifacts/faiss_index/`，属于可再生运行产物。

## 在线问答链路

用户提问后，`myai_rag.query_rewrite` 先判断问题能否独立检索。对于“那净利润呢？”这类追问，只读取最近的用户问题，提取并补齐公司、年份和财务意图；助手生成的历史答案不会进入检索 Query。若指代对象无法确认，系统先请求澄清，不执行无目标检索。

这一步是 RAG 检索前的 Query Rewriting，不是通用聊天记忆。改写后的独立 Query 才进入 `myai_rag.api` 的三条路由：

1. 知识边界：实时信息、报告年份外问题、提示注入、隐私或虚构请求直接拒答。
2. 结构化财务：高置信的公司、年份和财务指标使用 `Decimal` 确定性取数与计算。
3. RAG：其余问题进入混合检索、融合、重排、生成与校验链路。

普通 RAG 路径依次执行公司范围约束下的 Dense 与中文 BM25、加权 RRF、Chunk 唯一标识去重、Chunk 级 Reranker、不同页面选择、相关性阈值、完整父页面扩展、LLM 生成、数字校验与页码引用。

API 与离线检索评测调用同一个 `retrieval.py`；`retrieval_config.py` 默认 Dense/BM25/Fused 为 10/20/30，RRF k=5、权重 2:1。448/96 和这些检索数值是当前项目基线，不代表普遍最优。索引构建也保留不同页上的同文 Chunk，不因正文相同丢失来源。

这里严格采用 Small-to-Big 顺序：小 Chunk 用于精确召回和重排，只有最终命中的 Chunk 才扩展到完整 PDF 页交给生成模型。依据是语雀 RAG 文档“二、进阶策略 → 基于文档结构的切分 → 父子索引（Parent–Child Indexing）”：先检索小切片，命中后再把所属父切片作为上下文。项目不会在 RRF 后无条件加入目标公司的全部 Chunk，也不会只按正文跨文件去重。

`debug=true` 时会返回 `original_query`、`rewritten_query`、`query_rewrite_used`、置信度和改写原因，便于确认问题发生在 Query 理解还是后续检索环节。

响应 `telemetry` 区分 `answered`、`refused`、`clarification`、`insufficient_evidence`、`retrieval_only` 和 `service_error`，记录实际执行阶段的毫秒耗时、重排路径、生成尝试次数及服务商返回的 token 用量。缺少 API Key、超时或解析失败不是正确拒答，返回 `success=false` 并保留 trace。尚未执行的阶段没有耗时，不用 0 冒充测量。

## Case-by-case 质量闭环

项目不使用 FAQ 相似问法缓存。FAQ 会绕过当前知识库检索，可能返回过期答案、缺失引用，也无法告诉我们本次问题究竟坏在哪一环。现在每次回答都会生成 `trace_id`，链路记录包括 Query 改写、路由、Dense/BM25/RRF/Reranker 排名、知识库版本、模型与 TopK 配置以及耗时，但不会重复保存 PDF 正文。

用户点差评后只创建 `pending_review` 案例，不会自动污染评测集或提示词。人工按照“原文 → PDF 解析 → Chunk → Dense/BM25 → 融合 → Reranker → 生成 → 引用/拒答”的顺序逐例归因，并决定 `evaluation_candidate`、`few_shot_candidate` 或 `ignore`。只有审核后的非忽略案例才能导出；few-shot 候选还必须填写人工纠正答案，才能由 `feedback.py` 生成示例。

依据是语雀 RAG 文档 [1.3 评估与迭代](https://www.yuque.com/zhongxian-iiot9/mp8m88/zbta3mgurgvuq4sq#KivP1)：多维指标体系、错误案例归因、难例挖掘与增强、用户反馈埋点和主动学习。Dense + Sparse、粗召回 + 精排及低置信结果治理来自 [1.4 检索与排序优化](https://www.yuque.com/zhongxian-iiot9/mp8m88/zbta3mgurgvuq4sq#cJUFQ)。这里把文档方法落成“发现一例、定位一例、审核一例、回归一例”的数据闭环。

`DENSE_TOP_K=10` 与 `BM25_TOP_K=20` 只是当前开发基线，不是语雀规定值，也不是理论固定值。语雀只支持混合召回与多阶段排序的方法选择，具体 K 必须在开发集上扫描，并同时比较 Recall@K、MRR、Reranker 候选量、p95 延迟和费用；方案冻结后才能用测试集确认。

## 本地与云端边界

本地执行 PDF 解析、自适应分块、BGE Embedding、FAISS、BM25、RRF、结构化财务计算和数字校验。云端执行 BGE Reranker 与生成模型。Reranker 超时或未配置 API Key 时，系统会使用本地 BM25 重排；生成模型没有可用 Key 时无法返回最终自然语言答案。

## 运行数据

`artifacts/`、`runtime/`、`.cache/` 和 `.env` 都是本地运行数据，不提交到 Git。
