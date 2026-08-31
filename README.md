# MyAI RAG

MyAI RAG 是一个面向企业金融报告的检索增强问答项目。系统从 PDF 报告构建本地知识库，使用 Dense 向量检索与中文 BM25 混合召回，通过 RRF 融合和页面级重排选择证据，最后调用 SiliconFlow 上的模型生成带页码来源的答案。

## 核心能力

- 元素级、token 感知的自适应分块，当前上限为 448 tokens、重叠上限为 96 tokens。
- BGE Embedding + FAISS 本地向量检索。
- 中文字符 unigram/bigram BM25 精确词检索。
- 加权 RRF、公司过滤、页面去重与 Small-to-Big 上下文扩展。
- BGE Reranker 云端精排，并提供本地 BM25 降级路径。
- 知识边界拒答、确定性财务数据路由、数字校验和页码引用。
- 120 题评测集，覆盖检索、排序、引用、拒答、路由和延迟指标。

## 系统结构

```mermaid
flowchart LR
    A[PDF reports] --> B[Adaptive chunking]
    B --> C[BGE embeddings]
    C --> D[FAISS index]
    B --> E[Chinese BM25 corpus]
    Q[Question] --> R[Boundary and route]
    R --> F[Dense + BM25]
    D --> F
    E --> F
    F --> G[Weighted RRF]
    G --> H[Page collapse and rerank]
    H --> I[Full-page context]
    I --> J[LLM generation and validation]
    J --> K[Answer with citations]
```

Embedding 与 FAISS 在本地运行；Reranker 和生成模型通过 SiliconFlow API 调用。没有 API Key 时，索引构建和离线检索评测仍可运行，但云端精排与答案生成不可用。

## 目录

```text
.
├── src/myai_rag/          # 应用源码
│   ├── api.py             # FastAPI 服务与完整问答链路
│   ├── chunking.py        # 自适应分块
│   ├── indexing.py        # PDF 解析与索引构建
│   ├── finance.py         # 确定性财务数据路由
│   ├── feedback.py        # 反馈与 few-shot 数据整理
│   ├── ui.py              # Streamlit 界面
│   ├── config.py          # 统一路径配置
│   └── cli.py             # 命令行入口
├── data/                  # 示例报告与结构化数据
├── evaluation/            # 数据集、脚本和代表性结果
├── tests/                 # 单元测试
├── docs/                  # 架构与评测说明
├── artifacts/             # 本地生成索引，不提交 Git
└── runtime/               # 反馈与缓存，不提交 Git
```

## 快速开始

建议使用 Python 3.10 或 3.11。

```bash
git clone https://github.com/iamnotchloe/myai.git
cd myai

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

cp .env.example .env
```

在 `.env` 中填写 `SILICONFLOW_API_KEY`。API Key 只放本地 `.env`，不要提交到 Git。

### 构建知识库

```bash
myai-build-index
```

索引会生成在 `artifacts/faiss_index/`。该目录是可再生运行产物，因此不进入版本控制。

### 启动后端与前端

```bash
myai-api
```

另开一个终端：

```bash
myai-ui
```

健康检查：<http://127.0.0.1:8001/>

API 文档：<http://127.0.0.1:8001/docs>

### 调用问答接口

```bash
curl -X POST http://127.0.0.1:8001/rag_query \
  -H 'Content-Type: application/json' \
  -d '{"question":"滨江消费品2021年营业收入是多少？"}'
```

调试检索链但不调用生成模型：

```bash
curl -X POST http://127.0.0.1:8001/rag_query \
  -H 'Content-Type: application/json' \
  -d '{"question":"滨江消费品2021年营业收入是多少？","debug":true,"retrieval_only":true}'
```

## 测试与评测

```bash
pytest

python evaluation/evaluate_retrieval.py \
  --dataset evaluation/datasets/dev_set_v2.jsonl \
  --tokenizer char-bigram \
  --show-failures

python evaluation/evaluate_api.py \
  --dataset evaluation/datasets/dev_set_v2.jsonl \
  --retrieval-only
```

检索评测关注 Hit@K、Recall@K、Precision@K、MRR、MAP 和 NDCG；端到端评测同时检查答案、引用、拒答、路由和延迟。

## 配置与文档

所有配置项均记录在 [.env.example](.env.example)。常用路径也可以通过 `MYAI_DOCUMENTS_DIR`、`MYAI_STRUCTURED_FINANCE_PATH`、`MYAI_INDEX_DIR`、`MYAI_RUNTIME_DIR` 和 `MYAI_CACHE_DIR` 覆盖。

- [架构说明](docs/architecture.md)
- [评测说明](docs/evaluation.md)
- [评测数据与结果](evaluation/README.md)

## 数据与安全

- 仓库内 PDF 为项目示例数据；替换为真实企业资料前，请确认授权和隐私要求。
- `.env`、模型缓存、生成索引、反馈数据和运行缓存均被 `.gitignore` 排除。
- 当前仓库尚未声明开源许可证；除非仓库所有者另行授权，不应默认获得再分发权利。
