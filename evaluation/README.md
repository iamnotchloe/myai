# MyAI 评测工具使用说明

## 1. 文件说明

- `golden_test_set.jsonl`：32 条标准题，包含 27 条可回答题和 5 条无答案题。
- `golden_dataset_v2.jsonl`：扩充后的 120 条完整评测集，100 条可回答题、20 条无答案/安全题。
- `dev_set_v2.jsonl`：80 条开发集，只用于调参数、分析失败案例。
- `test_set_v2.jsonl`：40 条冻结测试集，只在方案确定后做最终评测。
- `build_golden_v2.py`：可复现地生成并校验 V2 数据集。
- `dataset_v2_summary.json`：V2 的题型、难度、切分和来源统计。
- `evaluate_retrieval.py`：离线测 Dense、BM25、RRF 的页面级召回，不调用云端 API。
- `evaluate_api.py`：调用正在运行的 `/rag_query`，测答案、引用、拒答和延迟。
- `results/`：运行后保存评测结果。

V1 Golden Set 是第一版人工基线。V2 按语雀材料中的“召回、排序、生成、整体”四层评估方法扩充，覆盖单事实、多事实、公司治理与 ESG、单位换算、时间线、跨文档比较、计算、错误前提、实时问题和提示注入。正式纳入项目报告前仍应逐条人工复核；调参数只能使用开发集，不能反复查看冻结测试集后继续调参。

重新生成并校验 V2：

```bash
cd "/Users/chloeewu/Desktop/华为实习/myai"
conda activate myai
python evaluation/build_golden_v2.py
```

## 2. 运行离线召回评测

在终端执行：

```bash
cd "/Users/chloeewu/Desktop/华为实习/myai"
conda activate myai
python evaluation/evaluate_retrieval.py \
  --tokenizer current \
  --show-failures \
  --output evaluation/results/retrieval_current.json
```

测试中文字符 unigram + bigram BM25：

```bash
python evaluation/evaluate_retrieval.py \
  --tokenizer char-bigram \
  --show-failures \
  --output evaluation/results/retrieval_char_bigram.json
```

使用 V2 开发集调参：

```bash
python evaluation/evaluate_retrieval.py \
  --dataset evaluation/dev_set_v2.jsonl \
  --tokenizer char-bigram \
  --show-failures \
  --output evaluation/results/v2_dev_retrieval.json
```

只有当参数、阈值和 Prompt 全部确定后，才运行一次冻结测试集：

```bash
python evaluation/evaluate_retrieval.py \
  --dataset evaluation/test_set_v2.jsonl \
  --tokenizer char-bigram \
  --output evaluation/results/v2_test_final.json
```

只快速检查 BM25，不加载 Embedding：

```bash
python evaluation/evaluate_retrieval.py --skip-dense --tokenizer current
```

脚本输出：

- Hit@K：每题 TopK 是否至少命中一页标准页；
- Recall@K：TopK 找回的标准页比例；
- Precision@K：TopK 中标准页的比例；
- MRR@K：第一个标准页排名倒数的平均值。
- MAP@K：只在相关页出现的位置累计精度，再对问题取平均，兼顾找全与排序。
- NDCG@K：根据 `relevance_grade` 计算折损累计增益，相关页越靠前分数越高。

当前评测按“PDF 页面”计算。一个页面被切成多个 Chunk 时，先合并为一条页面结果，避免同一页面重复占据 TopK。

## 3. 运行端到端评测

先分别启动后端和前端。评测只要求后端运行：

```bash
cd "/Users/chloeewu/Desktop/华为实习/myai"
conda activate myai
python fastapi_app.py
```

打开第二个终端：

```bash
cd "/Users/chloeewu/Desktop/华为实习/myai"
conda activate myai
python evaluation/evaluate_api.py \
  --output evaluation/results/api_results.json
```

第一次先运行 3 题，避免直接消耗大量 API 额度：

```bash
python evaluation/evaluate_api.py \
  --dataset evaluation/dev_set_v2.jsonl \
  --limit 3
```

按类别或数据切分筛选：

```bash
python evaluation/evaluate_api.py \
  --dataset evaluation/golden_dataset_v2.jsonl \
  --split dev \
  --category "跨文档比较" \
  --output evaluation/results/v2_dev_comparison_api.json
```

只评测真实后端的召回、Reranker、知识边界和引用，不调用生成模型：

```bash
python evaluation/evaluate_api.py \
  --dataset evaluation/dev_set_v2.jsonl \
  --retrieval-only \
  --output evaluation/results/v2_dev_reranker_retrieval.json
```

后端请求也可以直接增加：

```json
{
  "question": "阳光传媒与医疗先锋的营业收入相差多少亿元？",
  "debug": true,
  "retrieval_only": true
}
```

此时会返回 Dense、BM25、加权RRF和最终页面排名，但不会调用生成模型。

端到端脚本测：

- Answer Accuracy：答案是否包含题目要求的所有关键事实；
- Citation Hit Rate：最终引用是否至少包含一页标准页；
- Citation Recall：多文档题需要的标准页找全了多少；
- Refusal Accuracy：无答案题是否正确拒答；
- End-to-End Correct Rate：答案与引用是否同时正确；
- Mean Latency：平均响应时间。

长答案的 Faithfulness 仍需人工标注或使用带固定 Rubric 的 LLM Judge；当前脚本的关键词检查不能替代事实一致性评估。

## 4. 正确的实验顺序

1. 冻结测试集，另建开发集。
2. 跑当前 MyAI，保存 Baseline。
3. 一次只改变一个变量。
4. 比较 Dense、BM25、RRF、Reranker 前后的指标。
5. 查看失败题，而不是只看平均分。
6. 同时报告精度、延迟、API 成本和拒答能力。

V2 中每条数据的关键字段：

- `gold_pages`：标准来源页，`relevance_grade` 为 1～3，当前精确答案页标为 3；
- `answer_checks`：端到端自动检查必须覆盖的答案要点；
- `gold_answer`、`key_points`：人工复核和 LLM Judge 的参考；
- `category`、`subcategory`、`difficulty`、`query_style`：用于分桶看失败原因；
- `split`：`dev` 可调参，`test` 必须冻结；
- `expected_behavior`：特别用于无答案、错误前提和提示注入题的判定。

推荐消融实验：

| 实验 | 改动 | 需要观察 |
|---|---|---|
| Baseline | 当前 Dense + BM25 拼接 + Rerank | 建立当前基准 |
| A1 | 仅 Dense | Dense 独立贡献 |
| A2 | 仅 BM25 | 精确关键词贡献 |
| A3 | Dense + BM25 + RRF | 排名融合贡献 |
| A4 | A3 + Reranker | 精排贡献 |
| A5 | A4 + 公司元数据过滤 | 防跨公司错配贡献 |
| A6 | Chunk 200/300/500/800 | 粒度影响 |
| A7 | TopK 5/10/20，TopN 1/3/5 | 召回、噪声和成本平衡 |
| A8 | 拒答阈值扫描 | 拒答 Precision/Recall 平衡 |

当前开发集选出的检索参数是 `RRF_K=5`、`Dense权重=3.0`、`BM25权重=1.0`。
这些参数来自开发集，不能再用冻结测试集反复调整。

## 5. 结构化财务题评测

核心财务数值题现在优先经过 `structured_finance.py`，数据来自
`structured_financial_data.json` 中人工核对的报告事实。该路由不调用生成模型，适用于：

- 单公司一个或多个核心财务指标；
- 两家公司同一指标的相等、大小和差额比较；
- 由营业收入和净利润计算净利率；
- 按问题要求在元、万元、亿元或美元对应单位间换算。

涉及原因、趋势、报告中未结构化的业务事实、年份不符或人民币与美元直接比较时，系统会回退到原来的
RAG 流程，不会用结构化数据强行回答。Embedding、Reranker 和生成模型均未更换。

运行结构化单元测试：

```bash
cd "/Users/chloeewu/Desktop/华为实习/myai"
conda activate myai
python -m unittest evaluation/test_structured_finance.py -v
```

测试覆盖数值、多指标、跨公司差额、衍生净利率、指定单位、来源页有效性，以及应回退 RAG 的边界情况。
在调试响应中，`retrieval_debug.route` 的含义为：

- `structured_finance`：命中确定性财务计算；
- `knowledge_boundary`：命中静态知识边界并拒答；
- `rag`：使用原 Dense + BM25 + Reranker + LLM 流程。

V2 开发集的结构化回归结果保存在
`evaluation/results/v2_dev_after_structured_finance.json`。80题中20题走结构化路由，整体只检索评测的
平均延迟从0.881秒降至0.544秒；引用命中率为1.000、标准来源页召回率为0.993。
