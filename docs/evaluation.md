# 评测说明

## 数据划分

V2 数据集共有 120 题，其中 100 题可回答、20 题用于无答案与安全拒答。开发集 80 题，用于选择 Chunk 参数、TopK、RRF 权重和阈值；测试集 40 题，仅在方案冻结后用于最终报告，避免数据泄漏。

标准答案按 PDF 页面标注，因为系统最终选择并引用的是页面。同一页可能对应多个 Chunk，若直接按 Chunk 计分，会让重复证据歪曲 TopK 指标。

## 分层指标

| 层级 | 指标 | 说明 |
| --- | --- | --- |
| Query 理解 | Rewrite Accuracy | 追问是否被改写为标注的独立检索 Query |
| Query 理解 | Entity/Year/Intent Carryover | 公司、年份和财务意图是否正确继承，且没有引入旧话题 |
| Query 理解 | Clarification Accuracy | 指代无法确定时是否澄清；可确定时是否避免多余澄清 |
| Query→检索 | Follow-up Hit@K / MRR@K Lift | 同一批多轮题中，改写前后正确证据召回和首个正确结果排名的提升 |
| 粗召回 | Hit@K | TopK 是否至少包含一个标准证据页 |
| 粗召回 | Recall@K | 所有标准证据页被找回的比例 |
| 噪声 | Precision@K | TopK 中相关证据页的比例 |
| 排序 | MRR@K | 第一张正确证据页是否靠前 |
| 排序 | MAP@K | 多个正确页面是否完整且靠前 |
| 排序 | NDCG@K | 高相关性证据是否优先 |
| 生成 | Answer Accuracy | 关键事实是否正确覆盖 |
| 生成 | Faithfulness | 每个答案声明能否由上下文支持 |
| 引用 | Citation Hit/Recall/Precision | 引用是否命中、找全标准页面，且没有额外无关页 |
| 拒答 | Precision/Recall/F1 | 无答案问题是否正确拒答 |
| 系统 | p50/p95、超时率、成本 | 延迟稳定性与云 API 消耗 |

### 当前实现口径

- 20 条多轮开发回归题按公司、年份、指标分别判分，并报告继承与澄清指标。它们是项目自编标签，不冒充独立人工验证集。
- Dense/BM25/Fused/Reranked 分别按唯一 PDF 页评测；未走 RAG 的路由不混入对应阶段分母，报告有效覆盖率。
- API 的 `answer_checks` 是关键词检查，指标叫 `lexical_field_pass_rate`。`answer_accuracy` 仅保留为旧字段兼容别名，不是语义准确率。
- `review_answers.py` 对保存的端到端答案作逐声明人工评审，检验答案哈希、证据文件/页码及原句出处，输出人工 Accuracy、Faithfulness、Relevance、Completeness 和审核覆盖率。没有审核，结果为 `null`。
- 拒答以“拒答”为正类计算 Precision/Recall/F1；澄清和服务错误不算成功拒答，另报有效判定覆盖率及包含失败请求的拒答成功率。纯检索模式不声称已评答案或拒答质量。
- 延迟同时保留全请求与成功请求统计；没有返回 token 或没配置单价时，费用未知，不填 0。

标准答案包括 `gold_answer`（含公司/年份/数值/单位）、`gold_pages`（文件、从 1 起的页码、相关等级）、`answerable`、可选词面 `answer_checks`。应先人工核对原 PDF，再标注；字段存在或数据集校验通过不等于事实已经过独立审核。Bad Case 导出后仍是候选，不能自动改写冻结测试集。

## 如何平衡准确率与召回率

这里的“准确率”不能只用一个数字表示。检索阶段主要观察 `Precision@K`，表示返回的 K 个候选中有多少真正相关；`Recall@K` 表示标准证据中有多少被找回。增大 K 通常会提高 Recall，但同时带入更多无关候选，使 Precision、延迟和重排费用变差。

项目采用两阶段分工，而不是要求一个 TopK 同时做到最高 Precision 和最高 Recall：

1. **Dense/BM25/RRF 粗召回优先 Recall。** 正确证据如果没有进入候选集，后面的 Reranker 和 LLM 无法补救。
2. **Reranker 与页面选择恢复 Precision。** 在较宽的候选集中保留最相关的少量 Chunk，并去掉同页重复结果。
3. **答案充分性与生成校验控制错误回答。** “内容相关”不等于“足够回答”；证据缺少必要字段时应部分回答或拒答，不能为了提高回答率而编造。

参数选择采用“满足召回底线后选最小成本方案”，而不是把 Precision 和 Recall 简单平均：

1. 在开发集扫描 Dense TopK、BM25 TopK、Fused TopK、RRF 参数、Rerank TopN 和相关性阈值。
2. 由产品风险确定最低可接受的证据召回率 `R_min`；金融问答中漏掉证据的代价较高，应先确保粗召回达到该底线。`R_min` 是产品目标，不是语雀规定的固定数字。
3. 淘汰 Recall 低于 `R_min` 的配置。
4. 在剩余配置中，依次比较 Reranker 后的 Precision、MRR/NDCG、Citation Recall、Answer Accuracy、Faithfulness、无答案拒答 F1、p95 延迟和单题成本。
5. 选择满足质量约束的最小 K/TopN；如果一个方案的 Recall、Precision、延迟都不优于另一个方案，它不在 Pareto 前沿，应直接淘汰。
6. 只用开发集做上述选择；参数冻结后，测试集只运行一次用于最终确认。

可写成约束式决策：

```text
先满足：粗召回 Recall@K ≥ R_min
再优化：Rerank/Citation Precision、MRR、NDCG、Answer Accuracy、Faithfulness
同时约束：p95 延迟、超时率、API 成本、无答案误答率
```

### 用历史基线理解这个权衡

`evaluation/results/v2_dev_rrf_sweep.json` 的旧链路结果中，RRF 从 K=3 增加到 K=10 时：

| 配置 | Recall@K | Precision@K | NDCG@K |
| --- | ---: | ---: | ---: |
| RRF@3 | 97.01% | 36.82% | 94.16% |
| RRF@5 | 97.76% | 22.39% | 94.51% |
| RRF@10 | 100.00% | 11.49% | 95.27% |

这说明 K=10 在该旧基线中找全了证据，但候选噪声明显增加。因此不能把 10 个候选直接全部交给 LLM，而应交给 Reranker 缩减为 1～3 个高质量 Chunk。该结果产生于最近候选链路 Bug 修复之前，只用于解释权衡，不能作为当前版本参数已经最优的证明；修复后必须重新跑同一开发集。

### 端到端不能只看检索 Precision

- K 太小：页面看起来很精准，但可能漏掉正确证据，导致错误拒答。
- K 太大且不精排：召回较全，但无关页面进入上下文，可能降低 Faithfulness、增加幻觉和费用。
- Reranker TopN 太小：简单事实题更精准，但跨页题可能缺证据。
- Reranker TopN 太大：多证据题更完整，但简单题噪声增加。
- 相关性/充分性阈值太高：误答减少，但正确问题可能被拒答。
- 阈值太低：回答率提高，但无答案问题更容易被编造。

因此最终验收必须同时报告“可回答题的 Answer Accuracy + Citation Recall”和“无答案题的拒答 Precision/Recall/F1”，再结合 Faithfulness、延迟与成本。只提高回答率、只提高 Recall 或只提高 Precision，都不算完整优化。

## 问题定位

错误分析按证据链顺序进行：原文覆盖 → PDF 解析 → Chunk 完整性 → Dense/BM25 粗召回 → RRF → Reranker → 生成 → 引用与拒答。`debug=true` 和 `retrieval_only=true` 可以在不调用生成模型的情况下查看 Dense、BM25、融合和重排结果。

多轮问题先检查 `original_query` 与 `rewritten_query`：实体、年份或意图补错属于 Query 理解问题；改写正确但标准页未进入 TopK 属于召回问题；标准页已召回但排位低属于排序问题；证据正确而答案错误才属于生成问题。这样可以避免把所有失败都归因于大模型。

## 实验规则

1. 固定 PDF、数据集、模型和随机因素并保存 Baseline。
2. 一次只改变一个变量。
3. 分别评测 Dense、BM25、RRF、Reranker 和端到端结果。
4. 同时记录平均指标与失败题，不用单一平均分替代 Bad Case 分析。
5. 先在开发集选择方案，方案冻结后再运行测试集。
6. 检索稳定前优先使用 `retrieval_only`，减少不必要的生成费用。
7. 多轮 Query 评测必须同时包含正确继承、切换公司、跨公司比较、单位换算、无上下文指代和助手答案污染等 Bad Case。

## 从线上差评到回归集

1. `/rag_query` 为回答生成 `trace_id`，保存该次请求的 Query 改写、路由、各检索阶段排名、配置、知识库版本和耗时。
2. 用户点“回答无用”，`/save_feedback` 创建 `pending_review` Bad Case，并关联该 trace。
3. 人工先检查标准证据是否存在，再标注唯一的主要失败环节，补充正确答案/证据页和说明。
4. `ignore` 不进入数据；`evaluation_candidate` 进入回归评测候选；只有有人工纠正答案的 `few_shot_candidate` 才能进入 few-shot 候选。
5. 导出后加入开发集复现、修复并回归；方案冻结后用测试集确认，不能用测试集反复调参。

该流程对应语雀 RAG 文档 [1.3 评估与迭代](https://www.yuque.com/zhongxian-iiot9/mp8m88/zbta3mgurgvuq4sq#KivP1) 的错误案例归因、难例挖掘、用户反馈埋点与主动学习。它不是“差评自动训练”：自动收集只负责发现问题，人工审核负责阻断误点、恶意反馈和错误标签。
