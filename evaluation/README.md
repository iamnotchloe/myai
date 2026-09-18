# 评测数据与结果

本目录保存可复现的评测脚本、数据集和代表性实验结果。

```text
evaluation/
├── datasets/                 # Golden Set、dev/test 与摘要
├── results/                  # 代表性实验输出
├── build_golden_v2.py        # 生成并校验 V2 数据集
├── evaluate_query_rewrite.py # 多轮追问改写与澄清离线评测
├── evaluate_retrieval.py     # BM25、Dense、RRF 离线评测
└── evaluate_api.py           # API、引用、拒答、路由和延迟评测
```

## 多轮 Query 改写评测

先验证多轮 Query 补全逻辑（不调用模型 API）：

```bash
python evaluation/evaluate_query_rewrite.py
```

该评测覆盖实体、年份、意图继承，公司切换、跨公司比较、单位换算、无上下文澄清和助手答案污染。随后再评测改写 Query 的 Hit@K/MRR 提升，才能判断它是否真正改善了 RAG 检索。

## 离线检索评测

先运行 `myai-build-index`，然后执行：

```bash
python evaluation/evaluate_retrieval.py \
  --dataset evaluation/datasets/dev_set_v2.jsonl \
  --tokenizer char-bigram \
  --show-failures
```

该脚本不会调用生成模型，适合比较 Chunk、Embedding、BM25、TopK 与 RRF 参数。

若要复现 Chunk 上限与 Overlap 的受控扫描：

```bash
PYTHONPATH=src python evaluation/sweep_chunking.py
```

新开发扫描写入 `evaluation/results/chunk_parameter_sweep_dev.json` 和对应 `.frozen.json`，不会覆盖旧结果。开发扫描不运行任何测试题；参数、模型、数据和 PDF 哈希冻结后，另行只评选中方案与固定 500/80 基线：

```bash
python evaluation/sweep_chunking.py \
  --evaluate-frozen evaluation/results/chunk_parameter_sweep_dev.frozen.json
```

旧结果只代表旧代码和配置，不能用作新 2:1 检索链路的当前成绩。

## API 评测

后端启动后执行：

```bash
python evaluation/evaluate_api.py \
  --dataset evaluation/datasets/dev_set_v2.jsonl \
  --retrieval-only
```

移除 `--retrieval-only` 后会进行端到端生成评测，并产生相应的 API 调用费用。`retrieval_only` 本身仍可能调用收费的云端重排。

默认 API 评测始终发送 `debug=true` 与题目 `history`，报告各阶段页面指标、引用 Precision/Recall、拒答混淆矩阵与 F1、分路由及分阶段 p50/p95、错误率和超时率。token 费用需显式传入 `--prompt-cost-per-million` / `--completion-cost-per-million`，仅估算已返回 token 的生成费用，不含未知重试、重排或 Embedding 费用。

## 人工答案评审

```bash
# 生成会产生模型调用费用；保存后可重复离线评审，不必重新请求 API
python evaluation/evaluate_api.py --output runtime/dev_answers.json
python evaluation/review_answers.py --report runtime/dev_answers.json \
  --output runtime/pending_reviews.jsonl
# 人工填写审核人、完整性确认、逐声明支持性和原文证据后
python evaluation/review_answers.py --report runtime/dev_answers.json \
  --reviews runtime/completed_reviews.jsonl --output runtime/reviewed_answers.json
```

审核结果必须绑定 `query_id` 与答案 SHA256。代码检查引用原句确实存在于返回的证据页；是否蕴含答案由人判断。报告人工语义准确率、Relevance、Completeness、Faithfulness、Claim Support 及审核覆盖率；无人审核时不产出这些成绩。`answer_checks` 仅做词面回归，包含正确关键词也可能事实错误。

## 数据使用规则

- `dev_set_v2.jsonl` 用于调参和错误分析。
- `test_set_v2.jsonl` 在方案冻结后用于最终报告。
- 评测以页面为证据单位，避免同页多个 Chunk 重复计分。
- `results/` 中的历史结果只代表对应代码与配置，不应表述为生产准确率。

## Bad Case 回流

线上差评不会直接进入数据集。先用 `myai-bad-cases list` 查看待审核案例，再用 `myai-bad-cases review` 标注失败环节、处理方式、正确答案或证据页，最后运行 `myai-bad-cases export`。导出的 `runtime/curated_bad_cases.jsonl` 是候选数据，仍需按既有数据集 schema 合并并执行校验。

这一步实现语雀 RAG 文档 [1.3 评估与迭代](https://www.yuque.com/zhongxian-iiot9/mp8m88/zbta3mgurgvuq4sq#KivP1) 中的“错误案例归因 → 难例挖掘与增强 → 用户反馈与主动学习”，避免未经审核的用户反馈直接污染 Golden Set。
