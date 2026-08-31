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

## API 评测

后端启动后执行：

```bash
python evaluation/evaluate_api.py \
  --dataset evaluation/datasets/dev_set_v2.jsonl \
  --retrieval-only
```

移除 `--retrieval-only` 后会进行端到端生成评测，并产生相应的 API 调用费用。

## 数据使用规则

- `dev_set_v2.jsonl` 用于调参和错误分析。
- `test_set_v2.jsonl` 在方案冻结后用于最终报告。
- 评测以页面为证据单位，避免同页多个 Chunk 重复计分。
- `results/` 中的历史结果只代表对应代码与配置，不应表述为生产准确率。
