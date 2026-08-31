# 代表性实验结果

本目录只保留能够说明当前技术选择的结果，删除了重复的中间运行文件。

| 文件 | 用途 |
| --- | --- |
| `bm25_char_bigram.json` | 中文字符 unigram/bigram BM25 基线 |
| `adaptive_chunking_448_96.json` | 448/96 token 自适应分块结果 |
| `v2_dev_rrf_sweep.json` | 开发集 RRF 参数扫描结果 |
| `v2_dev_reranker_retrieval_final.json` | 页面重排与引用基线 |
| `v2_dev_after_structured_finance.json` | 加入结构化财务路由后的开发集结果 |

这些文件是历史实验快照。比较结果时必须同时核对数据集、代码版本、模型、阈值和运行模式，不能把开发集指标表述为生产准确率。
