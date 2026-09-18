# 代表性实验结果

本目录只保留能够说明当前技术选择的结果，删除了重复的中间运行文件。

| 文件 | 用途 |
| --- | --- |
| `bm25_char_bigram.json` | 中文字符 unigram/bigram BM25 基线 |
| `adaptive_chunking_448_96.json` | 448/96 token 自适应分块结果 |
| `v2_dev_rrf_sweep.json` | 开发集 RRF 参数扫描结果 |
| `v2_dev_reranker_retrieval_final.json` | 页面重排与引用基线 |
| `v2_dev_after_structured_finance.json` | 加入结构化财务路由后的开发集结果 |
| `alignment_rebuilt_dev_20260918.json` | 从 10 份 PDF 重建 448/96 索引，使用共享 2:1 检索链路的开发集验证 |

这些文件是历史实验快照。比较结果时必须同时核对数据集、代码版本、模型、阈值和运行模式，不能把开发集指标表述为生产准确率。

2026-09-18 的对齐验证重新构建了 120 个 Chunk，保留旧索引未覆盖。80 道开发题中的 67 道可回答题用于评分，RRF@3 Hit=1.0000、Recall=0.9925、Precision=0.3781、MRR=0.9851、NDCG=0.9820。报告保存数据集、元数据和向量索引 SHA256。该实验仅覆盖 PDF 建库、召回与融合，未调用云端重排或生成，未运行测试集；不代表完整答案正确率，也没有重新证明分块参数最优。
