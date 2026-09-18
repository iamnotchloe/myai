# 已修复问题

## RRF 后候选被无条件扩大，且过早折叠为完整页

- 状态：已修复
- 影响环节：混合召回 → RRF → Reranker → 上下文扩展
- 原行为：先对全库 Dense/BM25 TopK 做公司过滤；RRF 后又加入目标公司的全部 Chunk；随后仅按正文去重，并在 Reranker 前折叠为完整页。
- 风险：正确公司 Chunk 可能在全局 TopK 之外而被过滤；加入全部 Chunk 会抵消粗召回并增加噪声、延迟和重排成本；只按正文会误删不同文件/页面的同文证据；完整页过早扩展会稀释命中 Chunk 的语义。
- 修复：直接计算目标公司范围内的 Dense/BM25 TopK；RRF 后只按 `source_file + page + start_index + content` 去重；Chunk 级重排并选择不同页面后，再扩展到完整父页面。
- 语雀依据：[RAG 文档](https://www.yuque.com/zhongxian-iiot9/mp8m88/zbta3mgurgvuq4sq)“二、进阶策略 → 基于文档结构的切分 → 父子索引（Parent–Child Indexing）”规定先检索小切片，命中后再使用父切片；[1.4 检索与排序优化](https://www.yuque.com/zhongxian-iiot9/mp8m88/zbta3mgurgvuq4sq#cJUFQ)支持混合召回后精排，但没有支持无条件加入整家公司全部 Chunk。
- 回归重点：单公司事实题、跨公司比较、相同正文出现在不同文件、正确页位于全库 Dense Top10 之外、长表格页面。
