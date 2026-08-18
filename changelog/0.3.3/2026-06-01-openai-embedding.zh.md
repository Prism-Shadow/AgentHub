# 支持 OpenAI 兼容的嵌入输入格式

- **Date:** 2026-06-01
- **Type:** feature
- **Scope:** `openai_embedding`, `openai`, `auto_client`, `types`
- **PR:** [#145](https://github.com/Prism-Shadow/agenthub/pull/145), [#146](https://github.com/Prism-Shadow/agenthub/pull/146), [#148](https://github.com/Prism-Shadow/agenthub/pull/148)

[English](2026-06-01-openai-embedding.md)

## 变更内容

- 为 OpenAI 兼容端点添加了文本嵌入支持（[#145](https://github.com/Prism-Shadow/agenthub/pull/145)），允许空的嵌入输入（[#146](https://github.com/Prism-Shadow/agenthub/pull/146)），并将嵌入客户端拆分到独立的 `openai_embedding/` 目录下，对应 `openai-embedding-compatible` 客户端类型（[#148](https://github.com/Prism-Shadow/agenthub/pull/148)）。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
