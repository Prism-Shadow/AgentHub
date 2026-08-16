# 支持带 phase 标签的 GPT-5.4；弃用 GPT-5.2

- **Date:** 2026-03-11
- **Type:** feature
- **Scope:** `gpt5_4`, `types`, `auto_client`, `base_client`, `llmsdk_docs`
- **PR:** [#87](https://github.com/Prism-Shadow/agenthub/pull/87)

[English](2026-03-11-gpt-5-4-phase-labels.md)

## 变更内容

- 通过 Responses API 支持 GPT-5.4 ([#87](https://github.com/Prism-Shadow/agenthub/pull/87))。
- Assistant 消息现在会携带 `phase` 标签，这些标签会被保留，并在重放时回传给服务端。
- GPT-5.2 已弃用。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
