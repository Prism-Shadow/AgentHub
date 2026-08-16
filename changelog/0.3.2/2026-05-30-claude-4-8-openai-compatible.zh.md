# 支持 Claude 4.8 与兼容 OpenAI Chat Completions 的客户端

- **Date:** 2026-05-30
- **Type:** feature
- **Scope:** `claude4_6`, `openai`, `auto_client`
- **PR:** [#138](https://github.com/Prism-Shadow/agenthub/pull/138)

[English](2026-05-30-claude-4-8-openai-compatible.md)

## 变更内容

- 支持 Claude 4.8 模型。
- 新增通用的 OpenAI Chat Completions API 兼容客户端，并通过显式的 `client_type` 进行路由 ([#138](https://github.com/Prism-Shadow/agenthub/pull/138))，因此任何 Chat Completions 风格的端点都无需专门的协议目录即可使用。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
