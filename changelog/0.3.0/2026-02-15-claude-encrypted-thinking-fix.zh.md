# 修复 Claude 模型中的加密思考消息

- **Date:** 2026-02-15
- **Type:** fix
- **Scope:** `claude4_5`, `base_client`, `types`, `tests`
- **PR:** [#74](https://github.com/Prism-Shadow/agenthub/pull/74)

[English](2026-02-15-claude-encrypted-thinking-fix.md)

## 变更内容

- 来自 Claude 的加密（redacted）思考块必须在历史记录中保留，并原样回传给服务端；客户端不再将其丢弃 ([#74](https://github.com/Prism-Shadow/agenthub/pull/74))。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
