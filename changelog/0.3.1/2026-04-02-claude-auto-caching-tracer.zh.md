# Claude 4.6 自动缓存；tracer 时间戳与轮次索引

- **Date:** 2026-04-02
- **Type:** feature
- **Scope:** `claude4_6`, `integration/tracer`, `types`, `base_client`, `gpt5_4`
- **PR:** [#95](https://github.com/Prism-Shadow/agenthub/pull/95), [#96](https://github.com/Prism-Shadow/agenthub/pull/96)

[English](2026-04-02-claude-auto-caching-tracer.md)

## 变更内容

- Claude 4.6 将 `cache_control` 从消息内容项移至顶层 API 参数，从而切换为自动提示缓存 ([#95](https://github.com/Prism-Shadow/agenthub/pull/95))；Bedrock 仍使用逐消息的缓存控制。
- `UniMessage`/`UniEvent` 新增 `created_at` 时间戳，tracer 会跟踪消息轮次 ([#96](https://github.com/Prism-Shadow/agenthub/pull/96))。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
