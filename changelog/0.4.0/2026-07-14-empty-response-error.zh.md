# 拒绝仅含思考内容的响应并引入 AgentHubError 基类

- **Date:** 2026-07-14
- **Type:** fix
- **Scope:** `errors`, `base_client`, `tests`
- **PR:** [#155](https://github.com/Prism-Shadow/agenthub/pull/155), [#157](https://github.com/Prism-Shadow/agenthub/pull/157)

[English](2026-07-14-empty-response-error.md)

## 变更内容

- 仅以思考输出结束的响应现在会在流式传输结束时立即抛出 `EmptyResponseError`，因为在下一轮回放仅含思考内容的 assistant 消息会触发 400 错误（[#157](https://github.com/Prism-Shadow/agenthub/pull/157)）。
- `EmptyResponseError` 与 `ToolCallArgumentParseError`（在流式工具调用参数格式错误时抛出，[#155](https://github.com/Prism-Shadow/agenthub/pull/155)）现在都继承新的 `AgentHubError` 基类，调用方由此可以在一处捕获所有由 AgentHub 抛出的错误。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
