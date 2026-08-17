# 统一的 Gemini 客户端与单一的采样参数契约

- **Date:** 2026-08-14
- **Type:** refactor
- **Scope:** `gemini3_7`, `gemini3`, `auto_client`, `registry`, `errors`
- **PR:** [#168](https://github.com/Prism-Shadow/agenthub/pull/168)
- **Breaking:** yes — `temperature` 现在在每个 Gemini 模型上都会抛出 `UnsupportedParameterError`，包括此前会将其透传的 `gemini-3.5-flash`、3.1 图像/TTS 模型和 2.5 系列

[English](2026-08-14-unify-gemini-clients.md)

## 变更内容

`gemini3` 与 `gemini3_7` 客户端目录在每种语言中合并为一个 `gemini3_7` 客户端，以其所服务的最新一代命名。除参数契约外，这两份实现此前已经逐行趋同，因此本次合并把剩余的差异归并到一个文件中：

- **Temperature 现在对整个系列都被拒绝。** 3.6 代弃用了采样参数，统一客户端将该契约应用于每个 Gemini 模型：在 `gemini-3.5-flash`、3.1 图像/TTS 模型以及 2.5 系列上设置 `temperature` 同样会抛出 `UnsupportedParameterError`，而旧客户端在这些模型上是将其透传的。
- 按模型划分的思考档位表合并为一张：2.5 系列仍完全丢弃 `thinking_level` 参数，图像模型保留 `minimal`/`high`，pro 代保留各自缩减后的集合，`gemini-3.7-*` 保留 `low`/`medium`/`high`。
- 随 3.7 支持引入的函数调用 id 往返现在覆盖整个系列：每个 `FunctionResponse` 都携带调用 id 和函数名，而在 id 之前产生的历史记录会降级为此前仅含函数名的形式。
- 路由收敛为一个分支：`gemini-3`/`gemini-embedding` 客户端类型（`gemini-3.7`、`gemini-3.6` 和 `gemini-3.5-flash-lite` 这些写法均包含它们）构造统一客户端，因此此前接受的每个客户端类型仍然可用。3.x 的文本、图像、TTS 和 embedding 模型的注册表条目指向 `gemini-3.7`，正如 Claude 4.7–5 的条目指向 `claude-5`。
- `llmsdk_docs/gemini3/` 保留为旧一代的线路协议参考；客户端目录合并，文档快照则不合并。

## 兼容性

向 `gemini-3.5-flash`、3.1 图像/TTS 模型或 2.5 系列传入 `temperature` 的代码现在会抛出 `UnsupportedParameterError`，而不再发送该值。请从 `UniConfig` 中移除该键；API 此前就已弃用采样参数并对这些模型忽略它，因此移除它不会改变输出。

此前能够路由的每个 `client_type` 写法仍然可以路由。
