# Playground 模型列表与 E2E 模型列表对齐

- **Date:** 2026-08-19
- **Type:** fix
- **Scope:** `integration`, `tests`
- **PR:** [#180](https://github.com/Prism-Shadow/agenthub/pull/180)
- **Issue:** [#178](https://github.com/Prism-Shadow/agenthub/issues/178)

[English](2026-08-19-playground-models.md)

## 变更内容

- Playground 的模型下拉列表改为列出 E2E 套件在各厂商自有 API 上实际测试的模型：`gpt-5.6-luna`、`text-embedding-3-large`、`gemini-3.7-flash`、`gemini-3.1-flash-image`、`gemini-3.1-flash-tts-preview`、`gemini-embedding-2`、`claude-sonnet-5`、`glm-5.3`、`kimi-k3`、`MiniMax-M3` 和 `deepseek-v4-flash`。移除了 `gpt-5.5`、`gemini-3.5-flash`、`gemini-3.1-flash-image-preview`、`claude-opus-4-7`、`claude-sonnet-4-6`、`kimi-k2.6`、`glm-5.1` 和 `deepseek-v4-pro`，并保留 "Custom model" 选项。
- 默认模型从 `gpt-5.5` 改为 `gpt-5.6-luna`，渲染出的下拉列表与请求未携带模型时服务端使用的兜底值均已同步。
- 下拉选项现在可以声明 `data-client-type`，由 `getSelectedClientType()` 作为 `client_type` 发送。`text-embedding-3-large` 借此指向 `openai-embedding` 客户端，因为其模型 id 本身无法完成路由。自定义模型的 client type 输入框行为保持不变。
- Python 与 TypeScript 两侧的 playground 一同修改，其测试也已相应更新。
