# OpenRouter 条目改用 OpenAI Responses 协议

- **Date:** 2026-08-19
- **Type:** refactor
- **Scope:** `registry`, `skills`
- **PR:** [#180](https://github.com/Prism-Shadow/agenthub/pull/180)

[English](2026-08-19-openrouter-responses-protocol.md)

## 变更内容

- 此前使用通用 `openai-chat` 客户端的 17 个 OpenRouter 注册表条目改为 `openai-responses`：包括五个 `anthropic/*` 模型、`google/gemini-3.5-flash`、`minimax/minimax-m3`、`nvidia/nemotron-3-ultra-550b-a55b:free`、四个 `openai/*` 模型、`qwen/qwen3.6-35b-a3b`、`stepfun/step-3.7-flash`、`tencent/hy3`、`x-ai/grok-4.5` 和 `xiaomi/mimo-v2.5`。
- 三个 SiliconFlow 条目仍保留 `openai-chat`；由模型专用客户端（`glm-5.3`、`kimi-k3`、`kimi-k2.6`、`deepseek-v4`、`openai-embedding`）服务的 OpenRouter 条目未作改动。
- `README.md` 与两份 `skills/*/reference/models.md` 现已写明该偏好：当网关同时提供多种协议时优先选用 `openai-responses`；SiliconFlow 仅提供 Chat Completions，因此在其上使用 `openai-chat`。
