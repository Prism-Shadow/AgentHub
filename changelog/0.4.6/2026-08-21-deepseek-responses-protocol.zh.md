# DeepSeek 改用 OpenAI Responses 协议

- **Date:** 2026-08-21
- **Type:** refactor
- **Scope:** `deepseek_v4`, `registry`, `tests`
- **PR:** [#185](https://github.com/Prism-Shadow/agenthub/pull/185)

[English](2026-08-21-deepseek-responses-protocol.md)

## 变更内容

- `DeepSeekV4Client` 改为调用 `/responses` 而非 `/chat/completions`，解析
  `response.reasoning_text.delta`、`response.output_text.delta`、
  `response.function_call_arguments.delta` 与 `response.completed` 事件，回放思维链时重建成
  `content` 为 `reasoning_text` 的 `reasoning` 条目。
- SiliconFlow 的两个条目 `deepseek-ai/DeepSeek-V4-Flash` 与 `deepseek-ai/DeepSeek-V4-Pro` 改用通用
  的 `openai-chat` 客户端：SiliconFlow 只提供 Chat Completions。OpenRouter 与官方条目仍走
  `deepseek-v4` 客户端。
- 空响应、工具参数与未知事件三个单测套件里的 DeepSeek 用例改为 Responses 线格式，前两个套件同时
  新增了通用 `openai-responses` 客户端的用例。

- `README.md` 新增一张表，列出每个 `client_type` 在线上说的协议：`google-genai`、`ant-messages`、
  `openai-responses`、`openai-chat` 与 OpenAI Embeddings。

## 配置映射

| `UniConfig` 键 | 线上字段 |
| --- | --- |
| `thinking_level` | `reasoning.effort`，按 DeepSeek 实际生效的值预先映射：`none` / `low` / `high` / `high` / `high` / `max`。effort 取 `none` 才能关闭思考，Chat Completions 的 `thinking` 开关在该端点上被忽略 |
| `thinking_summary` | 不下发：该端点接受 `summary` 但从不生成 |
| `max_tokens` | `max_output_tokens` |
| `system_prompt` | `instructions` |
| `temperature` | 非 `1.0` 时抛 `UnsupportedParameterError` |
| `tool_choice` | 只接受 `auto` 与 `none`，其余抛 `UnsupportedParameterError` |
| `fast_mode` | 抛 `UnsupportedParameterError`：不支持 `service_tier` |
| `prompt_caching` | 只接受 `ENABLE`，缓存由服务端自动处理 |

用量字段为 `input_tokens`（含 `input_tokens_details.cached_tokens`）与 `output_tokens`
（含 `output_tokens_details.reasoning_tokens`）。
