# DeepSeek moves onto the OpenAI Responses protocol

- **Date:** 2026-08-21
- **Type:** refactor
- **Scope:** `deepseek_v4`, `registry`, `tests`

[中文版](2026-08-21-deepseek-responses-protocol.zh.md)

## What changed

- `DeepSeekV4Client` calls `/responses` instead of `/chat/completions`, reading the
  `response.reasoning_text.delta`, `response.output_text.delta`,
  `response.function_call_arguments.delta` and `response.completed` events, and rebuilding a
  replayed chain of thought as a `reasoning` item whose `content` is `reasoning_text`.
- The two SiliconFlow entries, `deepseek-ai/DeepSeek-V4-Flash` and
  `deepseek-ai/DeepSeek-V4-Pro`, moved to the generic `openai-chat` client: SiliconFlow serves
  Chat Completions only. The OpenRouter and official entries keep the `deepseek-v4` client.
- The DeepSeek cases in the empty-response, tool-call-argument and unknown-event unit suites
  moved to the Responses wire shape, and those first two suites gained a case for the generic
  `openai-responses` client alongside it.

- `README.md` gained a table of the wire protocol each `client_type` speaks: `google-genai`,
  `ant-messages`, `openai-responses`, `openai-chat`, and OpenAI Embeddings.

## Configuration behavior

| `UniConfig` key | On the wire |
| --- | --- |
| `thinking_level` | `reasoning.effort`, pre-mapped to what DeepSeek settles on: `none` / `low` / `high` / `high` / `high` / `max`. Effort `none` turns thinking off; the Chat Completions `thinking` toggle is ignored on this endpoint |
| `thinking_summary` | left out: the endpoint accepts `summary` but never generates one |
| `max_tokens` | `max_output_tokens` |
| `system_prompt` | `instructions` |
| `temperature` | `UnsupportedParameterError` unless it is `1.0` |
| `tool_choice` | `auto` and `none` only; anything else raises `UnsupportedParameterError` |
| `fast_mode` | `UnsupportedParameterError`: `service_tier` is not supported |
| `prompt_caching` | `ENABLE` only; caching is automatic |

Usage arrives as `input_tokens` with `input_tokens_details.cached_tokens`, and `output_tokens`
with `output_tokens_details.reasoning_tokens`.
