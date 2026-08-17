# Generic OpenAI Responses and Anthropic Messages protocol clients

- **Date:** 2026-08-17
- **Type:** feature
- **Scope:** `openai_responses`, `ant_messages`, `openai_chat`, `auto_client`, `tests`
- **Breaking:** yes — the `openai` client folder was renamed to `openai_chat` and its class `OpenaiClient` to `OpenaiChatClient`; deep imports of the old path stop working.

[中文版](2026-08-17-protocol-clients.zh.md)

## What changed

- `openai_responses` (`OpenaiResponsesClient`, client type `openai-responses`) was added: a
  generic OpenAI Responses-compatible client covering OpenAI, OpenRouter
  (`https://openrouter.ai/api/v1`), DeepSeek (`https://api.deepseek.com`), Z.AI
  (`https://api.z.ai/api/v1`), and MiniMax (`https://api.minimax.io/v1`).
- `ant_messages` (`AntMessagesClient`, client type `ant-messages`) was added: a generic
  Anthropic Messages-compatible client covering Anthropic, OpenRouter
  (`https://openrouter.ai/api`), DeepSeek (`https://api.deepseek.com/anthropic`), Z.AI
  (`https://api.z.ai/api/anthropic`), and MiniMax (`https://api.minimax.io/anthropic`).
- The generic Chat Completions client moved from `openai`/`OpenaiClient` to
  `openai_chat`/`OpenaiChatClient` with client type `openai-chat`; the bare `openai` client
  type keeps routing to it as an alias, and registry entries now name `openai-chat`.
- Both new clients pass `temperature` through, default `prompt_caching` to the provider's
  automatic cache (`ENABLE`; other values raise `UnsupportedParameterError`), and map
  `fast_mode` per [fast mode support](2026-08-17-fast-mode.md).
- e2e coverage: `deepseek-v4-flash`, `glm-5.2` (GLM Coding Plan base URLs), and `MiniMax-M3`
  run through all three protocol clients; `openai/gpt-5.6-luna` runs through all three via
  OpenRouter under `RUN_SLOW_TEST`. The test `Model` gained a `base_url` override plus
  `deepseek`/`zai`/`minimax` providers, and parametrize ids append the client type.

## Protocol implementation

- `openai_responses` streams thinking from both `response.reasoning_text.delta`
  (DeepSeek/Z.AI/MiniMax/OpenRouter dialect) and `response.reasoning_summary_text.delta`
  (OpenAI dialect), and finishes reasoning items on `response.output_item.done`, recording a
  fidelity of `channel` (`summary` when the item carried summaries) plus
  `encrypted_content`/`signature`/`format` when present. Replayed reasoning items always
  include the `summary` key (the OpenAI API rejects them without it) and rebuild the thinking
  text into `summary` or reasoning-text `content` according to the recorded channel;
  reasoning item ids are not replayed. Assistant-message `phase` is recorded and resent the
  way the `gpt5_6` client does, and text buffered inside a message is flushed before each
  top-level item to keep the wire order.
- `ant_messages` sends the credential through both `x-api-key` and `Authorization: Bearer`
  headers (`AsyncAnthropic(api_key=key, auth_token=key)`), which every covered server
  accepts. `ThinkingLevel.NONE` maps to `thinking: {"type": "disabled"}` explicitly (Z.AI
  thinks by default); other levels map to `adaptive` with `output_config.effort`, plus
  `display: "summarized"` under `thinking_summary`. Thinking blocks replay their `signature`
  only when one was recorded. Final usage merges `message_start` and `message_delta`, with
  the delta's input/cache counts winning when present (gateways report zeros at start), and
  `output_tokens_details.thinking_tokens` feeding `thoughts_tokens` when a server reports it.
