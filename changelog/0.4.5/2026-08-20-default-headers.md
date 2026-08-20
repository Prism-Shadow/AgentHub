# Clients accept default headers for endpoints that demand their own

- **Date:** 2026-08-20
- **Type:** feature
- **Scope:** `auto_client`, `claude5`, `gemini3_7`, `integration`, `tests`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[中文版](2026-08-20-default-headers.zh.md)

## What changed

- `AutoLLMClient` and every client take a `default_headers` / `defaultHeaders` construction
  option, sent with every request to that endpoint. Gateways that reject a request whose client
  they do not recognize (an `unauthorized_client_error`, a missing `HTTP-Referer` or `X-Title`,
  an `Anthropic-Beta` the endpoint requires) can be reached by declaring what they ask for.
- The option is a connection property, not a request one: it is passed once at construction and
  never enters `UniConfig`, so it does not vary per request and does not reach the model.
- The playground gained an **Extra Headers** field under Base URL. It takes a JSON object, travels
  as `default_headers` in the panel config, and rebuilds the session client when it changes; text
  that is not a JSON object marks the field instead of being sent.
- Tests in both languages start a local HTTP server, list models through the OpenAI, Anthropic and
  Gemini clients, and assert the declared headers arrived — plus that a client declaring none adds
  none: `src_py/tests/test_default_headers.py`, `src_ts/tests/default-headers.test.ts`.

## Where each SDK carries them

| SDK | Clients | Carried as |
| --- | --- | --- |
| OpenAI | `openai_chat`, `openai_responses`, `gpt5_6`, `minimax_m3`, `deepseek_v4`, `glm5_3`, `kimi_k3`, `openai_embedding` | `default_headers` |
| Anthropic | `claude5` (direct and Bedrock), `ant_messages` | `default_headers` |
| Google GenAI | `gemini3_7` | `http_options.headers` |
