# AutoLLMClient lists the model ids an endpoint serves

- **Date:** 2026-08-20
- **Type:** feature
- **Scope:** `auto_client`, `claude5`, `gemini3_7`, `integration`, `tests`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[中文版](2026-08-20-list-models.zh.md)

## What changed

- `list_models()` / `listModels()` joined `AutoLLMClient` in both languages, returning the model
  ids the configured endpoint serves in the order the endpoint returned them. The
  OpenAI-compatible clients (`openai_chat`, `openai_responses`, `gpt5_6`, `minimax_m3`,
  `deepseek_v4`, `glm5_3`, `kimi_k3`, `openai_embedding`) and the Anthropic Messages clients
  (`claude5`, `ant_messages`) read their SDK's `models.list()`; `gemini3_7` reads the Gemini model
  list. Paging belongs to the SDKs, so a result spans every page the endpoint serves.
- `gemini3_7` returns the last path segment of each name, so both `models/gemini-3.7-flash` and
  `publishers/google/models/gemini-3.7-flash` become `gemini-3.7-flash` — the spelling
  `AutoLLMClient` takes as `model`.
- `UnsupportedOperationError` joined `errors` in both languages, reporting a capability the routed
  client does not have rather than a rejected parameter value, and `claude5` raises it on a
  `bedrock://` base URL because the Bedrock SDK client carries no models resource.
- `LLMClient` declares the method abstract, so every client implements it.
- The playground gained a **List models** button under its Base URL field, backed by a
  `POST /api/models` route that builds a client from the panel's model, API key, base URL and
  client type and answers with the ids. A listing the endpoint rejects comes back as a `400`
  carrying the provider's message, so a bad key or an unreachable base URL reads as an error
  rather than an empty list.
- Offline tests in both languages route a fake models endpoint through every client and pin the
  Gemini path stripping and the Bedrock rejection: `src_py/tests/test_list_models.py`,
  `src_ts/tests/list-models.test.ts`.

## Endpoint by protocol

| Protocol | Clients | Request |
| --- | --- | --- |
| OpenAI Chat Completions | `openai_chat`, `deepseek_v4`, `glm5_3`, `kimi_k3`, `openai_embedding` | `GET {base}/models` |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | `GET {base}/models` |
| Anthropic Messages | `claude5`, `ant_messages` | `GET {base}/v1/models` |
| Gemini generateContent | `gemini3_7` | `GET {base}/v1beta/models` |
