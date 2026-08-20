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
- A listing is filtered to what the routed client can be used for. A protocol client
  (`openai-chat`, `openai-responses`, `ant-messages`, `openai-embedding`) is named explicitly and
  speaks for whatever the endpoint serves, so its listing comes back whole; a client deduced from a
  model id keeps only the ids that deduce back to it, so a gateway fronting many vendors narrows to
  that client's own models.
- `AutoLLMClient._client_class_for_model` split the routing conditions out of
  `_create_client_for_model`, which now looks up the class and constructs it. The conditions live
  in one place and answer both questions: which client serves a model id, and whether a listed id
  belongs to the client in hand.
- `gemini3_7` returns the last path segment of each name, so both `models/gemini-3.7-flash` and
  `publishers/google/models/gemini-3.7-flash` become `gemini-3.7-flash` — the spelling
  `AutoLLMClient` takes as `model`.
- `UnsupportedOperationError` joined `errors` in both languages, reporting a capability the routed
  client does not have rather than a rejected parameter value, and `claude5` raises it on a
  `bedrock://` base URL because the Bedrock SDK client carries no models resource.
- The playground lists into its own Model dropdown: a **List models** control on the Model label
  row calls `POST /api/models` and adds each returned id as an option, tagged with the client type
  the listing ran under. Selecting one keeps that client type on screen in the Client type field,
  rather than routing by something the panel does not show, and an edit there is written back to
  the option so it survives the next selection. A rejected listing shows the provider's message
  under the field instead.
- Offline tests in both languages route a fake models endpoint through every client and pin the
  filtering, the Gemini path stripping and the Bedrock rejection:
  `src_py/tests/test_list_models.py`, `src_ts/tests/list-models.test.ts`.

## Endpoint by protocol

| Protocol | Clients | Request |
| --- | --- | --- |
| OpenAI Chat Completions | `openai_chat`, `deepseek_v4`, `glm5_3`, `kimi_k3`, `openai_embedding` | `GET {base}/models` |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | `GET {base}/models` |
| Anthropic Messages | `claude5`, `ant_messages` | `GET {base}/v1/models` |
| Gemini generateContent | `gemini3_7` | `GET {base}/v1beta/models` |
