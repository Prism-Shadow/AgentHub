# Every streaming client skips gateway heartbeat events

- **Date:** 2026-08-18
- **Type:** fix
- **Scope:** `openai_responses`, `openai_chat`, `ant_messages`, `gemini3_7`, `tests`
- **PR:** [#174](https://github.com/Prism-Shadow/agenthub/pull/174)
- **Issue:** [Prism-Shadow/penguin-harness#286](https://github.com/Prism-Shadow/penguin-harness/issues/286)

[中文版](2026-08-18-gateway-heartbeats.zh.md)

## What changed

- `gpt5_6`, `openai_responses`, and `minimax_m3` clients (Python and TypeScript) recognize
  the `keepalive` stream event as a known no-op and skip it silently. Gateways in front of
  Responses-compatible servers (one-api-style proxies, OpenRouter) inject
  `{"type": "keepalive", "sequence_number": N}` heartbeats into the SSE stream on long
  generations; these previously fell through to the unknown-event guard and killed the
  stream with `ValueError`/`Error`: `Unknown output: {"type":"keepalive","sequence_number":3}`.
- `openai_chat`, `deepseek_v4`, `glm5_3`, and `kimi_k3` clients read a chunk's `choices`
  only when the field is populated. A heartbeat is not a Chat Completions chunk, so
  `choices` arrives as `None`/`undefined` rather than an empty list, and the previous
  length check killed the stream with `TypeError: object of type 'NoneType' has no len()` /
  `TypeError: Cannot read properties of undefined (reading 'length')`.
- `claude5` and `ant_messages` clients treat the Messages API `ping` event as a known no-op
  alongside `text`, `thinking`, `signature`, and `input_json`. Both Anthropic SDKs drop
  `ping` at the SSE layer, so it reaches the transform only from a gateway that relabels it
  onto another event, where it previously raised `Unknown output: {"type":"ping"}`.
- `gemini3_7` clients mark a chunk carrying neither candidates nor usage as `unused`, and
  the stream loop skips it instead of emitting an empty `delta` event. A heartbeat arriving
  after the final chunk previously became the last event of the stream and failed the
  `Last event must carry usage_metadata` check in `base_client`.
- The unknown-event guards themselves are unchanged: genuinely unknown event types,
  including provider error events, still raise.
- Offline regression tests in both languages stream heartbeats through every streaming
  client — interleaved between normal events and after the final event — and pin the guards
  for unknown events: `src_py/tests/test_keepalive_events.py`,
  `src_ts/tests/keepalive-events.test.ts`.

## Heartbeat handling by protocol

| Protocol | Clients | Heartbeat shape | Handling |
| --- | --- | --- | --- |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | `{"type": "keepalive", "sequence_number": N}` | known no-op event type |
| OpenAI Chat Completions | `openai_chat`, `deepseek_v4`, `glm5_3`, `kimi_k3` | a chunk carrying no `choices` | no choices and no usage yields no event |
| Anthropic Messages | `claude5`, `ant_messages` | `{"type": "ping"}` | known no-op event type |
| Gemini generateContent | `gemini3_7` | a chunk carrying no candidates and no usage | `unused` event, skipped by the stream loop |
