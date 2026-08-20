# Unknown stream output is skipped unless AGENTHUB_DEBUG is set

- **Date:** 2026-08-20
- **Type:** fix
- **Scope:** `utils`, `claude5`, `ant_messages`, `openai_responses`, `gemini3_7`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[中文版](2026-08-20-unknown-stream-events.zh.md)

## What changed

- The streaming clients no longer end a stream over output they do not recognize. `claude5`,
  `ant_messages`, `openai_responses`, `gpt5_6` and `minimax_m3` mark an unknown event `unused`, and
  `gemini3_7` skips an unknown content part. A gateway that injected
  `{"type": "ping", "cost": "@"}` into an OpenAI Responses stream previously raised
  `ValueError`/`Error`: `Unknown output: {"type":"ping","cost":"@"}` mid-generation.
- `is_debug_enabled()` / `isDebugEnabled()` joined `utils` in both languages, reading
  `AGENTHUB_DEBUG` per event. Set it to anything other than `0`, `false`, `no` or `off` and every
  unrecognized event or part raises again, with the same message as before.
- The per-protocol lists of known no-op events are unchanged, so a heartbeat a client already knew
  (`ping` on Anthropic Messages, `keepalive` on OpenAI Responses) is still recognized by name
  rather than by falling through. [The 0.4.3 fix](../0.4.3/2026-08-18-gateway-heartbeats.md) taught
  each family only its own spelling, which is how a relay's Anthropic-style ping reached a
  Responses client's guard.
- Offline tests in both languages stream heartbeats, cross-protocol spellings, in-protocol unknown
  events, gateway error frames and unknown Gemini parts through every client, and pin that each one
  raises once `AGENTHUB_DEBUG` is set: `src_py/tests/test_unknown_events.py`,
  `src_ts/tests/unknown-events.test.ts`.

## Handling by protocol

| Protocol | Clients | Unknown output | With `AGENTHUB_DEBUG` |
| --- | --- | --- | --- |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | event marked `unused` | raises |
| Anthropic Messages | `claude5`, `ant_messages` | event marked `unused` | raises |
| Gemini generateContent | `gemini3_7` | content part skipped | raises |
| OpenAI Chat Completions | `openai_chat`, `deepseek_v4`, `glm5_3`, `kimi_k3` | a chunk with no `choices` yields no event | unchanged |
