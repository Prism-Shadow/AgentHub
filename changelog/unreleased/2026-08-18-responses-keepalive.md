# Responses-protocol clients skip gateway keepalive heartbeats

- **Date:** 2026-08-18
- **Type:** fix
- **Scope:** `gpt5_6`, `openai_responses`, `minimax_m3`, `tests`
- **Issue:** [Prism-Shadow/penguin-harness#286](https://github.com/Prism-Shadow/penguin-harness/issues/286)

[中文版](2026-08-18-responses-keepalive.zh.md)

## What changed

- `gpt5_6`, `openai_responses`, and `minimax_m3` clients (Python and TypeScript) recognize
  the `keepalive` stream event as a known no-op and skip it silently. Gateways in front of
  Responses-compatible servers (one-api-style proxies, OpenRouter) inject
  `{"type": "keepalive", "sequence_number": N}` heartbeats into the SSE stream on long
  generations; these previously fell through to the unknown-event guard and killed the
  stream with `ValueError`/`Error`: `Unknown output: {"type":"keepalive","sequence_number":3}`.
- The unknown-event guard itself is unchanged: genuinely unknown event types, including
  provider error events, still raise.
- Offline regression tests in both languages stream keepalive heartbeats interleaved
  between normal Responses events through all three clients and pin the guard for unknown
  events — `src_py/tests/test_keepalive_events.py`, `src_ts/tests/keepalive-events.test.ts`.
