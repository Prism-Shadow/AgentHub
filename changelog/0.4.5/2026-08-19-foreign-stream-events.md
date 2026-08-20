# Streaming clients skip stream events injected from outside the protocol

- **Date:** 2026-08-19
- **Type:** fix
- **Scope:** `utils`, `ant_messages`, `claude5`, `openai_responses`, `tests`

[中文版](2026-08-19-foreign-stream-events.zh.md)

## What changed

- `is_foreign_no_op_event` / `isForeignNoOpEvent` joined `utils` in both languages, and the
  `claude5`, `ant_messages`, `openai_responses`, `gpt5_6`, and `minimax_m3` clients consult it
  before their unknown-event guard raises. An event is skipped only where all three hold: its
  type sits outside the namespace the protocol owns, its type names no error, and no field
  holds a non-empty object or array.
- A gateway heartbeat spelled the way a different protocol spells it no longer kills the
  stream. `{"type": "ping", "cost": "@"}` reaching an OpenAI Responses client previously
  raised `ValueError`/`Error`: `Unknown output: {"type":"ping","cost":"@"}`. Each family had
  learned only its own spelling in
  [the 0.4.3 heartbeat fix](../0.4.3/2026-08-18-gateway-heartbeats.md) — `ping` for Anthropic
  Messages, `keepalive` for OpenAI Responses — so either spelling crossing into the other
  protocol still raised.
- Unknown events inside the protocol's own namespace still raise, as do foreign events naming
  an error (`{"type": "gateway_error", "message": "upstream 502"}`) and foreign events carrying
  a payload (`{"type": "relay_frame", "data": {"text": "dropped"}}`).
- The offline regression tests in both languages stream foreign gateway events through the
  Responses and Messages clients, and pin each rejection the guard keeps:
  `src_py/tests/test_keepalive_events.py`, `src_ts/tests/keepalive-events.test.ts`.

## Namespaces the protocols own

| Protocol | Clients | Event type prefixes |
| --- | --- | --- |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | `response.` |
| Anthropic Messages | `claude5`, `ant_messages` | `message_`, `content_block_` |
