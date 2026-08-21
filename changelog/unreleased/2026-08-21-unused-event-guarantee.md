# An internal "unused" event never reaches a caller

- **Date:** 2026-08-21
- **Type:** refactor
- **Scope:** `base_client`, `types`, `skills`, `tests`
- **PR:** [#185](https://github.com/Prism-Shadow/agenthub/pull/185)

[中文版](2026-08-21-unused-event-guarantee.zh.md)

## What changed

- `streaming_response` / `streamingResponse` drop an event whose `event_type` is `unused`
  instead of forwarding it, and raise on one when `AGENTHUB_DEBUG` is set, so a client that
  skips its own filter fails in CI rather than shipping empty events.
- The `EventType` definition in `types.py` / `types.ts` states what the four kinds mean and
  that `unused` never leaves the client.
- The development skill records two client rules: `unused` stays inside the client, and the
  message transform keeps the order of the content items, flushing collected message text
  before appending an item that becomes its own input entry.
- `test_unknown_events.py` / `unknown-events.test.ts` run every client over the ignorable
  events its protocol carries with the debug guard on, and a deliberately leaky client covers
  both halves of the base-client behavior.
