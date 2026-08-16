# Automatic caching for Claude 4.6; tracer timestamps and round index

- **Date:** 2026-04-02
- **Type:** feature
- **Scope:** `claude4_6`, `integration/tracer`, `types`, `base_client`, `gpt5_4`
- **PR:** [#95](https://github.com/Prism-Shadow/agenthub/pull/95), [#96](https://github.com/Prism-Shadow/agenthub/pull/96)

## What changed

- Claude 4.6 switched to automatic prompt caching by moving `cache_control` from message content items to a top-level API parameter ([#95](https://github.com/Prism-Shadow/agenthub/pull/95)); Bedrock still uses per-message cache control.
- `UniMessage`/`UniEvent` gained `created_at` timestamps, and the tracer tracks message rounds ([#96](https://github.com/Prism-Shadow/agenthub/pull/96)).

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
