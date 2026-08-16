# Add abort support and agent skills

- **Date:** 2026-05-28
- **Type:** feature
- **Scope:** `abort_signal`, `base_client`, `integration/playground`, `skills`
- **PR:** [#128](https://github.com/Prism-Shadow/agenthub/pull/128), [#130](https://github.com/Prism-Shadow/agenthub/pull/130), [#133](https://github.com/Prism-Shadow/agenthub/pull/133)

## What changed

- Streaming requests accept an abort signal in both Python and TypeScript ([#128](https://github.com/Prism-Shadow/agenthub/pull/128)), the abort waiter is reused during streaming ([#133](https://github.com/Prism-Shadow/agenthub/pull/133)), and the playground gained an abort control ([#130](https://github.com/Prism-Shadow/agenthub/pull/130)).
- Added the `agenthub-python` and `agenthub-typescript` SDK usage skills under `skills/` (#121, #129).

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
