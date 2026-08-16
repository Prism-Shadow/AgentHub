# Support GPT-5.4 with phase labels; deprecate GPT-5.2

- **Date:** 2026-03-11
- **Type:** feature
- **Scope:** `gpt5_4`, `types`, `auto_client`, `base_client`, `llmsdk_docs`
- **PR:** [#87](https://github.com/Prism-Shadow/agenthub/pull/87)

## What changed

- GPT-5.4 is supported via the Responses API ([#87](https://github.com/Prism-Shadow/agenthub/pull/87)).
- Assistant messages now carry `phase` labels, which are preserved and sent back to the server on replay.
- GPT-5.2 is deprecated.

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
