# Fix encrypted thinking messages in Claude models

- **Date:** 2026-02-15
- **Type:** fix
- **Scope:** `claude4_5`, `base_client`, `types`, `tests`
- **PR:** [#74](https://github.com/Prism-Shadow/agenthub/pull/74)

## What changed

- Encrypted (redacted) thinking blocks from Claude must be preserved in history and sent back to the server unchanged; the client no longer drops them ([#74](https://github.com/Prism-Shadow/agenthub/pull/74)).

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
