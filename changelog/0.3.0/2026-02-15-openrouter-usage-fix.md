# Fix token usage calculation from the OpenRouter provider

- **Date:** 2026-02-15
- **Type:** fix
- **Scope:** `utils`, `glm5`, `qwen3`, `integration/tracer`, `tests`
- **PR:** [#73](https://github.com/Prism-Shadow/agenthub/pull/73)

## What changed

- OpenRouter occasionally omits reasoning tokens from completion tokens; the usage metadata calculation compensates for it in both Python and TypeScript ([#73](https://github.com/Prism-Shadow/agenthub/pull/73)).

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
