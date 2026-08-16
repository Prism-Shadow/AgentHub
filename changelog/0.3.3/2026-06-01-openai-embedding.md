# Support OpenAI-compatible embedding input format

- **Date:** 2026-06-01
- **Type:** feature
- **Scope:** `openai_embedding`, `openai`, `auto_client`, `types`
- **PR:** [#145](https://github.com/Prism-Shadow/agenthub/pull/145), [#146](https://github.com/Prism-Shadow/agenthub/pull/146), [#148](https://github.com/Prism-Shadow/agenthub/pull/148)

[中文版](2026-06-01-openai-embedding.zh.md)

## What changed

- Added text embedding support for OpenAI-compatible endpoints ([#145](https://github.com/Prism-Shadow/agenthub/pull/145)), allowed empty embedding inputs ([#146](https://github.com/Prism-Shadow/agenthub/pull/146)), and split the embedding client into its own `openai_embedding/` folder with the `openai-embedding-compatible` client type ([#148](https://github.com/Prism-Shadow/agenthub/pull/148)).

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
