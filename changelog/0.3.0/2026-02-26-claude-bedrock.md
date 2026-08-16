# Support Claude on Amazon Bedrock

- **Date:** 2026-02-26
- **Type:** feature
- **Scope:** `claude4_5`, `llmsdk_docs`, `tests`
- **PR:** [#79](https://github.com/Prism-Shadow/agenthub/pull/79)

## What changed

- Claude models are supported through Amazon Bedrock ([#79](https://github.com/Prism-Shadow/agenthub/pull/79)).
- Bedrock does not accept image URLs, so the client fetches images and converts them to base64 before sending.

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
