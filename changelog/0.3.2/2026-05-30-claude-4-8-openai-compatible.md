# Support Claude 4.8 and an OpenAI Chat Completions-compatible client

- **Date:** 2026-05-30
- **Type:** feature
- **Scope:** `claude4_6`, `openai`, `auto_client`
- **PR:** [#138](https://github.com/Prism-Shadow/agenthub/pull/138)

[中文版](2026-05-30-claude-4-8-openai-compatible.zh.md)

## What changed

- Claude 4.8 models are supported.
- Added a generic OpenAI Chat Completions API-compatible client with explicit `client_type` routing ([#138](https://github.com/Prism-Shadow/agenthub/pull/138)), so any Chat Completions-style endpoint can be used without a dedicated protocol folder.

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
