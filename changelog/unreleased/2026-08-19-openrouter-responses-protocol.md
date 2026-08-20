# OpenRouter entries move to the OpenAI Responses protocol

- **Date:** 2026-08-19
- **Type:** refactor
- **Scope:** `registry`, `skills`
- **PR:** [#180](https://github.com/Prism-Shadow/agenthub/pull/180)

[中文版](2026-08-19-openrouter-responses-protocol.zh.md)

## What changed

- The 17 OpenRouter registry entries that used the generic `openai-chat` client moved to
  `openai-responses`: the five `anthropic/*` models, `google/gemini-3.5-flash`,
  `minimax/minimax-m3`, `nvidia/nemotron-3-ultra-550b-a55b:free`, the four `openai/*`
  models, `qwen/qwen3.6-35b-a3b`, `stepfun/step-3.7-flash`, `tencent/hy3`,
  `x-ai/grok-4.5`, and `xiaomi/mimo-v2.5`.
- The three SiliconFlow entries stayed on `openai-chat`, and OpenRouter entries served by
  a model-specific client (`glm-5.3`, `kimi-k3`, `kimi-k2.6`, `deepseek-v4`,
  `openai-embedding`) were left alone.
- `README.md` and both `skills/*/reference/models.md` now state the preference: on a
  gateway serving more than one protocol, reach for `openai-responses`, and use
  `openai-chat` on SiliconFlow, which serves Chat Completions only.
