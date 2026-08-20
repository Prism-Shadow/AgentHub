# Playground model list follows the E2E model list

- **Date:** 2026-08-19
- **Type:** fix
- **Scope:** `integration`, `tests`
- **PR:** [#180](https://github.com/Prism-Shadow/agenthub/pull/180)
- **Issue:** [#178](https://github.com/Prism-Shadow/agenthub/issues/178)

[中文版](2026-08-19-playground-models.zh.md)

## What changed

- The playground model dropdown was rewritten to list the models the E2E suites exercise
  against each vendor's own API: `gpt-5.6-luna`, `text-embedding-3-large`,
  `gemini-3.7-flash`, `gemini-3.1-flash-image`, `gemini-3.1-flash-tts-preview`,
  `gemini-embedding-2`, `claude-sonnet-5`, `glm-5.3`, `kimi-k3`, `MiniMax-M3`, and
  `deepseek-v4-flash`. `gpt-5.5`, `gemini-3.5-flash`, `gemini-3.1-flash-image-preview`,
  `claude-opus-4-7`, `claude-sonnet-4-6`, `kimi-k2.6`, `glm-5.1`, and `deepseek-v4-pro`
  were dropped, and the "Custom model" option was kept.
- The default model moved from `gpt-5.5` to `gpt-5.6-luna`, both in the rendered dropdown
  and in the server-side fallback applied when a request carries no model.
- A dropdown option can now declare `data-client-type`, which `getSelectedClientType()`
  sends as `client_type`. `text-embedding-3-large` uses it to reach the
  `openai-embedding` client, whose model id does not route on its own. The custom-model
  client-type field keeps working unchanged.
- Both the Python and TypeScript playgrounds were changed together and their tests
  updated to match.
