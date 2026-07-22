# Gemini 3.6 SDK Documentation

This directory documents the Gemini 3.6 protocol generation, which starts with
`gemini-3.6-flash` and `gemini-3.5-flash-lite` and applies to all future Gemini model
releases. Content is snapshotted from the official documentation
(https://ai.google.dev/gemini-api/docs/latest-model).

The request/response wire format is shared with the Gemini 3 generation (see
[../gemini3/](../gemini3/README.md)); this generation changes the parameter contract:

- **Sampling parameters are deprecated**: `temperature`, `top_p`, and `top_k` are ignored by
  the API for these models and will return an HTTP 400 error in future model generations.
- **Model turn prefill is disallowed**: API requests ending with a non-empty `model` role turn
  return an HTTP 400 error.
- Thinking is configured with the `thinking_level` enum (no `thinking_budget`).

## Documentation

- [latest-model.md](./docs/latest-model.md) - The API changes introduced by this generation
- [gemini-3.6-flash.md](./docs/gemini-3.6-flash.md) - Gemini 3.6 Flash model spec
- [gemini-3.5-flash-lite.md](./docs/gemini-3.5-flash-lite.md) - Gemini 3.5 Flash-Lite model spec
- [thinking.md](./docs/thinking.md) - Thinking levels across Gemini 3.x models

For the SDK usage guides (function calling, streaming, thought signatures, TTS, image
generation, embeddings), refer to [../gemini3/docs/](../gemini3/README.md); they apply
unchanged to this generation.
