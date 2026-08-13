# Gemini 3.7 SDK Documentation

This directory documents the Gemini 3.7 protocol generation, which starts with
`gemini-3.7-flash`. Content is snapshotted from the official documentation
(https://ai.google.dev/gemini-api/docs/latest-model).

The request/response wire format and the parameter contract are identical to the
Gemini 3.6 generation (see [../gemini3_6/](../gemini3_6/README.md)): sampling
parameters stay deprecated, model turn prefill stays disallowed, and thinking is
configured with the `thinking_level` enum. The one model-visible change:

- **`minimal` thinking is gone**: `gemini-3.7-flash` supports only `low`, `medium`,
  and `high` and returns an HTTP 400 error for `thinking_level: "minimal"`, which
  the 3.6-generation flash models accept.

## Documentation

- [latest-model.md](./docs/latest-model.md) - The generation overview and pricing
- [gemini-3.7-flash.md](./docs/gemini-3.7-flash.md) - Gemini 3.7 Flash model spec
- [thinking.md](./docs/thinking.md) - Thinking levels across Gemini 3.x models

For the SDK usage guides (function calling, streaming, thought signatures, TTS, image
generation, embeddings), refer to [../gemini3/docs/](../gemini3/README.md); they apply
unchanged to this generation.
