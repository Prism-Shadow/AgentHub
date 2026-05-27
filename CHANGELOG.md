# Changelog

Here, we record the addition and removal times of models, major functional updates, bug fixes, and release times of key versions.

- [2026-05-28] Add abort support and agent skills.

- [2026-05-27] Support Gemini 3.5, Gemini Embedding 2, Claude 4.7, Kimi-K2.6, GLM-5.1, DeepSeek V4 and Qwen3.6 models. Qwen3 models are deprecated.

- [2026-04-28] Release version 0.3.1.

- [2026-04-28] Support Gemini 3.1 Flash TTS and GPT-5.5 models. Add UModelVerse vendor.

- [2026-04-22] Gemini 3.1 Flash Image (Nano Banana 2) model is supported.

- [2026-04-02] Switch to automatic caching for Claude 4.6 (but not for bedrock yet). Add message timestamp and round index to the tracer tool.

- [2026-03-11] GPT-5.4 is supported. We now add `phase` labels to assistant messages, and preserve and send them to the server. GPT-5.2 is deprecated.

- [2026-03-04] Claude 4.6 is supported. We switch to using the adaptive thinking and `effort` parameter instead of the thinking budget. Supports Gemini on Vertex AI. Add Kimi-K2.5 model. Claude 4.5 models are deprecated.

- [2026-02-26] Supports Claude on Amazon Bedrock. Bedrock requires image base64 encoding, we convert images to base64 in the client.

- [2026-02-15] Fix encrypted thinking message in Claude models. It needs to be preserved and sent to the server.

- [2026-02-15] Fix the calculation of token usage in from OpenRouter provider.

- [2026-02-13] Support GLM-5 model, GLM-4.7 is deprecated.

- [2026-01-21] Supports GPT-5.2 via the Responses API. Add Qwen3 models support.

- [2026-01-20] Support prompt caching for Claude 4.5 models.

- [2026-01-19] Support Claude 4.5 and GLM-4.7 models.

- [2026-01-16] Support Gemini 3 models.
