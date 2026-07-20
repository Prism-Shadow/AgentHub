# Changelog

Here, we record the addition and removal times of models, major functional updates, bug fixes, and release times of key versions. Each entry keeps one brief line; the full details of a change live in a file under [changelog/](changelog/).

- [2026-07-20] Release version 0.4.0. Content items now carry an opaque `fidelity` payload that absorbs the former `signature`/`phase` fields (breaking), and OpenAI-compatible clients use it to replay thinking through exactly the reasoning field the upstream produced. ([details](changelog/0.4.0/2026-07-20-reasoning-field-fidelity.md))

- [2026-07-17] Add the `agenthub-dev` skill that fixes the model-support development workflow, and the `changelog/` details directory. ([details](changelog/0.4.0/2026-07-17-agenthub-dev-skill.md))

- [2026-07-14] Raise `EmptyResponseError` when a model completes a response with thinking output only, since sending it back would fail with a 400 error. It and `ToolCallArgumentParseError` now inherit the new `AgentHubError` base class. ([details](changelog/0.4.0/2026-07-14-empty-response-error.md))

- [2026-06-10] Support Claude 5 models. ([details](changelog/0.4.0/2026-06-10-claude-5.md))

- [2026-06-01] Release version 0.3.3. Support OpenAI-compatible embedding input format. ([details](changelog/0.3.3/2026-06-01-openai-embedding.md))

- [2026-05-30] Release version 0.3.2.

- [2026-05-30] Support Claude 4.8 models and an OpenAI Chat Completions API-compatible client. ([details](changelog/0.3.2/2026-05-30-claude-4-8-openai-compatible.md))

- [2026-05-28] Add abort support and agent skills. ([details](changelog/0.3.2/2026-05-28-abort-support-and-skills.md))

- [2026-05-27] Support Gemini 3.5, Gemini Embedding 2, Claude 4.7, Kimi-K2.6, GLM-5.1, DeepSeek V4 and Qwen3.6 models. Qwen3 models are deprecated. ([details](changelog/0.3.2/2026-05-27-model-refresh.md))

- [2026-04-28] Release version 0.3.1.

- [2026-04-28] Support Gemini 3.1 Flash TTS and GPT-5.5 models. Add UModelVerse vendor. ([details](changelog/0.3.1/2026-04-28-gemini-tts-gpt-5-5-modelverse.md))

- [2026-04-22] Gemini 3.1 Flash Image (Nano Banana 2) model is supported. ([details](changelog/0.3.1/2026-04-22-nano-banana-2.md))

- [2026-04-02] Switch to automatic caching for Claude 4.6 (but not for bedrock yet). Add message timestamp and round index to the tracer tool. ([details](changelog/0.3.1/2026-04-02-claude-auto-caching-tracer.md))

- [2026-03-11] GPT-5.4 is supported. We now add `phase` labels to assistant messages, and preserve and send them to the server. GPT-5.2 is deprecated. ([details](changelog/0.3.0/2026-03-11-gpt-5-4-phase-labels.md))

- [2026-03-04] Claude 4.6 is supported. We switch to using the adaptive thinking and `effort` parameter instead of the thinking budget. Supports Gemini on Vertex AI. Add Kimi-K2.5 model. Claude 4.5 models are deprecated. ([details](changelog/0.3.0/2026-03-04-claude-4-6-adaptive-thinking.md))

- [2026-02-26] Supports Claude on Amazon Bedrock. Bedrock requires image base64 encoding, we convert images to base64 in the client. ([details](changelog/0.3.0/2026-02-26-claude-bedrock.md))

- [2026-02-15] Fix encrypted thinking message in Claude models. It needs to be preserved and sent to the server. ([details](changelog/0.3.0/2026-02-15-claude-encrypted-thinking-fix.md))

- [2026-02-15] Fix the calculation of token usage in from OpenRouter provider. ([details](changelog/0.3.0/2026-02-15-openrouter-usage-fix.md))

- [2026-02-13] Support GLM-5 model, GLM-4.7 is deprecated. ([details](changelog/0.3.0/2026-02-13-glm-5.md))

- [2026-01-21] Supports GPT-5.2 via the Responses API. Add Qwen3 models support. ([details](changelog/0.2.0/2026-01-21-gpt-5-2-qwen3.md))

- [2026-01-20] Support prompt caching for Claude 4.5 models. ([details](changelog/0.2.0/2026-01-20-claude-prompt-caching.md))

- [2026-01-19] Support Claude 4.5 and GLM-4.7 models. ([details](changelog/0.2.0/2026-01-19-claude-4-5-glm-4-7.md))

- [2026-01-16] Support Gemini 3 models. ([details](changelog/0.2.0/2026-01-16-gemini-3.md))
