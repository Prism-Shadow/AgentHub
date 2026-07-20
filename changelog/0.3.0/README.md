# Version 0.3.0

Released on 2026-03-11.

- [2026-03-11] GPT-5.4 is supported. We now add `phase` labels to assistant messages, and preserve and send them to the server. GPT-5.2 is deprecated. ([details](2026-03-11-gpt-5-4-phase-labels.md))

- [2026-03-04] Claude 4.6 is supported. We switch to using the adaptive thinking and `effort` parameter instead of the thinking budget. Supports Gemini on Vertex AI. Add Kimi-K2.5 model. Claude 4.5 models are deprecated. ([details](2026-03-04-claude-4-6-adaptive-thinking.md))

- [2026-02-26] Supports Claude on Amazon Bedrock. Bedrock requires image base64 encoding, we convert images to base64 in the client. ([details](2026-02-26-claude-bedrock.md))

- [2026-02-15] Fix encrypted thinking message in Claude models. It needs to be preserved and sent to the server. ([details](2026-02-15-claude-encrypted-thinking-fix.md))

- [2026-02-15] Fix the calculation of token usage in from OpenRouter provider. ([details](2026-02-15-openrouter-usage-fix.md))

- [2026-02-13] Support GLM-5 model, GLM-4.7 is deprecated. ([details](2026-02-13-glm-5.md))
