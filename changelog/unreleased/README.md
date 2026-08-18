# Unreleased

[中文版](README.zh.md)

- [2026-08-17] The Claude 4.6 client merges into the unified `claude5` client, which rejects `temperature` for the whole family. ([details](2026-08-17-unify-claude-clients.md), [#171](https://github.com/Prism-Shadow/agenthub/pull/171))
- [2026-08-17] Generic `openai_responses` and `ant_messages` protocol clients cover OpenAI, OpenRouter, DeepSeek, Z.AI, and MiniMax; the generic chat client renames to `openai_chat`. ([details](2026-08-17-protocol-clients.md), [#171](https://github.com/Prism-Shadow/agenthub/pull/171))
- [2026-08-17] GPT-5.6 support: the `gpt5_6` client serves GPT-5.4/5.5/5.6, with official and OpenRouter registry pricing. ([details](2026-08-17-gpt-5.6.md), [#171](https://github.com/Prism-Shadow/agenthub/pull/171))
- [2026-08-17] `UniConfig.fast_mode` maps to OpenAI `service_tier: "priority"` and Anthropic `speed: "fast"`. ([details](2026-08-17-fast-mode.md), [#171](https://github.com/Prism-Shadow/agenthub/pull/171))
- [2026-08-16] The dev skill specifies the client implementation standard: minimal replay probed against the live API, the reference-client shape, and delta-only streaming. ([details](2026-08-16-client-implementation-standard.md), [#170](https://github.com/Prism-Shadow/agenthub/pull/170))
- [2026-08-16] Changelog entries carry a metadata block with explicit PR links, and the dev skill gains a capture-serialization rule and a shared-test rule. ([details](2026-08-16-changelog-format-and-skill-rules.md), [#170](https://github.com/Prism-Shadow/agenthub/pull/170))
- [2026-08-14] GLM-5.3 support (docs-only, API pre-launch) and the GLM series clients merge into one unified client. ([details](2026-08-14-glm-5.3.md), [#169](https://github.com/Prism-Shadow/agenthub/pull/169))
- [2026-08-14] The Gemini 3 and 3.6/3.7 clients merge into one unified client that rejects `temperature` for the whole family. ([details](2026-08-14-unify-gemini-clients.md), [#168](https://github.com/Prism-Shadow/agenthub/pull/168))
- [2026-08-13] Gemini 3.7 generation support: `gemini-3.7-flash` on the shared 3.6-protocol client, with per-model thinking-level clamping. ([details](2026-08-13-gemini-3.7.md), [#168](https://github.com/Prism-Shadow/agenthub/pull/168))
- [2026-08-03] Official MiniMax M3 direct Responses API and Token Plan Subscription Key support. ([details](2026-08-03-minimax-m3.md), [#167](https://github.com/Prism-Shadow/agenthub/pull/167))
- [2026-07-24] Gemini 3 clients clamp thinking levels to what each model actually supports — fixes `gemini-3.1-pro` rejecting `ThinkingLevel.NONE` with "Thinking level MINIMAL is not supported". ([details](2026-07-24-gemini-thinking-level-clamp.md), [#166](https://github.com/Prism-Shadow/agenthub/pull/166))
