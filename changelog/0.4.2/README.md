# 0.4.2 (unreleased)

- [2026-08-14] GLM-5.3 support (docs-only, API pre-launch) and the GLM series clients merge into one unified client. ([details](2026-08-14-glm-5.3.md))
- [2026-08-14] The Gemini 3 and 3.6/3.7 clients merge into one unified client that rejects `temperature` for the whole family. ([details](2026-08-14-unify-gemini-clients.md))
- [2026-08-13] Gemini 3.7 generation support: `gemini-3.7-flash` on the shared 3.6-protocol client, with per-model thinking-level clamping. ([details](2026-08-13-gemini-3.7.md))
- [2026-08-03] Official MiniMax M3 direct Responses API and Token Plan Subscription Key support. ([details](2026-08-03-minimax-m3.md))
- [2026-07-24] Gemini 3 clients clamp thinking levels to what each model actually supports — fixes `gemini-3.1-pro` rejecting `ThinkingLevel.NONE` with "Thinking level MINIMAL is not supported". ([details](2026-07-24-gemini-thinking-level-clamp.md))
