# 0.4.5

[中文版](README.zh.md)

- [2026-08-20] The playground panel groups its fields and starts in debug mode. ([details](2026-08-20-playground-panel.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] Clients accept default headers for endpoints that demand their own. ([details](2026-08-20-default-headers.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] `AutoLLMClient` lists the model ids the configured endpoint serves, filtered to the routed client, and the playground lists them into its model dropdown. ([details](2026-08-20-list-models.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] The Responses-protocol message transforms match the GPT client, including when buffered text is flushed. ([details](2026-08-20-responses-input-chain.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] Unknown stream output is skipped unless `AGENTHUB_DEBUG` is set. ([details](2026-08-20-unknown-stream-events.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-19] OpenRouter registry entries move from the `openai-chat` client to `openai-responses`. ([details](2026-08-19-openrouter-responses-protocol.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
- [2026-08-19] The live E2E suites retry rate-limited responses instead of failing the run. ([details](2026-08-19-e2e-rate-limit-retry.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
- [2026-08-19] `claude-opus-5` joins the registry, and the official GPT-5.6 entry is named `gpt-5.6-sol`. ([details](2026-08-19-opus-5-and-gpt-5.6-sol.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
- [2026-08-19] The playground model dropdown follows the E2E model list, and an option can declare its client type. ([details](2026-08-19-playground-models.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
- [2026-08-19] GLM-5.3 launched: the docs snapshot, registry pricing, and E2E model move to the live API. ([details](2026-08-19-glm-5.3-ga.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
