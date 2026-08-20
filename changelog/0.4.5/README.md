# 0.4.5

[中文版](README.zh.md)

- [2026-08-20] `AutoLLMClient` lists the model ids the configured endpoint serves, and the playground lists them from an API key and a base URL. ([details](2026-08-20-list-models.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] The Responses-protocol message transforms read as one if/else chain. ([details](2026-08-20-responses-input-chain.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-19] Streaming clients skip stream events injected from outside the protocol, and still raise on unknown events inside it. ([details](2026-08-19-foreign-stream-events.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
