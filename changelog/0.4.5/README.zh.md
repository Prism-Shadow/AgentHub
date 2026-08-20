# 0.4.5

[English](README.md)

- [2026-08-20] Client 支持传入默认 header，以对接要求特定 header 的 endpoint。([详情](2026-08-20-default-headers.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] `AutoLLMClient` 可以列出所配置的 endpoint 提供的模型 id，playground 也能凭 API key 与 base URL 把它们列出来。([详情](2026-08-20-list-models.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] Responses 协议的消息转换与 GPT client 保持一致，包括缓冲文本的 flush 时机。([详情](2026-08-20-responses-input-chain.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] 未知的流式输出默认静默跳过，`AGENTHUB_DEBUG` 开启时才抛出。([详情](2026-08-20-unknown-stream-events.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
