# 0.4.5

[English](README.md)

- [2026-08-20] Playground 面板按分区归类，并默认以 debug 模式启动。([详情](2026-08-20-playground-panel.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] Client 支持传入默认 header，以对接要求特定 header 的 endpoint。([详情](2026-08-20-default-headers.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] `AutoLLMClient` 可以列出所配置的 endpoint 提供的模型 id（按所路由的 client 过滤），playground 会把它们列进模型下拉框。([详情](2026-08-20-list-models.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] Responses 协议的消息转换与 GPT client 保持一致，包括缓冲文本的 flush 时机。([详情](2026-08-20-responses-input-chain.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-20] 未知的流式输出默认静默跳过，`AGENTHUB_DEBUG` 开启时才抛出。([详情](2026-08-20-unknown-stream-events.zh.md), [#181](https://github.com/Prism-Shadow/agenthub/pull/181))
- [2026-08-19] OpenRouter 注册表条目由 `openai-chat` 客户端改为 `openai-responses`。([详情](2026-08-19-openrouter-responses-protocol.zh.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
- [2026-08-19] 实况 E2E 套件遇到限流响应时改为重试，而非直接判定整轮失败。([详情](2026-08-19-e2e-rate-limit-retry.zh.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
- [2026-08-19] 注册表新增 `claude-opus-5`，官方 GPT-5.6 条目改名为 `gpt-5.6-sol`。([详情](2026-08-19-opus-5-and-gpt-5.6-sol.zh.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
- [2026-08-19] Playground 模型下拉列表与 E2E 模型列表对齐，选项可自行声明 client type。([详情](2026-08-19-playground-models.zh.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
- [2026-08-19] GLM-5.3 正式上线：文档快照、注册表定价与 E2E 模型改为取自已上线的 API。([详情](2026-08-19-glm-5.3-ga.zh.md), [#180](https://github.com/Prism-Shadow/agenthub/pull/180))
