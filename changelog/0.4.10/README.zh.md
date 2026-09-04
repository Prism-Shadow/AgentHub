# 0.4.10

[English](README.md)

- [2026-09-03] 将 AgentHub 思考等级映射到 vLLM 的 `enable_thinking` 请求开关。([详情](2026-09-03-vllm-thinking-switch.zh.md), [#197](https://github.com/Prism-Shadow/agenthub/pull/197))
- [2026-09-03] 按所服务的模型选择 vLLM 思考开关，不再一律发送 `enable_thinking`。([详情](2026-09-03-vllm-per-model-thinking.zh.md), [#198](https://github.com/Prism-Shadow/agenthub/pull/198))
- [2026-09-03] 支持 Gemini 3.8 Flash，共享的 Gemini 客户端按其重命名，按牌价登记，并让所有思考等级与思考摘要取值都下发到请求。([详情](2026-09-03-gemini-3.8-flash.zh.md), [#199](https://github.com/Prism-Shadow/agenthub/pull/199))
