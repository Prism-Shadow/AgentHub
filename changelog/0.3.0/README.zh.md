# 版本 0.3.0

[English](README.md)

发布于 2026-03-11。

- [2026-03-11] 支持 GPT-5.4。现在会为 assistant 消息添加 `phase` 标签，并将其保留、回传给服务端。GPT-5.2 已弃用。([详情](2026-03-11-gpt-5-4-phase-labels.zh.md), [#87](https://github.com/Prism-Shadow/agenthub/pull/87))

- [2026-03-04] 支持 Claude 4.6。改为使用自适应思考与 `effort` 参数，取代思考预算。支持 Vertex AI 上的 Gemini。新增 Kimi-K2.5 模型。Claude 4.5 系列模型已弃用。([详情](2026-03-04-claude-4-6-adaptive-thinking.zh.md), [#81](https://github.com/Prism-Shadow/agenthub/pull/81), [#82](https://github.com/Prism-Shadow/agenthub/pull/82))

- [2026-02-26] 支持 Amazon Bedrock 上的 Claude。Bedrock 要求图像使用 base64 编码，因此我们在客户端将图像转换为 base64。([详情](2026-02-26-claude-bedrock.zh.md), [#79](https://github.com/Prism-Shadow/agenthub/pull/79))

- [2026-02-15] 修复 Claude 模型中的加密思考消息。该消息需要被保留并回传给服务端。([详情](2026-02-15-claude-encrypted-thinking-fix.zh.md), [#74](https://github.com/Prism-Shadow/agenthub/pull/74))

- [2026-02-15] 修复来自 OpenRouter 提供方的 token 用量计算。([详情](2026-02-15-openrouter-usage-fix.zh.md), [#73](https://github.com/Prism-Shadow/agenthub/pull/73))

- [2026-02-13] 支持 GLM-5 模型，GLM-4.7 已弃用。([详情](2026-02-13-glm-5.zh.md), [#71](https://github.com/Prism-Shadow/agenthub/pull/71))
