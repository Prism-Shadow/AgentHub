# 未发布

[English](README.md)

- [2026-08-17] 通用 `openai_responses` 与 `ant_messages` 协议 client 覆盖 OpenAI、OpenRouter、DeepSeek、Z.AI 与 MiniMax；通用 chat client 重命名为 `openai_chat`。([详情](2026-08-17-protocol-clients.zh.md))
- [2026-08-17] GPT-5.6 支持：`gpt5_6` client 服务 GPT-5.4/5.5/5.6，registry 收录官方与 OpenRouter 价格。([详情](2026-08-17-gpt-5.6.zh.md))
- [2026-08-17] `UniConfig.fast_mode` 映射为 OpenAI `service_tier: "priority"` 与 Anthropic `speed: "fast"`。([详情](2026-08-17-fast-mode.zh.md))
- [2026-08-16] 开发 skill 规定了客户端实现标准：以实况探测确定的最小回放集合、参考客户端的写法、以及只按 delta 的流式处理。([详情](2026-08-16-client-implementation-standard.zh.md), [#170](https://github.com/Prism-Shadow/agenthub/pull/170))
- [2026-08-16] 变更日志条目带有包含显式 PR 链接的元数据块，开发 skill 新增了一条实况抓取序列化规则和一条共享测试规则。([详情](2026-08-16-changelog-format-and-skill-rules.zh.md), [#170](https://github.com/Prism-Shadow/agenthub/pull/170))
- [2026-08-14] GLM-5.3 支持（仅依据文档，API 尚未上线），且 GLM 系列客户端合并为一个统一客户端。([详情](2026-08-14-glm-5.3.zh.md), [#169](https://github.com/Prism-Shadow/agenthub/pull/169))
- [2026-08-14] Gemini 3 与 3.6/3.7 客户端合并为一个统一客户端，并对整个系列拒绝 `temperature`。([详情](2026-08-14-unify-gemini-clients.zh.md), [#168](https://github.com/Prism-Shadow/agenthub/pull/168))
- [2026-08-13] Gemini 3.7 代支持：`gemini-3.7-flash` 运行在共享的 3.6 协议客户端上，并按模型钳制思考档位。([详情](2026-08-13-gemini-3.7.zh.md), [#168](https://github.com/Prism-Shadow/agenthub/pull/168))
- [2026-08-03] 官方 MiniMax M3 直连 Responses API 与 Token Plan Subscription Key 支持。([详情](2026-08-03-minimax-m3.zh.md), [#167](https://github.com/Prism-Shadow/agenthub/pull/167))
- [2026-07-24] Gemini 3 客户端将思考档位钳制到各模型实际支持的范围 —— 修复了 `gemini-3.1-pro` 以 "Thinking level MINIMAL is not supported" 拒绝 `ThinkingLevel.NONE` 的问题。([详情](2026-07-24-gemini-thinking-level-clamp.zh.md), [#166](https://github.com/Prism-Shadow/agenthub/pull/166))
