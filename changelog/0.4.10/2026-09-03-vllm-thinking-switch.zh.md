# 通过 AgentHub 等级控制 vLLM 思考开关

- **Date:** 2026-09-03
- **Type:** fix
- **Scope:** `openai_chat_vllm_adapter`, `auto_client`, `docs`
- **PR:** [#197](https://github.com/Prism-Shadow/agenthub/pull/197)

[English](2026-09-03-vllm-thinking-switch.md)

## 变更内容

- 为通过 vLLM OpenAI 兼容 Chat Completions API 提供的模型新增显式 `openai-chat-vllm-adapter` 客户端类型。
- 将 `thinking_level: none` 映射为 `chat_template_kwargs.enable_thinking: false`，并将 AgentHub 的其他思考等级映射为 `true`。
- 未选择思考等级时不添加 `chat_template_kwargs`，并保持通用 `openai-chat` 请求不变。
