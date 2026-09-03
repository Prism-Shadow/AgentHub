# Control Qwen vLLM thinking through AgentHub levels

- **Date:** 2026-09-03
- **Type:** fix
- **Scope:** `qwen_vllm`, `auto_client`, `docs`

[中文版](2026-09-03-qwen-vllm-thinking-switch.zh.md)

## What changed

- Added the explicit `qwen-vllm` client type for Qwen models served through vLLM's OpenAI-compatible Chat Completions API.
- Mapped `thinking_level: none` to `chat_template_kwargs.enable_thinking: false` and every enabled AgentHub level to `true`.
- Left `chat_template_kwargs` absent when no level is selected and kept the generic `openai-chat` request unchanged.
