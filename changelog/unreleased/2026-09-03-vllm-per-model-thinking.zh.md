# 按所服务的模型映射 vLLM 思考开关

- **Date:** 2026-09-03
- **Type:** fix
- **Scope:** `vllm_openai_chat`, `docs`
- **PR:** [#PENDING](https://github.com/Prism-Shadow/agenthub/pull/PENDING)

[English](2026-09-03-vllm-per-model-thinking.md)

## 变更内容

- `vllm-openai-chat` 改为按所服务的模型 id（转小写后按子串匹配）选择 `chat_template_kwargs` 的形态，
  不再对所有模型一律发送 `enable_thinking`。
- Qwen3.8-Flash-Next、Qwen3.6-35B-A3B、Qwen3.5-0.8B 与 Qwen3.5-9B 仍使用 `enable_thinking` 布尔开关。
- Qwen3.8-27B 以 `enable_thinking: false` 关闭思考，并以 `reasoning_effort` 选择自适应模式；该字段只接受
  `low`/`medium`/`xhigh`，因此 `high` 与 `max` 收敛到 `xhigh`。
- DeepSeek-V4-Pro、DeepSeek-V4-Flash 与 DeepSeek-V4-Flash-Vision-Exp 以 `thinking: true` 搭配
  `reasoning_effort`；该字段只接受 `low`/`high`/`max`，因此 `medium` 与 `xhigh` 收敛到 `high`；`none`
  完全不发送 `chat_template_kwargs`，这正是这些模板判定为关闭的方式。
- 不在表内的模型回退到 `enable_thinking` 布尔开关。
