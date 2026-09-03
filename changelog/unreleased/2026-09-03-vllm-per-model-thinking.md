# Map the vLLM thinking switch per served model

- **Date:** 2026-09-03
- **Type:** fix
- **Scope:** `vllm_openai_chat`, `docs`
- **PR:** [#PENDING](https://github.com/Prism-Shadow/agenthub/pull/PENDING)

[中文版](2026-09-03-vllm-per-model-thinking.zh.md)

## What changed

- `vllm-openai-chat` now picks the `chat_template_kwargs` shape from the served model id, matched
  as a lowercased substring, instead of sending `enable_thinking` to every model.
- Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Qwen3.5-0.8B and Qwen3.5-9B keep the `enable_thinking`
  boolean.
- Qwen3.8-27B disables thinking with `enable_thinking: false` and selects its adaptive modes with
  `reasoning_effort`, which accepts only `low`/`medium`/`xhigh`, so `high` and `max` clamp to
  `xhigh`.
- DeepSeek-V4-Pro, DeepSeek-V4-Flash and DeepSeek-V4-Flash-Vision-Exp pair `thinking: true` with
  `reasoning_effort`, which accepts only `low`/`high`/`max`, so `medium` and `xhigh` clamp to
  `high`; `none` sends no `chat_template_kwargs` at all, which is how those templates read as off.
- A model outside the table falls back to the `enable_thinking` boolean.
