# Map the vLLM thinking switch per served model

- **Date:** 2026-09-03
- **Type:** fix
- **Scope:** `openai_chat_vllm_adapter`, `docs`
- **PR:** [#198](https://github.com/Prism-Shadow/agenthub/pull/198)

[中文版](2026-09-03-vllm-per-model-thinking.zh.md)

## What changed

- `openai-chat-vllm-adapter` now picks the `chat_template_kwargs` shape from the served model
  id, matched as a lowercased substring, instead of sending `enable_thinking` to every model.
- Qwen3.6-35B-A3B, Qwen3.5-0.8B and Qwen3.5-9B keep the `enable_thinking` boolean.
- Qwen3.8-27B and Qwen3.8-Flash-Next ship the same chat template byte for byte and share one
  profile: thinking off is `enable_thinking: false`, and the adaptive modes are selected with
  `reasoning_effort`, which the template accepts only as `low`/`medium`/`xhigh`, so `high` and
  `max` clamp to `xhigh`.
- DeepSeek-V4-Pro and DeepSeek-V4-Flash pair `thinking: true` with `reasoning_effort`. Their
  encoding module asserts `reasoning_effort in ['max', None, 'high']`, so `low` through `xhigh`
  all send `high` — `low` would fail the request outright — and because that module branches on
  `max` alone, every level below `max` renders the same prompt.
- DeepSeek-V4-Flash-Vision-Exp has a different copy of that module, which accepts
  `low`/`high`/`max`, so it keeps the finer scale on its own profile: `low` sends `low`, and
  `medium` and `xhigh` clamp to `high`.
- For all three DeepSeek models, `none` sends no `chat_template_kwargs` at all, which is how they
  read as off.
- A model outside the table falls back to the `enable_thinking` boolean.
- New reference section `llmsdk_docs/openai_chat_vllm_adapter/`: the five Qwen chat templates
  snapshotted byte-identical from Hugging Face with their source URL, snapshot date, license and
  checksum, plus a note on the three DeepSeek V4 models, which publish no chat template and
  define their parameters in an `encoding/encoding_dsv4.py` module instead. The section states
  the convention that a model added to the adapter has its template added here too.
