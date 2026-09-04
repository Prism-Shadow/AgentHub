# 按所服务的模型映射 vLLM 思考开关

- **Date:** 2026-09-03
- **Type:** fix
- **Scope:** `openai_chat_vllm_adapter`, `docs`
- **PR:** [#198](https://github.com/Prism-Shadow/agenthub/pull/198)

[English](2026-09-03-vllm-per-model-thinking.md)

## 变更内容

- `openai-chat-vllm-adapter` 改为按所服务的模型 id（转小写后按子串匹配）选择
  `chat_template_kwargs` 的形态，不再对所有模型一律发送 `enable_thinking`。
- Qwen3.6-35B-A3B、Qwen3.5-0.8B 与 Qwen3.5-9B 仍使用 `enable_thinking` 布尔开关。
- Qwen3.8-27B 与 Qwen3.8-Flash-Next 的 chat template 逐字节相同，因此共用一份 profile：以
  `enable_thinking: false` 关闭思考，并以 `reasoning_effort` 选择自适应模式；该 template 只接受
  `low`/`medium`/`xhigh`，因此 `high` 与 `max` 收敛到 `xhigh`。
- DeepSeek-V4-Pro 与 DeepSeek-V4-Flash 以 `thinking: true` 搭配 `reasoning_effort`。二者的 encoding
  模块断言 `reasoning_effort in ['max', None, 'high']`，因此 `low` 到 `xhigh` 一律发送 `high`——发送
  `low` 会直接导致请求失败；又因为该模块只对 `max` 分支，`max` 以下各档渲染出的 prompt 完全一致。
- DeepSeek-V4-Flash-Vision-Exp 的该模块是另一份副本，接受 `low`/`high`/`max`，因此单列一份 profile 并
  保留更细的档位：`low` 发送 `low`，`medium` 与 `xhigh` 收敛到 `high`。
- 三个 DeepSeek 模型在 `none` 档均完全不发送 `chat_template_kwargs`，这正是它们判定为关闭的方式。
- 不在表内的模型回退到 `enable_thinking` 布尔开关。
- 新增参考文档目录 `llmsdk_docs/openai_chat_vllm_adapter/`：五个 Qwen 模型的 chat template 按字节
  原样自 Hugging Face 快照，并记录来源 URL、快照日期、许可证与校验和；另附一篇说明，交代三个
  DeepSeek V4 模型并不发布 chat template，其参数定义在 `encoding/encoding_dsv4.py` 模块中。该目录
  同时确立约定：向该 client 新增模型时，一并把它的 chat template 收录于此。
