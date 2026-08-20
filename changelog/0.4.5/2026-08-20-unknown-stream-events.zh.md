# 未知的流式输出默认静默跳过，AGENTHUB_DEBUG 开启时才抛出

- **Date:** 2026-08-20
- **Type:** fix
- **Scope:** `utils`, `claude5`, `ant_messages`, `openai_responses`, `gemini3_7`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[English](2026-08-20-unknown-stream-events.md)

## 变更内容

- 流式 client 不再因为遇到自己不认识的输出而中断整个流。`claude5`、`ant_messages`、
  `openai_responses`、`gpt5_6` 与 `minimax_m3` 把未知事件标记为 `unused`，`gemini3_7` 跳过未知的
  content part。此前网关向 OpenAI Responses 流中注入 `{"type": "ping", "cost": "@"}` 时，会在生成
  途中抛出 `ValueError`/`Error`：`Unknown output: {"type":"ping","cost":"@"}`。
- 两种语言的 `utils` 新增 `is_debug_enabled()` / `isDebugEnabled()`，逐事件读取 `AGENTHUB_DEBUG`。
  将其设为 `0`、`false`、`no`、`off` 以外的任意值，未识别的事件或 part 会重新抛出，报错信息与此前
  一致。
- 各协议已知的无操作事件名单保持不变，因此 client 本就认识的心跳（Anthropic Messages 的 `ping`、
  OpenAI Responses 的 `keepalive`）仍按名字识别，而不是靠兜底跳过。
  [0.4.3 的修复](../0.4.3/2026-08-18-gateway-heartbeats.zh.md)只让每个家族认识了自己那套拼法，这正是
  中转站发出的 Anthropic 风格 ping 会落到 Responses client 守卫上的原因。
- 两种语言的离线测试让心跳、跨协议拼法、协议内部的未知事件、网关错误帧以及未知的 Gemini part 流过
  全部 client，并钉住在设置 `AGENTHUB_DEBUG` 后每一种都会抛出：
  `src_py/tests/test_unknown_events.py`、`src_ts/tests/unknown-events.test.ts`。

## 各协议的处理方式

| 协议 | Client | 未知输出 | 设置 `AGENTHUB_DEBUG` 后 |
| --- | --- | --- | --- |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | 事件标记为 `unused` | 抛出 |
| Anthropic Messages | `claude5`, `ant_messages` | 事件标记为 `unused` | 抛出 |
| Gemini generateContent | `gemini3_7` | 跳过该 content part | 抛出 |
| OpenAI Chat Completions | `openai_chat`, `deepseek_v4`, `glm5_3`, `kimi_k3` | 不带 `choices` 的 chunk 不产生事件 | 不变 |
