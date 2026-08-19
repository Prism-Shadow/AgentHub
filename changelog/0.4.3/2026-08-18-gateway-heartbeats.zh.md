# 所有流式 client 跳过网关的心跳事件

- **Date:** 2026-08-18
- **Type:** fix
- **Scope:** `openai_responses`, `openai_chat`, `ant_messages`, `gemini3_7`, `tests`
- **PR:** [#174](https://github.com/Prism-Shadow/agenthub/pull/174)
- **Issue:** [Prism-Shadow/penguin-harness#286](https://github.com/Prism-Shadow/penguin-harness/issues/286)

[English](2026-08-18-gateway-heartbeats.md)

## 变更内容

- `gpt5_6`、`openai_responses` 与 `minimax_m3` client（Python 与 TypeScript）把 `keepalive`
  流事件识别为已知的无操作事件并静默跳过。位于 Responses 兼容服务前面的网关（one-api 风格代理、
  OpenRouter）会在长生成期间向 SSE 流注入 `{"type": "keepalive", "sequence_number": N}` 心跳；
  此前这些事件会落入未知事件守卫，以 `ValueError`/`Error` 中断整个流：
  `Unknown output: {"type":"keepalive","sequence_number":3}`。
- `openai_chat`、`deepseek_v4`、`glm5_3` 与 `kimi_k3` client 仅在 `choices` 字段有值时才读取它。
  心跳并不是一个 Chat Completions chunk，因此 `choices` 会以 `None`/`undefined` 而非空列表到达，
  此前的长度判断会以 `TypeError: object of type 'NoneType' has no len()` /
  `TypeError: Cannot read properties of undefined (reading 'length')` 中断整个流。
- `claude5` 与 `ant_messages` client 把 Messages API 的 `ping` 事件与 `text`、`thinking`、
  `signature`、`input_json` 一同视为已知的无操作事件。两种语言的 Anthropic SDK 都会在 SSE 层丢弃
  `ping`，因此它只有在网关把它改挂到其他事件上时才会到达转换函数，此前会抛出
  `Unknown output: {"type":"ping"}`。
- `gemini3_7` client 把既无 candidates 又无 usage 的 chunk 标记为 `unused`，流循环直接跳过，
  而不再产生一个空的 `delta` 事件。此前若心跳出现在最后一个 chunk 之后，它会成为流的最后一个事件，
  并触发 `base_client` 的 `Last event must carry usage_metadata` 校验失败。
- 未知事件守卫本身未变：真正未知的事件类型（包括服务方的错误事件）仍会抛出。
- 两种语言的离线回归测试让心跳穿插在正常事件之间、并出现在最后一个事件之后，流过全部流式 client，
  同时钉住未知事件的守卫：`src_py/tests/test_keepalive_events.py`、
  `src_ts/tests/keepalive-events.test.ts`。

## 各协议的心跳处理

| 协议 | Client | 心跳形态 | 处理方式 |
| --- | --- | --- | --- |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | `{"type": "keepalive", "sequence_number": N}` | 已知的无操作事件类型 |
| OpenAI Chat Completions | `openai_chat`, `deepseek_v4`, `glm5_3`, `kimi_k3` | 不带 `choices` 的 chunk | 既无 choices 又无 usage 则不产生事件 |
| Anthropic Messages | `claude5`, `ant_messages` | `{"type": "ping"}` | 已知的无操作事件类型 |
| Gemini generateContent | `gemini3_7` | 既无 candidates 又无 usage 的 chunk | 标记为 `unused`，由流循环跳过 |
