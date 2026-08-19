# Responses 协议 client 跳过网关的 keepalive 心跳

- **Date:** 2026-08-18
- **Type:** fix
- **Scope:** `gpt5_6`, `openai_responses`, `minimax_m3`, `tests`
- **Issue:** [Prism-Shadow/penguin-harness#286](https://github.com/Prism-Shadow/penguin-harness/issues/286)

[English](2026-08-18-responses-keepalive.md)

## 变更内容

- `gpt5_6`、`openai_responses` 与 `minimax_m3` client（Python 与 TypeScript）把
  `keepalive` 流事件识别为已知的无操作事件并静默跳过。位于 Responses 兼容服务前面的网关
  （one-api 风格代理、OpenRouter）会在长生成期间向 SSE 流注入
  `{"type": "keepalive", "sequence_number": N}` 心跳；此前这些事件会落入未知事件守卫，
  以 `ValueError`/`Error` 中断整个流：`Unknown output: {"type":"keepalive","sequence_number":3}`。
- 未知事件守卫本身未变：真正未知的事件类型（包括服务方的错误事件）仍会抛出。
- 两种语言的离线回归测试让 keepalive 心跳穿插在正常 Responses 事件之间流过全部三个
  client，并钉住未知事件的守卫 —— `src_py/tests/test_keepalive_events.py`、
  `src_ts/tests/keepalive-events.test.ts`。
