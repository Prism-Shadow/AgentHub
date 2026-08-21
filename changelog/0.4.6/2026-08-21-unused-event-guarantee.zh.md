# 内部的 "unused" 事件绝不会流到调用方

- **Date:** 2026-08-21
- **Type:** refactor
- **Scope:** `base_client`, `types`, `skills`, `tests`
- **PR:** [#185](https://github.com/Prism-Shadow/agenthub/pull/185)

[English](2026-08-21-unused-event-guarantee.md)

## 变更内容

- `streaming_response` / `streamingResponse` 遇到 `event_type` 为 `unused` 的事件会直接丢弃而不再
  向外抛出，并在设置了 `AGENTHUB_DEBUG` 时报错，于是漏掉自身过滤的客户端会在 CI 里失败，而不是把
  空事件发出去。
- `types.py` / `types.ts` 里的 `EventType` 定义写清了四种取值的含义，以及 `unused` 不会离开客户端。
- 开发 skill 记下两条客户端规则：`unused` 只在客户端内部使用；消息 transform 必须保持内容条目的
  顺序，在追加会成为独立输入条目的项之前先把已收集的正文落盘。
- `test_unknown_events.py` / `unknown-events.test.ts` 在打开 debug 守卫的情况下，让每个客户端跑一遍
  自己协议里的可忽略事件；另有一个故意「漏事件」的客户端覆盖基类两种模式下的行为。
