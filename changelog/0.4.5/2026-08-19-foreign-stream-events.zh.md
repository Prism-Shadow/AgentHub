# 流式 client 跳过协议之外注入的流事件

- **Date:** 2026-08-19
- **Type:** fix
- **Scope:** `utils`, `ant_messages`, `claude5`, `openai_responses`, `tests`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[English](2026-08-19-foreign-stream-events.md)

## 变更内容

- 两种语言的 `utils` 新增 `is_foreign_no_op_event` / `isForeignNoOpEvent`，`claude5`、
  `ant_messages`、`openai_responses`、`gpt5_6` 与 `minimax_m3` client 在未知事件守卫抛出之前
  先向它询问。只有同时满足三个条件的事件才会被跳过：事件类型位于该协议自己的命名空间之外、
  事件类型不指向错误、且没有任何字段持有非空的对象或数组。
- 网关发出的心跳若按另一种协议的拼法命名，不再中断整个流。`{"type": "ping", "cost": "@"}`
  到达 OpenAI Responses client 时，此前会抛出 `ValueError`/`Error`：
  `Unknown output: {"type":"ping","cost":"@"}`。
  [0.4.3 的心跳修复](../0.4.3/2026-08-18-gateway-heartbeats.zh.md)只让每个家族认识了自己那套
  拼法——Anthropic Messages 的 `ping`、OpenAI Responses 的 `keepalive`——因此任一拼法串到另一种
  协议上仍会抛出。
- 协议自身命名空间内的未知事件仍会抛出；指向错误的外来事件
  （`{"type": "gateway_error", "message": "upstream 502"}`）与携带负载的外来事件
  （`{"type": "relay_frame", "data": {"text": "dropped"}}`）同样仍会抛出。
- 两种语言的离线回归测试让外来的网关事件流过 Responses 与 Messages 两族 client，并钉住守卫保留
  的每一类拒绝：`src_py/tests/test_keepalive_events.py`、`src_ts/tests/keepalive-events.test.ts`。

## 各协议拥有的命名空间

| 协议 | Client | 事件类型前缀 |
| --- | --- | --- |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | `response.` |
| Anthropic Messages | `claude5`, `ant_messages` | `message_`, `content_block_` |
