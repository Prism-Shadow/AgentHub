# Responses 协议的消息转换与 GPT client 保持一致

- **Date:** 2026-08-20
- **Type:** refactor
- **Scope:** `openai_responses`, `minimax_m3`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[English](2026-08-20-responses-input-chain.md)

## 变更内容

- `openai_responses` 与 `minimax_m3` 改为按 content item 的类型走一条 `if`/`elif` 链构造输入项，
  并且只在每条消息结束时 flush 一次缓冲文本——也就是 `gpt5_6` 原本的写法。文本与图片的前置
  `continue`，连同原先在每个顶层项之前执行的那次 flush，一并去掉。
- 当一条消息把文本和顶层项（reasoning、function call、function call output）混在一起时，重放顺序
  发生变化：文本改为排在该顶层项之后，且被顶层项隔开的两段文本会合并进同一个 message 条目。
  assistant 的 `[text, tool_call]` 会重放为 `[function_call, message]`，这正是 `gpt5_6` 一直以来
  发送的形态。只含文本与图片的消息不受影响。

## 各消息形态的重放顺序

| 消息 | 改动前 | 改动后 |
| --- | --- | --- |
| `[text, tool_call]` | `message`、`function_call` | `function_call`、`message` |
| `[text, thinking, text, tool_call]` | `message`、`reasoning`、`message`、`function_call` | `reasoning`、`function_call`、`message` |
| `[text, image_url]` | `message` | `message` |
| `[thinking, text]` | `reasoning`、`message` | `reasoning`、`message` |
