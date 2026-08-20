# Responses 协议的消息转换写成一条 if/else 链

- **Date:** 2026-08-20
- **Type:** refactor
- **Scope:** `openai_responses`, `minimax_m3`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[English](2026-08-20-responses-input-chain.md)

## 变更内容

- `openai_responses` 与 `minimax_m3` 改为按 content item 的类型走一条 `if`/`elif` 链构造输入项，
  也就是 `gpt5_6` 原本的写法，替换掉原先「文本与图片先 `continue`、顶层项再走第二条链」的结构。
- 那段用于让顶层项（reasoning、function call、function call output）排在其前方文本之后的缓冲文本
  flush，移到了循环开头作为一个前置判断，因此两个转换产出的输入项与顺序都和改动前一致。`gpt5_6`
  仍然在每条消息结束时才 flush。
