# Responses 输入条目保持助手产出时的顺序

- **Date:** 2026-08-21
- **Type:** fix
- **Scope:** `deepseek_v4`, `openai_responses`
- **PR:** [#185](https://github.com/Prism-Shadow/agenthub/pull/185)

[English](2026-08-21-responses-input-order.md)

## 变更内容

- `DeepSeekV4Client` 与 `OpenaiResponsesClient` 在追加 reasoning、`function_call` 或
  `function_call_output` 条目之前，先把已经收集的消息正文落盘，因此「先说话再调工具」的助手轮次
  会按正文、调用、结果的顺序回放。
- 修复前助手消息被排到它本应领先的条目之后，DeepSeek 会在下一轮返回
  `400 No tool output found for tool call <id>`。
