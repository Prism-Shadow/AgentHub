# Responses 输入条目保持助手产出时的顺序

- **Date:** 2026-08-21
- **Type:** fix
- **Scope:** `deepseek_v4`, `openai_responses`, `gpt5_6`, `minimax_m3`, `tests`
- **PR:** [#185](https://github.com/Prism-Shadow/agenthub/pull/185)

[English](2026-08-21-responses-input-order.md)

## 变更内容

- 四个 Responses 协议客户端 —— `DeepSeekV4Client`、`OpenaiResponsesClient`、`GPT5_6Client`
  与 `MiniMaxM3Client` —— 在追加 reasoning、`function_call` 或 `function_call_output` 条目之前，
  先把已经收集的消息正文落盘，因此「先说话再调工具」的助手轮次会按正文、调用、结果的顺序回放。
- 修复前助手消息被排到它本应领先的条目之后，DeepSeek 会在下一轮返回
  `400 No tool output found for tool call <id>`。
- `test_message_order.py` / `message-order.test.ts` 用「思考 + 正文 + 工具调用」的一轮对话钉住每个
  客户端产出的顺序：Responses 客户端保持三者为顺序排列的独立条目，Anthropic 与 Gemini 客户端保持
  为同一条消息内按序排列的块，Chat Completions 客户端保持正文加 `tool_calls` 的形状（该协议本就
  无法表达交错）。
