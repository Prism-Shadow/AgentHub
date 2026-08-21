# Responses input items keep the order the assistant produced them in

- **Date:** 2026-08-21
- **Type:** fix
- **Scope:** `deepseek_v4`, `openai_responses`, `gpt5_6`, `minimax_m3`, `tests`
- **PR:** [#185](https://github.com/Prism-Shadow/agenthub/pull/185)

[中文版](2026-08-21-responses-input-order.zh.md)

## What changed

- All four Responses-protocol clients — `DeepSeekV4Client`, `OpenaiResponsesClient`,
  `GPT5_6Client` and `MiniMaxM3Client` — flush the message text collected so far before
  appending a reasoning, `function_call` or `function_call_output` item, so an assistant turn
  that spoke before calling a tool replays as text, then call, then output.
- Before the fix the assistant message was appended after the items it preceded, which DeepSeek
  answered with `400 No tool output found for tool call <id>` on the next turn.
- `test_message_order.py` / `message-order.test.ts` pin the emitted order for every client
  against a turn of thinking, text and a tool call: the Responses clients keep the three as
  separate items in order, the Anthropic and Gemini clients keep them as ordered blocks inside
  one message, and the Chat Completions clients keep their text-plus-`tool_calls` shape, which
  carries no interleaving.
