# Responses input items keep the order the assistant produced them in

- **Date:** 2026-08-21
- **Type:** fix
- **Scope:** `deepseek_v4`, `openai_responses`

[中文版](2026-08-21-responses-input-order.zh.md)

## What changed

- `DeepSeekV4Client` and `OpenaiResponsesClient` flush the message text collected so far
  before appending a reasoning, `function_call` or `function_call_output` item, so an assistant
  turn that spoke before calling a tool replays as text, then call, then output.
- Before the fix the assistant message was appended after the items it preceded, which DeepSeek
  answered with `400 No tool output found for tool call <id>` on the next turn.
