# The Responses-protocol message transforms match the GPT client

- **Date:** 2026-08-20
- **Type:** refactor
- **Scope:** `openai_responses`, `minimax_m3`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[中文版](2026-08-20-responses-input-chain.zh.md)

## What changed

- `openai_responses` and `minimax_m3` build their input items through a single `if`/`elif` chain
  over the content-item type and flush buffered text once, at the end of each message — the shape
  `gpt5_6` already used. The early-`continue` guards for text and images went away with the flush
  that used to run before every top-level item.
- Replay order changed for a message that mixes text with a top-level item (a reasoning item, a
  function call, a function call output). The text now follows the item instead of preceding it,
  and two text spans separated by such an item arrive merged into one message entry. An assistant
  turn of `[text, tool_call]` replays as `[function_call, message]`, which is what `gpt5_6` has
  always sent. A message carrying only text and images is unaffected.

## Replay order by message shape

| Message | Before | After |
| --- | --- | --- |
| `[text, tool_call]` | `message`, `function_call` | `function_call`, `message` |
| `[text, thinking, text, tool_call]` | `message`, `reasoning`, `message`, `function_call` | `reasoning`, `function_call`, `message` |
| `[text, image_url]` | `message` | `message` |
| `[thinking, text]` | `reasoning`, `message` | `reasoning`, `message` |
