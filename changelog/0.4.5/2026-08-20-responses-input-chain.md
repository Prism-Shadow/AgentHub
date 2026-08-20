# The Responses-protocol message transforms read as one if/else chain

- **Date:** 2026-08-20
- **Type:** refactor
- **Scope:** `openai_responses`, `minimax_m3`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[中文版](2026-08-20-responses-input-chain.zh.md)

## What changed

- `openai_responses` and `minimax_m3` build their input items through a single `if`/`elif` chain
  over the content-item type, the shape `gpt5_6` already used, replacing the early-`continue`
  guards for text and images that were followed by a second chain for the top-level items.
- The buffered-text flush that keeps a top-level item (reasoning, function call, function call
  output) behind the text it followed moved to a guard at the top of the loop, so both transforms
  emit the same input items in the same order as before. `gpt5_6` keeps its own flush at the end
  of each message.
