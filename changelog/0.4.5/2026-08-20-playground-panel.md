# The playground panel groups its fields and starts in debug mode

- **Date:** 2026-08-20
- **Type:** feature
- **Scope:** `integration`, `tests`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[中文版](2026-08-20-playground-panel.zh.md)

## What changed

- The configuration panel reads as two labelled sections instead of two anonymous three-column
  grids. **Connection** holds Model, API Key, Base URL and Extra Headers; **Generation** holds
  Thinking Level, Thinking Summary, Tool Choice, Trace ID, and then System Prompt and Tools across
  two columns each. Both sections share one `1 / 2 / 4` column grid, so each row fills out rather
  than trailing off.
- `start_playground_server` / `startPlaygroundServer` default `AGENTHUB_DEBUG` to `1`. The
  playground exists to show what a model and its endpoint actually send, so unknown stream output
  raises there instead of being skipped; an `AGENTHUB_DEBUG` already in the environment still wins.
