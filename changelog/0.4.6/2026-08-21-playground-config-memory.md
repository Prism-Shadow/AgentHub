# The playground remembers its configuration across a reload

- **Date:** 2026-08-21
- **Type:** feature
- **Scope:** `integration`, `tests`
- **PR:** [#184](https://github.com/Prism-Shadow/agenthub/pull/184)

[中文版](2026-08-21-playground-config-memory.zh.md)

## What changed

- The playground writes the configuration panel to `localStorage` under
  `agenthub.playground.config` on every edit and restores it when the page loads: model, client
  type, API key, base URL, extra headers, thinking level, thinking summary, tool choice, system
  prompt, tools, and trace ID. The text fields are stored as typed, so an unfinished JSON edit
  comes back the way it was left.
- A model the dropdown does not list — one added by **List models**, or typed in as a custom id —
  is restored as a custom entry carrying its client type.
- The API key is stored with the rest of the panel, so it stays in the browser profile that opened
  the playground until the field is cleared.
- A browser that refuses `localStorage` still runs the playground; it just starts from the
  defaults every time.
