# Unify the Claude series clients

- **Date:** 2026-08-17
- **Type:** refactor
- **Scope:** `claude5`, `auto_client`, `tests`
- **PR:** [#171](https://github.com/Prism-Shadow/agenthub/pull/171)
- **Breaking:** yes — the `claude4_6` client folder was removed (deep imports of `Claude4_6Client` stop working), and `temperature` on Claude 4.6 models now raises `UnsupportedParameterError` instead of being passed through.

[中文版](2026-08-17-unify-claude-clients.zh.md)

## What changed

- The `claude4_6` client merged into `claude5` (`Claude5Client`), which now serves the whole
  Claude 4.6+ series; the `claude-*-4-6` client types route there.
- `temperature` raises `UnsupportedParameterError` for the whole family (the API dropped it
  starting with the 4.7 generation; the unified client rejects it on 4.6 as well).
- `ThinkingLevel.XHIGH` degrades to `output_config.effort: "high"` on Claude 4.6 models
  (no xhigh effort there) and stays `"xhigh"` on 4.7 and later.
- `thinking_summary` maps to `thinking.display: "summarized"` for the whole family
  (verified live: Claude 4.6 accepts the `display` field).
- `fast_mode` raises on Claude 4.6 models, matching the API's model support.

## Compatibility

- Import `Claude5Client` from `claude5` instead of `Claude4_6Client` from `claude4_6`; the
  `claude-4-6` client-type string keeps working unchanged.
- Requests that set `temperature` on Claude 4.6 models must drop the parameter.
