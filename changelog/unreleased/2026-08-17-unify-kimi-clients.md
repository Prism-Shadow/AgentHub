# Unify the Kimi series clients

- **Date:** 2026-08-17
- **Type:** refactor
- **Scope:** `kimi_k3`, `auto_client`, `tests`
- **PR:** [#171](https://github.com/Prism-Shadow/agenthub/pull/171)
- **Breaking:** yes — the `kimi_k2_6` client folder was removed; deep imports of `KimiK2_6Client` stop working.

[中文版](2026-08-17-unify-kimi-clients.zh.md)

## What changed

- The `kimi_k2_6` client merged into `kimi_k3` (`KimiK3Client`), which now serves the whole
  Kimi K2.5+ series; the `kimi-k2.5`/`kimi-k2.6` client types route there.
- Thinking configuration branches on the model generation: K2.x models keep the
  `extra_body.thinking` config (`disabled` for `ThinkingLevel.NONE`, `enabled`+`keep: all`
  otherwise) while K3 models keep `reasoning_effort` (`NONE` degrades to `low`).
- `tool_choice: "required"` stays K3-only; K2.x models raise `UnsupportedParameterError`
  for it, and forcing a specific tool raises family-wide.
- `trace_id` is sent as `prompt_cache_key` on K2.x models only (K3 context caching is
  automatic).
- The `temperature`/`fast_mode`/`prompt_caching` rejection messages now name the Kimi
  family instead of a single generation.

## Compatibility

- Import `KimiK3Client` from `kimi_k3` instead of `KimiK2_6Client` from `kimi_k2_6`; the
  `kimi-k2.5` and `kimi-k2.6` client-type strings keep working unchanged.
