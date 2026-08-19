# Thinking levels gain a max tier

- **Date:** 2026-08-19
- **Type:** feature
- **Scope:** `types`, `claude5`, `deepseek_v4`, `gemini3_7`, `tests`
- **PR:** [#176](https://github.com/Prism-Shadow/agenthub/pull/176)

[中文版](2026-08-19-thinking-level-max.zh.md)

## What changed

- `ThinkingLevel` gained `MAX`, above `XHIGH`: the ladder is now `NONE`, `LOW`, `MEDIUM`,
  `HIGH`, `XHIGH`, `MAX` in both languages. Every client maps it to the deepest effort its
  vendor accepts and degrades silently where no such level exists, so no client raises for
  a thinking level.
- `deepseek_v4` re-mapped its efforts to DeepSeek's current vocabulary, which accepts
  `low`/`high`/`max` and maps `medium` and `xhigh` onto `high` server-side: `LOW` now sends
  `low` (it sent `high`), `XHIGH` now sends `high` (it sent `max`), and `MAX` sends `max`.
- `gemini3_7` marks a chunk as `unused` when nothing was read out of it — no content items,
  no usage, no finish reason — instead of inferring a gateway heartbeat from the absence of
  candidates and usage on the raw chunk. Streams that omit usage are unaffected either way.
- The playground's thinking-level selector gained a `Max` option, and `XHigh` is described
  as the extended tier rather than the maximum one.
- `gemini3_7` documented which models return thought summaries for `thinking_summary`:
  probed live on 2026-08-19, `gemini-2.5-flash` and `gemini-3.1-pro-preview` return thought
  parts while `gemini-3.5-flash`, `gemini-3.6-flash`, and `gemini-3.7-flash` return none.
  The request already carries `include_thoughts`; no client change was needed.

## Thinking level mapping

| AgentHub | OpenAI Responses | Claude 4.6+ | Gemini | DeepSeek V4 | GLM-5.3 | GLM-5.2 | Kimi K3 | MiniMax M3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `NONE` | `none` | thinking omitted | `minimal` | thinking disabled | `low` | thinking disabled | `low` | `none` |
| `LOW` | `low` | `low` | `low` | `low` | `low` | `low` | `low` | `low` |
| `MEDIUM` | `medium` | `medium` | `medium` | `high` | `high` | `medium` | `high` | `medium` |
| `HIGH` | `high` | `high` | `high` | `high` | `high` | `high` | `high` | `high` |
| `XHIGH` | `xhigh` | `xhigh` (`high` on 4.6) | `high` | `high` | `max` | `xhigh` | `max` | `high` |
| `MAX` | `max` | `max` | `high` | `max` | `max` | `max` | `max` | `high` |

Gemini levels then clamp to the set the target model accepts, and GLM-5.3 and Kimi K3 cannot
disable thinking, so `NONE` rides on their lightest effort.

## Reference documentation

- `llmsdk_docs/deepseek_v4/docs/thinking-mode.md` re-synced from
  https://api-docs.deepseek.com/guides/thinking_mode: `reasoning_effort` accepts
  `low`/`high`/`max`, and the compatibility mapping sends `medium` and `xhigh` to `high`.
