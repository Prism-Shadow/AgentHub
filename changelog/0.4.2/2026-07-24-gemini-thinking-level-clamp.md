# Gemini 3 clients clamp thinking levels to what each model supports

## What changed

- `gemini3/` clients (Python and TypeScript) no longer forward a thinking level the
  target model rejects. `_convert_thinking_level` now clamps the mapped Gemini level to
  the closest level the model supports, rounding up on ties, instead of letting the API
  fail with `400 Thinking level MINIMAL is not supported for this model`.
- Per-model support matrix (from the refreshed official thinking doc, verified live on
  2026-07-24 against the Gemini API):
  - `gemini-3.1-pro*`: `low`, `medium`, `high` — so NONE now degrades to `low` (it
    previously mapped to `minimal` and always errored).
  - `gemini-3-pro*`: `low`, `high` (per docs; the official endpoint has decommissioned
    this model — kept for Vertex) — NONE degrades to `low`, MEDIUM rounds up to `high`.
  - `*-image` models (`gemini-3.1-flash-image`, `gemini-3-pro-image`, and their
    `-preview`/`-lite` variants): `minimal`, `high` only — LOW degrades to `minimal`,
    MEDIUM rounds up to `high`.
  - Flash text models keep the full `minimal`/`low`/`medium`/`high` set and are
    unchanged (NONE still maps to `minimal`).
- The `gemini3_6/` clients are untouched: both routed models (`gemini-3.6-flash`,
  `gemini-3.5-flash-lite`) accept all four levels.

## Capture findings

- `gemini-3.1-pro-preview` + `minimal` → 400 "Thinking level MINIMAL is not supported
  for this model. Please retry with other thinking level."; `low` and `medium` succeed.
- `gemini-3.1-flash-image` + `low`/`medium` → the same 400, confirming the
  minimal/high-only image matrix.
- `gemini-3-pro-preview` is no longer served on the official endpoint ("no longer
  available" for every request), so its `low`/`high` matrix comes from the docs table.
- `gemini-3.1-flash-lite` (absent from the docs table) accepts `minimal`.
- Raw probes: `api_captures/gemini3/thinking_levels.probe.jsonl`.

## Docs

- `llmsdk_docs/gemini3/docs/thinking.md`: replaced the outdated three-column
  supported/not-supported table with the live page's per-model table (adds
  gemini-3.6/3.5 rows and the `gemini-3-pro-preview` low/high row).

## Tests

- New offline regression tests pin the clamp matrix in both languages:
  `src_py/tests/test_thinking_level_mapping.py`,
  `src_ts/tests/thinking-level-mapping.test.ts`.
