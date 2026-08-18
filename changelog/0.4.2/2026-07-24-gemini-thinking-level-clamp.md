# Gemini 3 clients clamp thinking levels to what each model supports

- **Date:** 2026-07-24
- **Type:** fix
- **Scope:** `gemini3`, `tests`
- **PR:** [#166](https://github.com/Prism-Shadow/agenthub/pull/166)

[中文版](2026-07-24-gemini-thinking-level-clamp.zh.md)

## What changed

- `gemini3/` clients (Python and TypeScript) no longer forward a thinking level the
  target model rejects. `_convert_thinking_level` clamps the mapped Gemini level to the
  closest level the model supports, rounding up on ties, instead of letting the API fail
  with `400 Thinking level MINIMAL is not supported for this model`.
- Per-model support matrix:
  - `gemini-3.1-pro*`: `low`, `medium`, `high` — NONE degrades to `low` (it previously
    mapped to `minimal` and always errored).
  - `gemini-3-pro*`: `low`, `high` — NONE degrades to `low`, MEDIUM rounds up to `high`.
  - `*-image` models (`gemini-3.1-flash-image`, `gemini-3-pro-image`, and their
    `-preview`/`-lite` variants): `minimal`, `high` only — LOW degrades to `minimal`,
    MEDIUM rounds up to `high`. The `-image` check runs first, so `gemini-3-pro-image`
    gets the image set, not the `gemini-3-pro` one.
  - Any other `*-pro*` model, a generic branch so future pro generations do not silently
    regress to the four-level default: `low`, `medium`, `high`.
  - `gemini-2.5*`: the parameter is dropped entirely and the model keeps its default
    dynamic thinking. The live API rejects every `thinking_level` value for the 2.5
    series ("Thinking level is not supported for this model"), contrary to the vendor
    table.
  - Flash text models keep the full `minimal`/`low`/`medium`/`high` set and are
    unchanged (NONE still maps to `minimal`).
- Degradation is silent: MEDIUM rounds up to `high` on `gemini-3-pro` and on image
  models, and NONE maps to `low` on 3.1-pro, so a caller who asked for no thinking is
  still billed for some.
- `llmsdk_docs/gemini3/docs/thinking.md`: the outdated three-column
  supported/not-supported table was replaced with the live page's per-model table, adding
  the gemini-3.6/3.5 rows, the `gemini-3-pro-preview` low/high row, and the 2.5 rows.
- Offline regression tests pin the clamp matrix in both languages —
  `src_py/tests/test_thinking_level_mapping.py`,
  `src_ts/tests/thinking-level-mapping.test.ts`.

## Drive-by fixes in the same PR

- `src_py/README.md` code blocks reformatted for ruff 0.16, which started format-checking
  markdown code blocks and broke `make lint` on every branch. Formatting only.
- The flaky system-prompt e2e ("kitten … meow") prompt was hardened in both languages;
  the old prompt only implied the word, and models roleplayed around it.
