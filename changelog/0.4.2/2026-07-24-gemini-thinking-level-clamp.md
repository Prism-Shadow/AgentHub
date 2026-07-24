# Gemini 3 clients clamp thinking levels to what each model supports

## What changed

- `gemini3/` clients (Python and TypeScript) no longer forward a thinking level the
  target model rejects. `_convert_thinking_level` now clamps the mapped Gemini level to
  the closest level the model supports, rounding up on ties, instead of letting the API
  fail with `400 Thinking level MINIMAL is not supported for this model`. This brings
  `gemini3` into compliance with the design rule documented on
  `UnsupportedParameterError` ("Thinking levels never raise this by design"); it was the
  one client violating it.
- Per-model support matrix (from the refreshed official thinking doc, verified live on
  2026-07-24 against the Gemini API):
  - `gemini-3.1-pro*`: `low`, `medium`, `high` — so NONE now degrades to `low` (it
    previously mapped to `minimal` and always errored).
  - `gemini-3-pro*`: `low`, `high` (per docs; the official endpoint has decommissioned
    this model — kept for Vertex) — NONE degrades to `low`, MEDIUM rounds up to `high`.
  - `*-image` models (`gemini-3.1-flash-image`, `gemini-3-pro-image`, and their
    `-preview`/`-lite` variants): `minimal`, `high` only — LOW degrades to `minimal`,
    MEDIUM rounds up to `high`. The `-image` check runs first, so `gemini-3-pro-image`
    gets the image set, not the `gemini-3-pro` one.
  - Any other `*-pro*` model (a generic branch, so future pro generations don't silently
    regress to the four-level default): `low`, `medium`, `high`.
  - `gemini-2.5*`: the vendor table claims `low`/`medium`/`high`, but the live API
    rejects **every** `thinking_level` value for the 2.5 series ("Thinking level is not
    supported for this model") — the capture outranks the docs, so the parameter is
    dropped entirely and the model keeps its default dynamic thinking. Reachable only
    via an explicit `client_type="gemini-3"` override or direct `Gemini3Client`
    construction; name-based routing never sends 2.5 models here.
  - Flash text models keep the full `minimal`/`low`/`medium`/`high` set and are
    unchanged (NONE still maps to `minimal`).
- Degradation is silent by design: the alternative to clamping is a hard 400, and the
  repo-wide rule is that thinking levels never raise. Two clamps round *up*
  (MEDIUM → `high` on `gemini-3-pro` and on image models) and NONE → `low` on
  3.1-pro bills thinking a caller asked to avoid — accepted cost, recorded here since
  the client has no logging infrastructure to surface it at runtime.
- The `gemini3_6/` clients are untouched: both routed models (`gemini-3.6-flash`,
  `gemini-3.5-flash-lite`) accept all four levels.

## Capture findings

- `gemini-3.1-pro-preview` + `minimal` → 400 "Thinking level MINIMAL is not supported
  for this model. Please retry with other thinking level."; `low` and `medium` succeed.
- `gemini-3.1-flash-image` + `low`/`medium` → the same 400, confirming the
  minimal/high-only image matrix.
- `gemini-2.5-pro`/`gemini-2.5-flash`/`gemini-2.5-flash-lite` + any level (including
  the officially-listed `low`) → 400 "Thinking level is not supported for this model.",
  contradicting the vendor table's 2.5 rows.
- `gemini-3-pro-preview` is no longer served on the official endpoint ("no longer
  available" for every request), so its `low`/`high` matrix comes from the docs table.
- `gemini-3.1-flash-lite` (absent from the docs table) accepts `minimal`.
- Raw probes: `api_captures/gemini3/thinking_levels.probe.jsonl`.

## Docs

- `llmsdk_docs/gemini3/docs/thinking.md`: replaced the outdated three-column
  supported/not-supported table with the live page's per-model table (adds
  gemini-3.6/3.5 rows, the `gemini-3-pro-preview` low/high row, and the 2.5 rows —
  which the capture above shows are not actually honored by the API).

## Tests

- Offline regression tests pin the clamp matrix in both languages (18 mirrored cases
  plus two config-level assertions that the clamped/dropped value is what
  `transform_uni_config_to_model_config` actually emits):
  `src_py/tests/test_thinking_level_mapping.py`,
  `src_ts/tests/thinking-level-mapping.test.ts`.

## Drive-by fixes in the same PR

- `src_py/README.md` code blocks reformatted for ruff 0.16: the Makefile lints with
  unpinned `uvx ruff`, and 0.16.0 started format-checking markdown code blocks, which
  broke `make lint` (and CI) on every branch. Formatting-only.
- The flaky system-prompt e2e ("kitten … meow") prompt hardened in both languages: the
  old prompt only implied the word and models roleplayed around it
  (`deepseek-v4-flash`, `gemini-3.6-flash:vertex` flaked in consecutive CI runs).
