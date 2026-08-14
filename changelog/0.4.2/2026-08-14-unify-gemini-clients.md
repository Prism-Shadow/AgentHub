# Unified Gemini client with a single sampling-parameter contract

The `gemini3` and `gemini3_7` client folders are merged into one `gemini3_7` client per
language, named for the newest generation it serves. Outside of the parameter contract the
two implementations had already converged line for line, so the merge folds the remaining
differences into one file:

- **Temperature is now rejected for the whole family.** The 3.6 generation deprecated the
  sampling parameters, and the unified client applies that contract to every Gemini model:
  setting `temperature` raises `UnsupportedParameterError` on `gemini-3.5-flash`, the 3.1
  image/TTS models, and the 2.5 series too, where the older client used to pass it through.
- The per-model thinking-level tables are merged into one: the 2.5 series still drops the
  `thinking_level` parameter entirely, image models keep `minimal`/`high`, pro generations
  keep their reduced sets, and `gemini-3.7-*` keeps `low`/`medium`/`high`.
- The function-call id round-trip introduced with 3.7 support now covers the whole family:
  every `FunctionResponse` carries the call id and function name, and pre-id histories
  degrade to the previous name-only form.
- Routing folds to one branch: `gemini-3`/`gemini-embedding` client types (which the
  `gemini-3.7`, `gemini-3.6`, and `gemini-3.5-flash-lite` spellings all contain) construct
  the unified client, so every previously accepted client type keeps working. Registry
  entries for the 3.x text, image, TTS, and embedding models point at `gemini-3.7`, as the
  Claude 4.7–5 entries point at `claude-5`.
- `llmsdk_docs/gemini3/` stays as the wire-protocol reference for the older generation;
  client folders merge, docs snapshots do not.
