# deepseek-v4-flash-vision-exp reads images, in a prompt and in a tool result

- **Date:** 2026-08-21
- **Type:** feature
- **Scope:** `deepseek_v4`, `registry`, `llmsdk_docs`, `tests`
- **PR:** [#185](https://github.com/Prism-Shadow/agenthub/pull/185)

[中文版](2026-08-21-deepseek-vision.zh.md)

## What changed

- `deepseek-v4-flash-vision-exp` joined the registry on the official API with `Text, Image`
  input, a 1M context window, and the `deepseek-v4-flash` price (¥1.5 / ¥4.5 per million
  tokens off-peak, ¥0.05 on a cache hit).
- `DeepSeekV4Client` sends an `image_url` item as an `input_image` content part and a
  `tool_result` image as an `input_image` inside `function_call_output`, so a tool can hand the
  model a picture it produced. Both an HTTP(S) URL and a base64 data URL are accepted.
- A model without vision refuses an image item with
  `DeepSeek <model> does not support image inputs.` rather than sending one: Chat Completions
  answers `400` and the Responses API silently substitutes placeholder text.
- `deepseek-v4-flash-vision-exp` replaced `deepseek-v4-flash` as the official DeepSeek entry in
  both E2E model lists, with image understanding enabled; the three protocol-mode entries stay
  on `deepseek-v4-flash`.
- `llmsdk_docs/deepseek_v4/` gained snapshots of the vision and Responses API guides, and its
  quickstart lists the third model.
