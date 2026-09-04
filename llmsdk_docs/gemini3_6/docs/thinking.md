# Thinking levels across Gemini 3.x

> Source: https://ai.google.dev/gemini-api/docs/thinking (snapshot 2026-08-13)

Thinking is controlled with the `thinking_level` enum in the generation config. Supported
values and defaults per model:

| Model                       | Default thinking | Supported levels             |
| --------------------------- | ---------------- | ---------------------------- |
| gemini-3.8-flash            | medium           | low, medium, high            |
| gemini-3.7-flash            | medium           | low, medium, high            |
| gemini-3.6-flash            | medium           | minimal, low, medium, high   |
| gemini-3.5-flash            | medium           | minimal, low, medium, high   |
| gemini-3.5-flash-lite       | minimal          | minimal, low, medium, high   |
| gemini-3.1-pro-preview      | high             | low, medium, high            |
| gemini-3.1-flash-lite-image | minimal          | minimal, high                |
| gemini-3-flash-preview      | high             | minimal, low, medium, high   |
| gemini-3-pro-preview        | high             | low, high                    |

Notes:

- Thought summaries are requested via the thinking config (`include_thoughts` in the
  google-genai SDK). A thought block may contain **only a signature with no summary** for
  simple requests or thought content types without text summaries.
- When managing conversation state yourself (stateless mode), you **must** resend all
  thought blocks and thought signatures exactly as they were received to maintain
  reasoning continuity. See [../../gemini3/docs/thought-signatures.md](../../gemini3/docs/thought-signatures.md).
- Thought tokens are reported in usage metadata (`thoughts_token_count` in the SDK).
